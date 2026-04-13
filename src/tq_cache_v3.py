"""
TurboQuant Cache V3 — fix critico: nao re-quantiza tokens ja quantizados.

O bug das versoes anteriores: cada decode step re-quantizava TODOS os tokens
antigos a partir da versao ja dequantizada (aproximada), causando compound error
que rapidamente diverge para NaN.

Fix: rastreia o boundary de quantizacao por layer. Tokens antes do boundary ja
estao quantizados (armazenados como FP16 aproximado). Somente tokens NOVOS que
acabaram de sair da residual window sao quantizados pela primeira vez.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from tq_core_v2 import TurboQuantMixed, TQMixedConfig


@dataclass
class TQCacheV3Config:
    k_bits_low: int = 3
    k_bits_high: int = 4
    v_bits_low: int = 3
    v_bits_high: int = 4
    n_outlier_channels: int = 16
    residual_window: int = 64
    protect_layers_init: int = 2
    protect_layers_final: int = 2
    seed: int = 42
    v_only: bool = True
    deferred: bool = True


class TurboQuantCacheV3(DynamicCache):

    def __init__(self, cfg: TQCacheV3Config, n_layers: int, head_dim: int, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.device = device
        self._prefill_done = False
        self._quant_boundary: dict[int, int] = {}

        if not cfg.v_only:
            self.tq_k = TurboQuantMixed(
                TQMixedConfig(dim=head_dim, bits_low=cfg.k_bits_low, bits_high=cfg.k_bits_high,
                              n_outlier_channels=cfg.n_outlier_channels, seed=cfg.seed),
                device=device,
            )
        self.tq_v = TurboQuantMixed(
            TQMixedConfig(dim=head_dim, bits_low=cfg.v_bits_low, bits_high=cfg.v_bits_high,
                          n_outlier_channels=cfg.n_outlier_channels, seed=cfg.seed + 1),
            device=device,
        )

        self.metrics = {
            "total_quantize_calls": 0,
            "tokens_quantized": 0,
            "tokens_skipped": 0,
            "bytes_fp16_equiv": 0,
            "bytes_compressed": 0,
        }

    def _is_protected(self, layer_idx: int) -> bool:
        if layer_idx < self.cfg.protect_layers_init:
            return True
        if layer_idx >= self.n_layers - self.cfg.protect_layers_final:
            return True
        return False

    def _quantize_slice(self, tq: TurboQuantMixed, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        idx, norms, mask = tq.quantize(flat, calibrate=True)
        hat = tq.dequantize(idx.reshape(-1, shape[-1]), norms.reshape(-1), mask)
        return hat.to(orig_dtype).reshape(shape)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        is_prefill = key_states.shape[-2] > 1
        if is_prefill and not self._prefill_done:
            pass
        elif not self._prefill_done:
            self._prefill_done = True

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if self._is_protected(layer_idx):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if self.cfg.deferred and is_prefill:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        k_full = self.key_cache[layer_idx]
        v_full = self.value_cache[layer_idx]
        seq_len = k_full.shape[-2]

        if seq_len <= self.cfg.residual_window:
            return k_full, v_full

        old_boundary = self._quant_boundary.get(layer_idx, 0)
        new_boundary = seq_len - self.cfg.residual_window

        if new_boundary <= old_boundary:
            return k_full, v_full

        newly_old_v = v_full[..., old_boundary:new_boundary, :].contiguous()
        n_new = new_boundary - old_boundary

        if n_new > 0:
            newly_old_v_hat = self._quantize_slice(self.tq_v, newly_old_v)

            if not self.cfg.v_only:
                newly_old_k = k_full[..., old_boundary:new_boundary, :].contiguous()
                newly_old_k_hat = self._quantize_slice(self.tq_k, newly_old_k)
                self.key_cache[layer_idx] = torch.cat([
                    k_full[..., :old_boundary, :],
                    newly_old_k_hat,
                    k_full[..., new_boundary:, :],
                ], dim=-2)

            self.value_cache[layer_idx] = torch.cat([
                v_full[..., :old_boundary, :],
                newly_old_v_hat,
                v_full[..., new_boundary:, :],
            ], dim=-2)

            self._quant_boundary[layer_idx] = new_boundary

            n_vectors = newly_old_v.reshape(-1, self.head_dim).shape[0]
            self.metrics["total_quantize_calls"] += 1
            self.metrics["tokens_quantized"] += n_new
            self.metrics["tokens_skipped"] += old_boundary
            fp16_bytes = n_vectors * self.head_dim * 2
            dim = self.head_dim
            n_out = self.cfg.n_outlier_channels
            avg_bits = ((dim - n_out) * self.cfg.v_bits_low + n_out * self.cfg.v_bits_high) / dim
            comp_bytes = n_vectors * (dim * avg_bits / 8 + 2)
            self.metrics["bytes_fp16_equiv"] += int(fp16_bytes)
            self.metrics["bytes_compressed"] += int(comp_bytes)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_metrics(self):
        m = dict(self.metrics)
        if m["bytes_fp16_equiv"] > 0:
            m["compression_ratio"] = m["bytes_fp16_equiv"] / max(m["bytes_compressed"], 1)
        else:
            m["compression_ratio"] = 1.0
        dim = self.head_dim
        n_out = self.cfg.n_outlier_channels
        m["avg_bits"] = ((dim - n_out) * self.cfg.v_bits_low + n_out * self.cfg.v_bits_high) / dim
        return m
