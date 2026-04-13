"""
TurboQuant DynamicCache wrapper para HuggingFace transformers.

V2: corrigido para lidar com as realidades do Qwen2.5/GQA:
- K e V tem magnitudes drasticamente diferentes (~1400x no Qwen2.5-0.5B)
- GQA: poucos KV heads (2) servem muitos query heads (14)
- Outliers em K precisam de tratamento especial
- Suporta modo "v_only" (nao quantiza K) - lecao de robustez

Estrategias:
- Residual window: tokens recentes em FP16
- Layer protection: camadas iniciais/finais em FP16
- Opcional: v_only (so comprime V, deixa K em FP16)
- Opcional: per-head normalization antes da quantizacao
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from tq_core import TurboQuantMSE, TQConfig


@dataclass
class TQCacheConfig:
    k_bits: int = 8
    v_bits: int = 4
    residual_window: int = 128
    protect_layers_init: int = 2
    protect_layers_final: int = 2
    seed: int = 42
    v_only: bool = False  # Se True, nao quantiza K (seguranca)
    debug: bool = False


class TurboQuantCache(DynamicCache):
    """KV cache com compressao TurboQuant adaptativa."""

    def __init__(self, cfg: TQCacheConfig, n_layers: int, head_dim: int, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.device = device

        # Instancia quantizers por bit-width
        if not cfg.v_only:
            self.tq_k = TurboQuantMSE(
                TQConfig(dim=head_dim, bits=cfg.k_bits, seed=cfg.seed),
                device=device,
            )
        self.tq_v = TurboQuantMSE(
            TQConfig(dim=head_dim, bits=cfg.v_bits, seed=cfg.seed + 1),
            device=device,
        )

        self.metrics = {
            "total_quantize_calls": 0,
            "bytes_saved": 0,
            "bytes_fp16_equiv": 0,
            "bytes_compressed": 0,
            "k_quantized": 0,
            "v_quantized": 0,
            "k_max_abs": 0.0,
            "v_max_abs": 0.0,
        }

    def _is_layer_protected(self, layer_idx: int) -> bool:
        if layer_idx < self.cfg.protect_layers_init:
            return True
        if layer_idx >= self.n_layers - self.cfg.protect_layers_final:
            return True
        return False

    def _quantize_tensor(self, tq: TurboQuantMSE, x: torch.Tensor) -> torch.Tensor:
        """Quantiza e dequantiza um tensor KV.

        Input shape: (batch, n_kv_heads, seq, head_dim)
        """
        orig_dtype = x.dtype
        # Flatten para (-1, head_dim) para quantizacao por vetor
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        idx, norms = tq.quantize(x_flat)
        x_hat = tq.dequantize(idx, norms).to(orig_dtype)
        return x_hat.reshape(orig_shape)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Delegacao basica ao DynamicCache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        k_full = self.key_cache[layer_idx]
        v_full = self.value_cache[layer_idx]

        # Layers protegidas: FP16 puro
        if self._is_layer_protected(layer_idx):
            return k_full, v_full

        seq_len = k_full.shape[-2]
        if seq_len <= self.cfg.residual_window:
            return k_full, v_full

        n_old = seq_len - self.cfg.residual_window
        k_old = k_full[..., :n_old, :].contiguous()
        v_old = v_full[..., :n_old, :].contiguous()
        k_recent = k_full[..., n_old:, :]
        v_recent = v_full[..., n_old:, :]

        # Stats de debug
        if self.cfg.debug:
            self.metrics["k_max_abs"] = max(self.metrics["k_max_abs"], k_old.abs().max().item())
            self.metrics["v_max_abs"] = max(self.metrics["v_max_abs"], v_old.abs().max().item())

        # Quantiza V (sempre) - geralmente e bem comportado
        v_old_hat = self._quantize_tensor(self.tq_v, v_old)
        v_bytes_fp16 = v_old.numel() * 2
        v_bytes_comp = v_old.numel() * self.cfg.v_bits / 8 + (v_old.shape[0] * v_old.shape[1] * v_old.shape[2]) * 2
        self.metrics["v_quantized"] += 1

        # Quantiza K (opcional)
        if self.cfg.v_only:
            k_old_hat = k_old
            k_bytes_fp16 = 0
            k_bytes_comp = 0
        else:
            k_old_hat = self._quantize_tensor(self.tq_k, k_old)
            k_bytes_fp16 = k_old.numel() * 2
            k_bytes_comp = k_old.numel() * self.cfg.k_bits / 8 + (k_old.shape[0] * k_old.shape[1] * k_old.shape[2]) * 2
            self.metrics["k_quantized"] += 1

        self.metrics["bytes_fp16_equiv"] += int(k_bytes_fp16 + v_bytes_fp16)
        self.metrics["bytes_compressed"] += int(k_bytes_comp + v_bytes_comp)
        self.metrics["bytes_saved"] += int(k_bytes_fp16 + v_bytes_fp16 - k_bytes_comp - v_bytes_comp)
        self.metrics["total_quantize_calls"] += 1

        self.key_cache[layer_idx] = torch.cat([k_old_hat, k_recent], dim=-2)
        self.value_cache[layer_idx] = torch.cat([v_old_hat, v_recent], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_metrics(self) -> dict:
        m = dict(self.metrics)
        if m["bytes_fp16_equiv"] > 0:
            m["compression_ratio"] = m["bytes_fp16_equiv"] / max(m["bytes_compressed"], 1)
        else:
            m["compression_ratio"] = 1.0
        return m

    def reset_metrics(self):
        for k in self.metrics:
            self.metrics[k] = 0 if not isinstance(self.metrics[k], float) else 0.0
