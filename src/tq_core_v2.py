"""
TurboQuant Core V2 — com 2 patches do paper:
  1. Outlier channel splitting (mixed-precision 3.5-bit)
  2. Suporte a deferred quantization (quantiza so no decode, nao no prefill)

Patch 1: identifica top-N canais com maior magnitude por head APOS rotacao,
quantiza eles com bits_high (ex: 4) e o resto com bits_low (ex: 3).
Resultado: media ~3.5 bits/canal com qualidade muito melhor que 3-bit uniforme.

Patch 2: a logica fica no cache wrapper (tq_cache_v2.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from tq_core import _beta_pdf, lloyd_max_beta, make_rotation


@dataclass
class TQMixedConfig:
    dim: int
    bits_low: int = 3
    bits_high: int = 4
    n_outlier_channels: int = 32
    seed: int = 42

    @property
    def avg_bits(self) -> float:
        n_normal = self.dim - self.n_outlier_channels
        return (n_normal * self.bits_low + self.n_outlier_channels * self.bits_high) / self.dim


class TurboQuantMixed:
    """TurboQuant com outlier channel splitting (Patch 1 do paper).

    Apos rotacao ortogonal, identifica os N canais com maior energia
    media e quantiza-os com mais bits. Canais normais usam menos bits.
    Resultado: melhor PPL na mesma media de bits.
    """

    def __init__(self, cfg: TQMixedConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.pi = make_rotation(cfg.dim, seed=cfg.seed, device=device, dtype=torch.float32)
        centroids_low = lloyd_max_beta(cfg.dim, cfg.bits_low, seed=cfg.seed)
        centroids_high = lloyd_max_beta(cfg.dim, cfg.bits_high, seed=cfg.seed)
        self.centroids_low = torch.from_numpy(centroids_low).to(device)
        self.centroids_high = torch.from_numpy(centroids_high).to(device)
        self._outlier_mask = None

    def _detect_outliers(self, y: torch.Tensor) -> torch.Tensor:
        """Detecta canais outlier baseado na energia media apos rotacao.

        Args:
            y: rotated unit vectors, shape (N, dim)

        Returns:
            mask: shape (dim,), True para canais outlier
        """
        energy = (y ** 2).mean(dim=0)  # (dim,)
        _, top_idx = energy.topk(self.cfg.n_outlier_channels)
        mask = torch.zeros(self.cfg.dim, dtype=torch.bool, device=self.device)
        mask[top_idx] = True
        return mask

    @torch.no_grad()
    def quantize(self, x: torch.Tensor, calibrate: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantiza com mixed precision.

        Args:
            x: shape (..., dim)
            calibrate: se True, recalcula outlier mask a partir deste batch

        Returns:
            idx: shape (..., dim), dtype uint8
            norms: shape (...,), dtype float16
            outlier_mask: shape (dim,), dtype bool
        """
        x = x.to(torch.float32)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])

        norms = x_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        unit = x_flat / norms
        y = unit @ self.pi.T

        if calibrate or self._outlier_mask is None:
            self._outlier_mask = self._detect_outliers(y)

        mask = self._outlier_mask
        idx = torch.zeros_like(y, dtype=torch.uint8)

        # Canais normais: bits_low
        normal_ch = ~mask
        if normal_ch.any():
            y_normal = y[:, normal_ch]
            dists = (y_normal.unsqueeze(-1) - self.centroids_low.view(1, -1)).abs()
            idx[:, normal_ch] = dists.argmin(dim=-1).to(torch.uint8)

        # Canais outlier: bits_high
        if mask.any():
            y_outlier = y[:, mask]
            dists = (y_outlier.unsqueeze(-1) - self.centroids_high.view(1, -1)).abs()
            idx[:, mask] = dists.argmin(dim=-1).to(torch.uint8)

        return (
            idx.reshape(orig_shape),
            norms.squeeze(-1).to(torch.float16).reshape(orig_shape[:-1]),
            mask,
        )

    @torch.no_grad()
    def dequantize(self, idx: torch.Tensor, norms: torch.Tensor,
                   outlier_mask: torch.Tensor) -> torch.Tensor:
        """Reconstroi com mixed precision."""
        orig_shape = idx.shape
        idx_flat = idx.reshape(-1, orig_shape[-1]).long()
        norms_flat = norms.reshape(-1)

        y_hat = torch.zeros_like(idx_flat, dtype=torch.float32)

        normal_ch = ~outlier_mask
        if normal_ch.any():
            y_hat[:, normal_ch] = self.centroids_low[idx_flat[:, normal_ch]]
        if outlier_mask.any():
            y_hat[:, outlier_mask] = self.centroids_high[idx_flat[:, outlier_mask]]

        unit_hat = y_hat @ self.pi
        result = unit_hat * norms_flat.unsqueeze(-1).to(torch.float32)
        return result.reshape(orig_shape)
