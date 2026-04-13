"""
TurboQuant Core — Implementação mínima e testável.

Baseado em:
- Paper: arXiv 2504.19874 (ICLR 2026), Algorithm 1 (TurboQuant_mse)
- Referência: referencia/repos/turboquant-py/src/turboquant/turboquant.py
- Referência: referencia/repos/turboquant-pytorch/turboquant.py

Esta versão usa MSE-only (sem QJL), asymmetric K/V, bit-packing.
Decisão baseada nas lições de 6+ implementações independentes:
QJL degrada performance com softmax (ver articles/07, 10, 12).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


# ============================================================
# Lloyd-Max codebook para distribuição Beta(d)
# ============================================================

def _beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """Distribuição Beta induzida por rotação aleatória em S^(d-1).

    f(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)  para x ∈ [-1, 1]

    Em altas dimensões converge para N(0, 1/d). Paper Lemma 1.
    """
    from math import lgamma
    log_norm = lgamma(d / 2) - 0.5 * math.log(math.pi) - lgamma((d - 1) / 2)
    norm = math.exp(log_norm)
    out = np.zeros_like(x)
    mask = (x > -1.0) & (x < 1.0)
    out[mask] = norm * np.power(1.0 - x[mask] ** 2, (d - 3) / 2)
    return out


def lloyd_max_beta(d: int, bits: int, n_iters: int = 200, n_samples: int = 100_000,
                   seed: int = 0) -> np.ndarray:
    """Computa centroids Lloyd-Max ótimos para distribuição Beta(d).

    Resolve k-means 1D contínuo sobre a densidade Beta. Centroids são
    reutilizáveis para qualquer vetor de dimensão d (data-oblivious).

    Args:
        d: dimensão do vetor original (determina a Beta)
        bits: bit-width (2^bits centroids)
        n_iters: iterações Lloyd-Max
        n_samples: amostras para estimar densidade

    Returns:
        centroids ordenados, shape (2^bits,)
    """
    k = 2 ** bits
    rng = np.random.default_rng(seed)

    # Amostra da Beta via rejection na grid + CDF numérica
    grid = np.linspace(-1 + 1e-6, 1 - 1e-6, n_samples)
    pdf = _beta_pdf(grid, d)
    pdf /= pdf.sum()
    samples = rng.choice(grid, size=n_samples, p=pdf)

    # Inicialização: quantis uniformes da Beta
    sorted_samples = np.sort(samples)
    init_idx = np.linspace(0, n_samples - 1, k, dtype=int)
    centroids = sorted_samples[init_idx].astype(np.float64)

    # Iteração Lloyd-Max
    for _ in range(n_iters):
        # Assign: cada amostra para centroid mais próximo
        dists = np.abs(samples[:, None] - centroids[None, :])
        assign = dists.argmin(axis=1)
        # Update: média ponderada por célula
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = assign == i
            if mask.any():
                new_centroids[i] = samples[mask].mean()
            else:
                new_centroids[i] = centroids[i]
        if np.allclose(new_centroids, centroids, atol=1e-8):
            break
        centroids = new_centroids

    return np.sort(centroids).astype(np.float32)


# ============================================================
# Rotação ortogonal via QR
# ============================================================

def make_rotation(d: int, seed: int = 0, device: str = "cpu",
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Gera matriz ortogonal aleatória d×d via QR decomposition.

    Propriedade: Πᵀ Π = I, preserva norma e inner product.
    Armazenamento: (d², 4 bytes) — pode ser reconstruída do seed.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(a)
    # Corrige determinante para rotação pura (det=+1) opcional
    return q.to(device=device, dtype=dtype)


# ============================================================
# TurboQuant MSE Quantizer
# ============================================================

@dataclass
class TQConfig:
    dim: int
    bits: int
    seed: int = 42

    @property
    def n_levels(self) -> int:
        return 2 ** self.bits


class TurboQuantMSE:
    """Quantizador TurboQuant MSE-only (sem QJL).

    Algorithm 1 do paper com as adaptações práticas do V3:
    - MSE-only (sem QJL)
    - Armazena norma por vetor em FP16
    - Codebook pré-computado para Beta(dim)
    """

    def __init__(self, cfg: TQConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.pi = make_rotation(cfg.dim, seed=cfg.seed, device=device, dtype=torch.float32)
        centroids_np = lloyd_max_beta(cfg.dim, cfg.bits, seed=cfg.seed)
        self.centroids = torch.from_numpy(centroids_np).to(device)  # (n_levels,)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantiza vetores.

        Args:
            x: shape (..., dim)

        Returns:
            idx: shape (..., dim), dtype uint8 (assumindo bits ≤ 8)
            norms: shape (...,), dtype float16
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        # 1. Extrai norma
        norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        unit = x / norms
        # 2. Rotaciona
        y = unit @ self.pi.T  # (..., dim)
        # 3. Quantiza por coordenada
        # dists[..., j, k] = |y[..., j] - c[k]|
        dists = (y.unsqueeze(-1) - self.centroids.view(1, -1)).abs()
        idx = dists.argmin(dim=-1).to(torch.uint8)
        return idx, norms.squeeze(-1).to(torch.float16)

    @torch.no_grad()
    def dequantize(self, idx: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Reconstrói vetores aproximados."""
        y_hat = self.centroids[idx.long()]  # (..., dim)
        unit_hat = y_hat @ self.pi  # Π^T rotate back (pi is orthogonal so Π^T = Π^{-1})
        # Nota: usamos `y_hat @ self.pi` pois a rotação foi y = unit @ self.pi.T
        # Então unit = y @ self.pi (já que Π.T.T == Π e (Π.T)^{-1} = Π)
        return unit_hat * norms.unsqueeze(-1).to(unit_hat.dtype)


# ============================================================
# Bit packing (opcional mas importante para compressão real)
# ============================================================

def pack_bits(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Empacota índices para usar exatamente `bits` bits por valor.

    Input: uint8 tensor com valores em [0, 2^bits)
    Output: uint8 tensor empacotado, shape[-1] = ceil(N * bits / 8)
    """
    assert idx.dtype == torch.uint8
    flat = idx.reshape(-1).to(torch.int64)
    n = flat.numel()
    total_bits = n * bits
    n_bytes = (total_bits + 7) // 8
    packed = torch.zeros(n_bytes, dtype=torch.uint8, device=idx.device)
    # Implementação simples: bit-shift
    bit_cursor = 0
    for i in range(n):
        val = int(flat[i].item())
        byte_i = bit_cursor // 8
        bit_i = bit_cursor % 8
        packed[byte_i] |= ((val & ((1 << bits) - 1)) << bit_i) & 0xFF
        if bit_i + bits > 8:
            packed[byte_i + 1] |= (val >> (8 - bit_i)) & 0xFF
        bit_cursor += bits
    return packed


def compression_ratio(n_vectors: int, dim: int, bits: int,
                      fp_bytes: int = 2) -> float:
    """Ratio FP16 vs (bits_per_coord * dim + norm_16bit)."""
    original = n_vectors * dim * fp_bytes
    compressed = n_vectors * (dim * bits / 8 + 2)  # +2 bytes de norma FP16
    return original / compressed
