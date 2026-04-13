"""
Teste sintético: valida contra bounds teóricos do paper.

Bounds esperados (Theorem 1, x ∈ S^(d-1)):
  b=1 → D_mse ≤ 0.360
  b=2 → D_mse ≤ 0.117
  b=3 → D_mse ≤ 0.030
  b=4 → D_mse ≤ 0.009
"""
import sys
import time
import torch

from tq_core import TurboQuantMSE, TQConfig, compression_ratio

PAPER_BOUNDS = {1: 0.360, 2: 0.117, 3: 0.030, 4: 0.009}

def test_distortion_bounds(d: int = 128, n_samples: int = 5000, device: str = "cuda"):
    print(f"\n=== Distortion vs Paper Bounds (d={d}, n={n_samples}) ===")
    torch.manual_seed(0)
    # Vetores uniformes em S^(d-1)
    x = torch.randn(n_samples, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    print(f"{'bits':>5} {'MSE':>12} {'Bound':>10} {'Ratio':>8} {'Status':>8}")
    print("-" * 50)
    results = []
    for bits in [1, 2, 3, 4]:
        t0 = time.time()
        tq = TurboQuantMSE(TQConfig(dim=d, bits=bits), device=device)
        setup_time = time.time() - t0

        t0 = time.time()
        idx, norms = tq.quantize(x)
        quant_time = time.time() - t0

        t0 = time.time()
        x_hat = tq.dequantize(idx, norms)
        dequant_time = time.time() - t0

        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        bound = PAPER_BOUNDS[bits]
        ratio = mse / bound
        status = "OK" if mse <= bound * 1.2 else "FAIL"
        print(f"{bits:>5} {mse:>12.6f} {bound:>10.6f} {ratio:>7.2f}x {status:>8}")
        results.append((bits, mse, bound, setup_time, quant_time, dequant_time))

    return results


def test_norm_preservation(d: int = 128, device: str = "cuda"):
    print(f"\n=== Norm preservation (d={d}) ===")
    torch.manual_seed(1)
    # Vetores com normas diversas
    x = torch.randn(1000, d, device=device) * torch.rand(1000, 1, device=device) * 10
    for bits in [2, 3, 4]:
        tq = TurboQuantMSE(TQConfig(dim=d, bits=bits), device=device)
        idx, norms = tq.quantize(x)
        x_hat = tq.dequantize(idx, norms)
        norm_err = (x.norm(dim=-1) - x_hat.norm(dim=-1)).abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
        print(f"  bits={bits}: mean norm error = {norm_err:.4f}, cosine sim = {cos:.4f}")


def test_inner_product_fidelity(d: int = 128, device: str = "cuda"):
    print(f"\n=== Inner product fidelity (d={d}) ===")
    torch.manual_seed(2)
    # Simula Q·K attention scores
    n_q, n_k = 100, 1000
    q = torch.randn(n_q, d, device=device)
    k = torch.randn(n_k, d, device=device)
    k = k / k.norm(dim=-1, keepdim=True)  # keys normalizados
    true_scores = q @ k.T  # (n_q, n_k)

    for bits in [2, 3, 4]:
        tq = TurboQuantMSE(TQConfig(dim=d, bits=bits), device=device)
        idx, norms = tq.quantize(k)
        k_hat = tq.dequantize(idx, norms)
        approx_scores = q @ k_hat.T
        abs_err = (true_scores - approx_scores).abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(
            true_scores.flatten(), approx_scores.flatten(), dim=0
        ).item()
        # Top-k agreement: para cada query, top-5 original vs top-5 quantizado
        k_top = 5
        top_true = true_scores.topk(k_top, dim=-1).indices
        top_approx = approx_scores.topk(k_top, dim=-1).indices
        agreement = (top_true == top_approx).float().mean().item()
        print(f"  bits={bits}: mean abs err = {abs_err:.4f}, score cos = {cos:.4f}, top-{k_top} match = {agreement:.2%}")


def test_compression_ratios(d: int = 128):
    print(f"\n=== Compression ratios (d={d}, vs FP16) ===")
    for bits in [2, 3, 4, 5]:
        r = compression_ratio(n_vectors=1000, dim=d, bits=bits)
        print(f"  bits={bits}: {r:.2f}x compression")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    test_distortion_bounds(d=128, device=device)
    test_norm_preservation(d=128, device=device)
    test_inner_product_fidelity(d=128, device=device)
    test_compression_ratios(d=128)

    print("\n=== Sweep de dimensões ===")
    for d in [64, 128, 256]:
        print(f"\n--- d={d} ---")
        test_distortion_bounds(d=d, n_samples=2000, device=device)


if __name__ == "__main__":
    main()
