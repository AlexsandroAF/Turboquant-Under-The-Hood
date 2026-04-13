"""
TurboQuant in 10 lines — your first quantized vector.

Run this to see the core algorithm in action:
  python examples/quickstart.py

No GPU needed. No model needed. Just pure math.
"""
import sys
sys.path.insert(0, "src")

import torch
from tq_core import TurboQuantMSE, TQConfig

# --- Step 1: Create a quantizer (128-dimensional, 3-bit) ---
tq = TurboQuantMSE(TQConfig(dim=128, bits=3), device="cpu")

# --- Step 2: Make a random vector (simulating a KV cache entry) ---
x = torch.randn(1, 128)  # one vector, 128 dimensions

# --- Step 3: Quantize it ---
indices, norm = tq.quantize(x)
# indices: 128 integers (0-7 for 3-bit), stored as uint8
# norm: the vector's length, stored as float16

# --- Step 4: Reconstruct it ---
x_hat = tq.dequantize(indices, norm)

# --- Step 5: See how close we got ---
# Note: the paper measures MSE on UNIT vectors (norm=1). Our random vector
# has norm ~11, so raw MSE is scaled by norm^2. Cosine similarity is the
# norm-independent metric that matters for attention scores.
cosine = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).item()
x_unit = x / x.norm(dim=-1, keepdim=True)
x_hat_unit = x_hat / x_hat.norm(dim=-1, keepdim=True)
mse_unit = ((x_unit - x_hat_unit) ** 2).sum().item()

print("TurboQuant Quickstart")
print("=" * 50)
print(f"Original vector:      {x[0, :5].tolist()}...")  # first 5 dims
print(f"Reconstructed:        {x_hat[0, :5].tolist()}...")
print(f"Indices (3-bit):      {indices[0, :10].tolist()}...")  # first 10
print(f"Norm (FP16):          {norm.item():.4f}")
print(f"Cosine similarity:    {cosine:.4f}")
print(f"MSE (unit vectors):   {mse_unit:.6f}")
print(f"Compression:          {128 * 2 / (128 * 3/8 + 2):.1f}x (FP16 -> 3-bit + norm)")
print()
print("Paper bound for 3-bit MSE (unit vectors): 0.030")
print(f"Our MSE (unit vectors):                   {mse_unit:.4f}")
print(f"{'PASS' if mse_unit < 0.045 else 'FAIL'} (within 1.5x of theoretical optimum)")
print()

# --- Bonus: see the effect across bit-widths ---
print("Bit-width sweep:")
print(f"{'bits':>5} {'MSE':>10} {'cosine':>10} {'compression':>12}")
print("-" * 40)
for bits in [1, 2, 3, 4]:
    tq_b = TurboQuantMSE(TQConfig(dim=128, bits=bits), device="cpu")
    vectors = torch.randn(100, 128)  # 100 random vectors
    idx, norms = tq_b.quantize(vectors)
    recon = tq_b.dequantize(idx, norms)
    mse_b = ((vectors - recon) ** 2).sum(dim=-1).mean().item()
    cos_b = torch.nn.functional.cosine_similarity(vectors, recon, dim=-1).mean().item()
    comp = 128 * 2 / (128 * bits/8 + 2)
    print(f"{bits:>5} {mse_b:>10.4f} {cos_b:>10.4f} {comp:>11.1f}x")
