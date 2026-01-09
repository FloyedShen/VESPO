#!/usr/bin/env python3
"""
Visualize Corrected Cauchy IS (centered at w=1) vs Original Cauchy and Welsch.

Corrected Cauchy: φ = (1+λ)w / (1 + λ(w-1)²) -- centered at w=1
Original Cauchy:  φ = (1+λ)w / (1 + λw²)     -- centered at w=0 (problematic!)
Welsch:           φ = exp(-0.5λ(w-1)²) × w   -- centered at w=1

f(w) for Corrected Cauchy:
f(w) = (1+λ)/(2λ) · ln(1 + λ(w-1)²) + (1+λ)/√λ · arctan(√λ·(w-1))
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# Set up the figure with 2 rows
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# w range
w = np.linspace(0.01, 2.0, 500)

# λ values to plot
lambdas = [1, 5, 10, 20, 35, 50, 100]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(lambdas)))

# ============================================================================
# Row 1: φ (effective gradient weight) comparison
# ============================================================================

# Left: Original Cauchy (problematic - centered at w=0)
ax1 = axes[0, 0]
for i, lam in enumerate(lambdas):
    phi = (1.0 + lam) * w / (1.0 + lam * w ** 2)
    ax1.plot(w, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.plot(w, w, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=w')
ax1.set_xlabel('w = π/π_ref', fontsize=11)
ax1.set_ylabel('φ (effective gradient weight)', fontsize=11)
ax1.set_title('Original Cauchy (centered at w=0)\nφ = (1+λ)w / (1 + λw²)', fontsize=12, color='red')
ax1.set_xlim(0, 2)
ax1.set_ylim(0, 2)
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.annotate('❌ WRONG: w<1 amplified!', xy=(0.3, 1.5), fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Middle: Corrected Cauchy (centered at w=1)
ax2 = axes[0, 1]
for i, lam in enumerate(lambdas):
    phi = (1.0 + lam) * w / (1.0 + lam * (w - 1.0) ** 2)
    ax2.plot(w, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(x=0.8, color='red', linestyle=':', alpha=0.3, linewidth=1)
ax2.axvline(x=1.2, color='red', linestyle=':', alpha=0.3, linewidth=1)
ax2.plot(w, w, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=w')
ax2.set_xlabel('w = π/π_ref', fontsize=11)
ax2.set_ylabel('φ (effective gradient weight)', fontsize=11)
ax2.set_title('Corrected Cauchy (centered at w=1)\nφ = (1+λ)w / (1 + λ(w-1)²)', fontsize=12, color='green')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.annotate('✓ Peak at w=1', xy=(1.05, 1.6), fontsize=10, color='green',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Right: Welsch (for comparison)
ax3 = axes[0, 2]
for i, lam in enumerate(lambdas):
    gate = np.exp(-0.5 * lam * (w - 1.0) ** 2)
    phi = gate * w
    ax3.plot(w, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.axvline(x=0.8, color='red', linestyle=':', alpha=0.3, linewidth=1)
ax3.axvline(x=1.2, color='red', linestyle=':', alpha=0.3, linewidth=1)
ax3.plot(w, w, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=w')
ax3.set_xlabel('w = π/π_ref', fontsize=11)
ax3.set_ylabel('φ (effective gradient weight)', fontsize=11)
ax3.set_title('Welsch (centered at w=1)\nφ = exp(-0.5λ(w-1)²) × w', fontsize=12, color='blue')
ax3.set_xlim(0, 2)
ax3.set_ylim(0, 2)
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.annotate('✓ Gaussian gate', xy=(1.3, 0.4), fontsize=10, color='blue',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ============================================================================
# Row 2: f(w) comparison
# ============================================================================

# Left: Original Cauchy f(w)
ax4 = axes[1, 0]
for i, lam in enumerate(lambdas):
    sqrt_lam = np.sqrt(lam)
    norm_factor = (1.0 + lam) / sqrt_lam
    f_w = norm_factor * np.arctan(sqrt_lam * w)
    ax4.plot(w, f_w, color=colors[i], linewidth=2, label=f'λ={lam}')

ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4.set_xlabel('w = π/π_ref', fontsize=11)
ax4.set_ylabel('f(w)', fontsize=11)
ax4.set_title('Original Cauchy f(w)\nf = (1+λ)/√λ · arctan(√λ·w)', fontsize=12, color='red')
ax4.set_xlim(0, 2)
ax4.legend(loc='lower right', fontsize=8)
ax4.grid(True, alpha=0.3)

# Middle: Corrected Cauchy f(w)
ax5 = axes[1, 1]
for i, lam in enumerate(lambdas):
    sqrt_lam = np.sqrt(lam)
    # f(w) = (1+λ)/(2λ) · ln(1 + λ(w-1)²) + (1+λ)/√λ · arctan(√λ·(w-1))
    log_term = (1.0 + lam) / (2.0 * lam) * np.log(1.0 + lam * (w - 1.0) ** 2)
    arctan_term = (1.0 + lam) / sqrt_lam * np.arctan(sqrt_lam * (w - 1.0))
    f_w = log_term + arctan_term
    ax5.plot(w, f_w, color=colors[i], linewidth=2, label=f'λ={lam}')

ax5.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax5.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax5.set_xlabel('w = π/π_ref', fontsize=11)
ax5.set_ylabel('f(w)', fontsize=11)
ax5.set_title('Corrected Cauchy f(w)\nf = (1+λ)/(2λ)·ln(1+λ(w-1)²) + (1+λ)/√λ·arctan(√λ(w-1))', fontsize=10, color='green')
ax5.set_xlim(0, 2)
ax5.legend(loc='lower right', fontsize=8)
ax5.grid(True, alpha=0.3)

# Right: Welsch f(w) - need to integrate exp(-0.5λ(w-1)²) × w
# This is complex, let's do numerical integration
ax6 = axes[1, 2]
from scipy import integrate

for i, lam in enumerate(lambdas):
    # Numerical integration for Welsch
    f_w_values = []
    for w_val in w:
        def integrand(x):
            return np.exp(-0.5 * lam * (x - 1.0) ** 2) * x
        result, _ = integrate.quad(integrand, 0, w_val)
        f_w_values.append(result)
    ax6.plot(w, f_w_values, color=colors[i], linewidth=2, label=f'λ={lam}')

ax6.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax6.set_xlabel('w = π/π_ref', fontsize=11)
ax6.set_ylabel('f(w)', fontsize=11)
ax6.set_title('Welsch f(w)\nf = ∫exp(-0.5λ(w-1)²)·w dw', fontsize=12, color='blue')
ax6.set_xlim(0, 2)
ax6.legend(loc='lower right', fontsize=8)
ax6.grid(True, alpha=0.3)

# ============================================================================
# Overall title and save
# ============================================================================
fig.suptitle('Comparison: Original Cauchy vs Corrected Cauchy vs Welsch\n'
             'Top row: φ(w) = effective gradient weight | Bottom row: f(w) = IS-Reshape function',
             fontsize=14, y=1.02)

plt.tight_layout()

# Save figure
output_path = '/Users/floyed/Documents/workplace/verl/recipe/is_shape/theory/figures/cauchy_centered_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

output_path_pdf = '/Users/floyed/Documents/workplace/verl/recipe/is_shape/theory/figures/cauchy_centered_comparison.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Figure saved to: {output_path_pdf}")

# ============================================================================
# Also create a focused comparison of Corrected Cauchy vs Welsch
# ============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Focused comparison: φ curves
ax_left = axes2[0]
w_focus = np.linspace(0.5, 1.5, 500)

# Plot both with same λ values
for i, lam in enumerate([5, 20, 50, 100]):
    color = plt.cm.viridis(i / 4)

    # Corrected Cauchy
    phi_cauchy = (1.0 + lam) * w_focus / (1.0 + lam * (w_focus - 1.0) ** 2)
    ax_left.plot(w_focus, phi_cauchy, color=color, linewidth=2, linestyle='-',
                 label=f'Cauchy λ={lam}')

    # Welsch
    phi_welsch = np.exp(-0.5 * lam * (w_focus - 1.0) ** 2) * w_focus
    ax_left.plot(w_focus, phi_welsch, color=color, linewidth=2, linestyle='--',
                 label=f'Welsch λ={lam}')

ax_left.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_left.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_left.axvline(x=0.8, color='red', linestyle=':', alpha=0.3, linewidth=1.5)
ax_left.axvline(x=1.2, color='red', linestyle=':', alpha=0.3, linewidth=1.5)
ax_left.plot(w_focus, w_focus, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=w')
ax_left.set_xlabel('w = π/π_ref', fontsize=12)
ax_left.set_ylabel('φ (effective gradient weight)', fontsize=12)
ax_left.set_title('Corrected Cauchy (solid) vs Welsch (dashed)\nBoth centered at w=1', fontsize=13)
ax_left.set_xlim(0.5, 1.5)
ax_left.set_ylim(0, 1.6)
ax_left.legend(loc='upper left', fontsize=9, ncol=2)
ax_left.grid(True, alpha=0.3)

# Key differences annotation
ax_right = axes2[1]
ax_right.axis('off')
text = """
Key Differences: Corrected Cauchy vs Welsch

┌─────────────────────────────────────────────────────────────────┐
│                  Corrected Cauchy                               │
│  φ = (1+λ)w / (1 + λ(w-1)²)                                     │
│                                                                 │
│  • Heavy-tailed: φ → 0 slowly as |w-1| → ∞                      │
│  • At w=1: φ = (1+λ)·1 / 1 = 1+λ (amplification at center!)     │
│  • More tolerant to large deviations                            │
│  • Robustness property: bounded influence function              │
│                                                                 │
│  f(w) = (1+λ)/(2λ)·ln(1+λ(w-1)²) + (1+λ)/√λ·arctan(√λ(w-1))    │
│       = log term + arctan term                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Welsch                                   │
│  φ = exp(-0.5λ(w-1)²) × w                                       │
│                                                                 │
│  • Light-tailed: φ → 0 exponentially as |w-1| → ∞               │
│  • At w=1: φ = 1·1 = 1 (no amplification)                       │
│  • Stronger suppression of outliers                             │
│  • Like "soft PPO clip"                                         │
│                                                                 │
│  f(w) = ∫ exp(-0.5λ(w-1)²)·w dw (no closed form)                │
└─────────────────────────────────────────────────────────────────┘

⚠️  Note: Corrected Cauchy still has φ(w=1) = 1+λ > 1
    This means it AMPLIFIES gradients at w=1!

    To fix: use φ = w / (1 + λ(w-1)²) without (1+λ) factor
    Then φ(w=1) = 1 (identity at center)
"""
ax_right.text(0.05, 0.95, text, transform=ax_right.transAxes, fontsize=11,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path2 = '/Users/floyed/Documents/workplace/verl/recipe/is_shape/theory/figures/cauchy_vs_welsch_focused.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path2}")
