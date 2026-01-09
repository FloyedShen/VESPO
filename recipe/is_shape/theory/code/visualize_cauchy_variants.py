#!/usr/bin/env python3
"""
Compare different Cauchy-like φ formulas:

1. Original (buggy):     φ = (1+λ)w / (1 + λw²)           -- centered at w=0
2. Shifted v1:           φ = (1+λ)w / (1 + λ(w-1)²)       -- centered at w=1, but φ(1)=1+λ
3. Shifted v2:           φ = w / (1 + λ(w-1)²)            -- centered at w=1, φ(1)=1
4. User's proposal:      φ = (1 + λ(w-1)) / (1 + λ²(w-1)²) -- centered at w=1, φ(1)=1
5. Welsch (reference):   φ = exp(-0.5λ(w-1)²) × w         -- centered at w=1, φ(1)=1
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

w = np.linspace(0.0, 2.0, 500)
lambdas = [1, 5, 10, 20, 50, 100]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(lambdas)))

# ============================================================================
# Top Left: User's proposal φ = (1 + λ(w-1)) / (1 + λ²(w-1)²)
# ============================================================================
ax1 = axes[0, 0]
for i, lam in enumerate(lambdas):
    x = w - 1.0  # centered at w=1
    phi = (1.0 + lam * x) / (1.0 + lam**2 * x**2)
    ax1.plot(w, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.plot(w, w, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=w')
ax1.set_xlabel('w = π/π_ref', fontsize=11)
ax1.set_ylabel('φ (effective gradient weight)', fontsize=11)
ax1.set_title("User's Proposal\nφ = (1 + λ(w-1)) / (1 + λ²(w-1)²)", fontsize=12, color='purple')
ax1.set_xlim(0, 2)
ax1.set_ylim(-0.5, 1.5)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.fill_between(w, -0.5, 0, alpha=0.1, color='red')
ax1.annotate('⚠ Can go negative!', xy=(0.2, -0.2), fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ============================================================================
# Top Right: Shifted v2 φ = w / (1 + λ(w-1)²) -- always positive
# ============================================================================
ax2 = axes[0, 1]
for i, lam in enumerate(lambdas):
    phi = w / (1.0 + lam * (w - 1.0)**2)
    ax2.plot(w, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.plot(w, w, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=w')
ax2.set_xlabel('w = π/π_ref', fontsize=11)
ax2.set_ylabel('φ (effective gradient weight)', fontsize=11)
ax2.set_title("Simple Shifted Cauchy\nφ = w / (1 + λ(w-1)²)", fontsize=12, color='green')
ax2.set_xlim(0, 2)
ax2.set_ylim(-0.5, 1.5)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.annotate('✓ Always ≥ 0\n✓ φ(1)=1', xy=(1.3, 0.3), fontsize=10, color='green',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# ============================================================================
# Bottom Left: Welsch for comparison
# ============================================================================
ax3 = axes[1, 0]
for i, lam in enumerate(lambdas):
    phi = np.exp(-0.5 * lam * (w - 1.0)**2) * w
    ax3.plot(w, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.plot(w, w, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=w')
ax3.set_xlabel('w = π/π_ref', fontsize=11)
ax3.set_ylabel('φ (effective gradient weight)', fontsize=11)
ax3.set_title("Welsch (Reference)\nφ = exp(-0.5λ(w-1)²) × w", fontsize=12, color='blue')
ax3.set_xlim(0, 2)
ax3.set_ylim(-0.5, 1.5)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

# ============================================================================
# Bottom Right: Comparison at λ=20
# ============================================================================
ax4 = axes[1, 1]
lam = 20
x = w - 1.0

# All variants
phi_user = (1.0 + lam * x) / (1.0 + lam**2 * x**2)
phi_simple = w / (1.0 + lam * (w - 1.0)**2)
phi_welsch = np.exp(-0.5 * lam * (w - 1.0)**2) * w

ax4.plot(w, phi_user, 'purple', linewidth=2.5, label='User: (1+λx)/(1+λ²x²)')
ax4.plot(w, phi_simple, 'green', linewidth=2.5, label='Simple: w/(1+λ(w-1)²)')
ax4.plot(w, phi_welsch, 'blue', linewidth=2.5, label='Welsch: exp(...)·w')
ax4.plot(w, w, 'k--', alpha=0.3, linewidth=1.5, label='φ=w')

ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4.axvline(x=0.8, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax4.axvline(x=1.2, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax4.set_xlabel('w = π/π_ref', fontsize=11)
ax4.set_ylabel('φ (effective gradient weight)', fontsize=11)
ax4.set_title(f'Direct Comparison at λ={lam}', fontsize=12)
ax4.set_xlim(0, 2)
ax4.set_ylim(-0.5, 1.5)
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.fill_between(w, -0.5, 0, alpha=0.1, color='red')

fig.suptitle('Comparing Cauchy-like φ Formulas (all centered at w=1)', fontsize=14, y=1.01)
plt.tight_layout()

output_path = '/Users/floyed/Documents/workplace/verl/recipe/is_shape/theory/figures/cauchy_variants_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# ============================================================================
# Additional: Show the f(w) functions
# ============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# f(w) for User's proposal
ax_f1 = axes2[0]
for i, lam in enumerate(lambdas):
    x = w - 1.0
    # f(w) = (1/λ)arctan(λx) + (1/2λ)ln(1+λ²x²)
    f_w = (1.0/lam) * np.arctan(lam * x) + (1.0/(2*lam)) * np.log(1.0 + lam**2 * x**2)
    ax_f1.plot(w, f_w, color=colors[i], linewidth=2, label=f'λ={lam}')

ax_f1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_f1.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_f1.set_xlabel('w = π/π_ref', fontsize=11)
ax_f1.set_ylabel('f(w)', fontsize=11)
ax_f1.set_title("User's Proposal f(w)\nf = (1/λ)arctan(λ(w-1)) + (1/2λ)ln(1+λ²(w-1)²)", fontsize=10, color='purple')
ax_f1.set_xlim(0, 2)
ax_f1.legend(loc='lower right', fontsize=9)
ax_f1.grid(True, alpha=0.3)

# f(w) for Simple shifted Cauchy
ax_f2 = axes2[1]
for i, lam in enumerate(lambdas):
    x = w - 1.0
    # f(w) = (1/2λ)ln(1+λx²) + (1/√λ)arctan(√λ·x)
    f_w = (1.0/(2*lam)) * np.log(1.0 + lam * x**2) + (1.0/np.sqrt(lam)) * np.arctan(np.sqrt(lam) * x)
    ax_f2.plot(w, f_w, color=colors[i], linewidth=2, label=f'λ={lam}')

ax_f2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_f2.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_f2.set_xlabel('w = π/π_ref', fontsize=11)
ax_f2.set_ylabel('f(w)', fontsize=11)
ax_f2.set_title("Simple Shifted Cauchy f(w)\nf = (1/2λ)ln(1+λ(w-1)²) + (1/√λ)arctan(√λ(w-1))", fontsize=10, color='green')
ax_f2.set_xlim(0, 2)
ax_f2.legend(loc='lower right', fontsize=9)
ax_f2.grid(True, alpha=0.3)

# f(w) for Welsch (numerical)
from scipy import integrate
ax_f3 = axes2[2]
for i, lam in enumerate(lambdas):
    f_w_values = []
    for w_val in w:
        def integrand(x):
            return np.exp(-0.5 * lam * (x - 1.0)**2) * x
        result, _ = integrate.quad(integrand, 0, w_val)
        f_w_values.append(result)
    ax_f3.plot(w, f_w_values, color=colors[i], linewidth=2, label=f'λ={lam}')

ax_f3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_f3.set_xlabel('w = π/π_ref', fontsize=11)
ax_f3.set_ylabel('f(w)', fontsize=11)
ax_f3.set_title("Welsch f(w)\nf = ∫exp(-0.5λ(w-1)²)·w dw", fontsize=10, color='blue')
ax_f3.set_xlim(0, 2)
ax_f3.legend(loc='lower right', fontsize=9)
ax_f3.grid(True, alpha=0.3)

fig2.suptitle('Corresponding f(w) Functions', fontsize=14, y=1.02)
plt.tight_layout()

output_path2 = '/Users/floyed/Documents/workplace/verl/recipe/is_shape/theory/figures/cauchy_variants_f_comparison.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path2}")

# ============================================================================
# Print summary table
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: Cauchy-like φ Formulas Comparison")
print("="*80)
print(f"{'Formula':<50} {'φ(w=1)':<10} {'φ(w=0) at λ=20':<15} {'Issue'}")
print("-"*80)

lam = 20
# User's proposal
phi_1 = 1.0
phi_0 = (1.0 - lam) / (1.0 + lam**2)
print(f"{'(1+λ(w-1))/(1+λ²(w-1)²)':<50} {phi_1:<10.2f} {phi_0:<15.4f} {'Can be negative!'}")

# Simple shifted
phi_1 = 1.0
phi_0 = 0.0 / (1.0 + lam * 1.0)
print(f"{'w/(1+λ(w-1)²)':<50} {phi_1:<10.2f} {phi_0:<15.4f} {'Always ≥ 0'}")

# Welsch
phi_1 = 1.0
phi_0 = np.exp(-0.5 * lam * 1.0) * 0.0
print(f"{'exp(-0.5λ(w-1)²)·w':<50} {phi_1:<10.2f} {phi_0:<15.4f} {'Always ≥ 0'}")

print("="*80)
