#!/usr/bin/env python3
"""
Analyze φ(x) = (1 + λx) / (1 + λ²x²) -- User's proposed formula

This is centered at x=0, with φ(0) = 1.

f(x) = (1/λ)arctan(λx) + (1/2λ)ln(1 + λ²x²)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

x = np.linspace(-0.5, 2.0, 500)
w = x  # x = w in this case
lambdas = [1, 5, 10, 20, 35, 50, 100]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(lambdas)))

# ============================================================================
# Top Left: φ(x) = (1 + λx) / (1 + λ²x²)
# ============================================================================
ax1 = axes[0, 0]
for i, lam in enumerate(lambdas):
    phi = (1.0 + lam * x) / (1.0 + lam**2 * x**2)
    ax1.plot(x, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
ax1.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.axvline(x=1.0, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='w=1')
ax1.plot(x, x, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=x')
ax1.set_xlabel('x = w', fontsize=11)
ax1.set_ylabel('φ (effective gradient weight)', fontsize=11)
ax1.set_title("User's Formula\nφ = (1 + λx) / (1 + λ²x²)", fontsize=12, color='purple')
ax1.set_xlim(-0.5, 2)
ax1.set_ylim(-0.5, 1.5)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.fill_between(x, -0.5, 0, alpha=0.1, color='red')
ax1.annotate('φ(0) = 1\n(peak at x=0)', xy=(0.05, 1.1), fontsize=10, color='blue',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ============================================================================
# Top Right: Original Cauchy φ = (1+λ)w / (1 + λw²) for comparison
# ============================================================================
ax2 = axes[0, 1]
for i, lam in enumerate(lambdas):
    phi = (1.0 + lam) * x / (1.0 + lam * x**2)
    ax2.plot(x, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
ax2.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(x=1.0, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='w=1')
ax2.plot(x, x, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=x')
ax2.set_xlabel('x = w', fontsize=11)
ax2.set_ylabel('φ (effective gradient weight)', fontsize=11)
ax2.set_title("Original Cauchy (Current Code)\nφ = (1+λ)w / (1 + λw²)", fontsize=12, color='red')
ax2.set_xlim(-0.5, 2)
ax2.set_ylim(-0.5, 1.5)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.annotate('φ(0) = 0\n(zero at origin)', xy=(0.05, 0.2), fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ============================================================================
# Bottom Left: f(x) for User's formula
# ============================================================================
ax3 = axes[1, 0]
for i, lam in enumerate(lambdas):
    # f(x) = (1/λ)arctan(λx) + (1/2λ)ln(1 + λ²x²)
    f_x = (1.0/lam) * np.arctan(lam * x) + (1.0/(2*lam)) * np.log(1.0 + lam**2 * x**2)
    ax3.plot(x, f_x, color=colors[i], linewidth=2, label=f'λ={lam}')

ax3.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.axvline(x=1.0, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax3.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.set_xlabel('x = w', fontsize=11)
ax3.set_ylabel('f(x)', fontsize=11)
ax3.set_title("User's f(x)\nf = (1/λ)arctan(λx) + (1/2λ)ln(1 + λ²x²)", fontsize=11, color='purple')
ax3.set_xlim(-0.5, 2)
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)

# ============================================================================
# Bottom Right: Direct comparison at λ=20
# ============================================================================
ax4 = axes[1, 1]
lam = 20

# User's formula
phi_user = (1.0 + lam * x) / (1.0 + lam**2 * x**2)
# Original Cauchy
phi_orig = (1.0 + lam) * x / (1.0 + lam * x**2)
# Welsch (centered at w=1)
phi_welsch = np.exp(-0.5 * lam * (x - 1.0)**2) * x

ax4.plot(x, phi_user, 'purple', linewidth=2.5, label="User: (1+λx)/(1+λ²x²)")
ax4.plot(x, phi_orig, 'red', linewidth=2.5, linestyle='--', label="Orig: (1+λ)x/(1+λx²)")
ax4.plot(x, phi_welsch, 'blue', linewidth=2.5, linestyle=':', label="Welsch (at w=1)")
ax4.plot(x, x, 'k--', alpha=0.3, linewidth=1.5, label='φ=x')

ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
ax4.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4.axvline(x=1.0, color='green', linestyle=':', alpha=0.7, linewidth=2)
ax4.set_xlabel('x = w', fontsize=11)
ax4.set_ylabel('φ', fontsize=11)
ax4.set_title(f'Direct Comparison at λ={lam}', fontsize=12)
ax4.set_xlim(-0.5, 2)
ax4.set_ylim(-0.5, 1.5)
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.fill_between(x, -0.5, 0, alpha=0.1, color='red')

fig.suptitle('Analysis of φ(x) = (1 + λx) / (1 + λ²x²)\n'
             'Centered at x=0 with φ(0)=1', fontsize=14, y=1.02)
plt.tight_layout()

output_path = '/Users/floyed/Documents/workplace/verl/recipe/is_shape/theory/figures/user_cauchy_formula.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# ============================================================================
# Print analysis
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS: φ(x) = (1 + λx) / (1 + λ²x²)")
print("="*80)

print("\n[Key Properties]")
print("  • φ(0) = 1 (peak at origin)")
print("  • φ(x) → 0 as x → ±∞")
print("  • Can be NEGATIVE when x < -1/λ")
print("\n[Values at key points for λ=20]:")
lam = 20
for x_val in [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    phi = (1.0 + lam * x_val) / (1.0 + lam**2 * x_val**2)
    print(f"  φ({x_val}) = {phi:.4f}")

print("\n[Corresponding f(x)]:")
print("  f(x) = (1/λ)arctan(λx) + (1/2λ)ln(1 + λ²x²)")
print("\n[Comparison with Original Cauchy]:")
print("  Original: φ = (1+λ)w / (1+λw²)  →  φ(0)=0, peak near w=1/√λ")
print("  User's:   φ = (1+λx)/(1+λ²x²)   →  φ(0)=1, peak at x=0")
print("="*80)
