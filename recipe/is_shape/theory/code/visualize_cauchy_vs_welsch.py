#!/usr/bin/env python3
"""
Visualize Cauchy vs Welsch IS: φ (effective gradient weight) as a function of w
for different λ values.

ACTUAL IMPLEMENTATION formulas:
- Cauchy (current code): φ = (1+λ)w / (1 + λw²)  -- centered at w=0, NOT w=1!
- Welsch: φ = exp(-0.5λ(w-1)²) × w  -- centered at w=1
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import matplotlib.pyplot as plt

# Set up the figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# w range
w = np.linspace(0.5, 1.5, 500)

# λ values to plot
lambdas = [1, 5, 10, 20, 35, 50, 100]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(lambdas)))

# ============================================================================
# Left subplot: Cauchy (ACTUAL CODE IMPLEMENTATION)
# φ = (1+λ)w / (1 + λw²)
# ============================================================================
ax1 = axes[0]

for i, lam in enumerate(lambdas):
    # ACTUAL implementation: φ = (1+λ)w / (1 + λw²)
    phi = (1.0 + lam) * w / (1.0 + lam * w ** 2)
    ax1.plot(w, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

# Reference lines
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='PPO clip boundary')
ax1.axvline(x=0.8, color='red', linestyle=':', alpha=0.3, linewidth=1)
ax1.axvline(x=1.2, color='red', linestyle=':', alpha=0.3, linewidth=1)

# Also plot w itself for reference
ax1.plot(w, w, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=w (no gate)')

ax1.set_xlabel('w = π(a|s) / π_ref(a|s)', fontsize=12)
ax1.set_ylabel('φ (effective gradient weight)', fontsize=12)
ax1.set_title('Cauchy IS (Current Code)\nφ = (1+λ)w / (1 + λw²)', fontsize=13)
ax1.set_xlim(0.5, 1.5)
ax1.set_ylim(0, 1.6)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Add annotation for the problem
ax1.annotate('w<1: φ INCREASES\nwith larger λ!',
             xy=(0.65, 1.15), fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================================
# Right subplot: Welsch
# φ = exp(-0.5λ(w-1)²) × w
# ============================================================================
ax2 = axes[1]

for i, lam in enumerate(lambdas):
    gate = np.exp(-0.5 * lam * (w - 1.0) ** 2)
    phi = gate * w
    ax2.plot(w, phi, color=colors[i], linewidth=2, label=f'λ={lam}')

# Reference lines
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='PPO clip boundary')
ax2.axvline(x=0.8, color='red', linestyle=':', alpha=0.3, linewidth=1)
ax2.axvline(x=1.2, color='red', linestyle=':', alpha=0.3, linewidth=1)

# Also plot w itself for reference
ax2.plot(w, w, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='φ=w (no gate)')

ax2.set_xlabel('w = π(a|s) / π_ref(a|s)', fontsize=12)
ax2.set_ylabel('φ (effective gradient weight)', fontsize=12)
ax2.set_title('Welsch IS\nφ = exp(-0.5λ(w-1)²) × w', fontsize=13)
ax2.set_xlim(0.5, 1.5)
ax2.set_ylim(0, 1.6)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Add annotation
ax2.annotate('Symmetric around w=1\n(like soft PPO clip)',
             xy=(1.05, 0.3), fontsize=10, color='green',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# ============================================================================
# Overall title and save
# ============================================================================
fig.suptitle('IS-Reshape: Effective Gradient Weight φ vs Importance Weight w\n'
             '(Red dotted lines show PPO clip boundaries at w=0.8 and w=1.2)',
             fontsize=13, y=1.02)

plt.tight_layout()

# Save figure
output_path = '/Users/floyed/Documents/workplace/verl/recipe/is_shape/theory/figures/cauchy_vs_welsch_phi.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Also save PDF
output_path_pdf = '/Users/floyed/Documents/workplace/verl/recipe/is_shape/theory/figures/cauchy_vs_welsch_phi.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Figure saved to: {output_path_pdf}")
