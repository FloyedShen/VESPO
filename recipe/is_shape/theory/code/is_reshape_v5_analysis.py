"""
IS Reshape v5 Numerical Analysis and Visualization

This script provides numerical verification of the IS Reshape v5 theory.
Run this to visualize the gradient weight function behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Helper functions
def sech2(x: np.ndarray) -> np.ndarray:
    """Compute sech²(x) = 1/cosh²(x)"""
    return 1.0 / np.cosh(x)**2


def compute_v5_weight(
    w: np.ndarray,
    A: np.ndarray,
    tau_h: float = 1.0,
    tau_c: float = 1.0,
    gamma_base: float = 0.1,
    gamma_max: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute IS Reshape v5 gradient weight.

    Returns:
        g: gradient weight
        gamma: adaptive γ value
        C: correctness metric
        h_w: trust region factor
    """
    log_w = np.log(w)

    # Correctness metric
    C = A * log_w

    # P_wrong = σ(-C/τ_c)
    P_wrong = 1.0 / (1.0 + np.exp(C / tau_c))

    # Adaptive γ
    gamma = gamma_base + (gamma_max - gamma_base) * P_wrong

    # Trust region
    h_w = sech2(tau_h * (w - 1) / 2)

    # Gradient weight
    rl_weight = w * h_w
    g = (1 - gamma) + gamma * rl_weight

    return g, gamma, C, h_w


def analyze_four_quadrants():
    """Analyze all four quadrants with typical values."""
    print("=" * 70)
    print("Four Quadrant Analysis")
    print("=" * 70)

    cases = [
        ("Case 1: w>1, A>0 (correct)", 2.0, 1.0),
        ("Case 2: w>1, A<0 (wrong)", 3.0, -1.0),
        ("Case 3: w<1, A>0 (wrong)", 0.5, 1.0),
        ("Case 4: w<1, A<0 (correct)", 0.5, -1.0),
    ]

    for name, w, A in cases:
        g, gamma, C, h_w = compute_v5_weight(np.array([w]), np.array([A]))

        print(f"\n{name}")
        print(f"  w = {w:.2f}, A = {A:.1f}")
        print(f"  C = A·log(w) = {C[0]:.3f}")
        print(f"  P_wrong = σ(-C/τ) = {1/(1+np.exp(C[0])):.3f}")
        print(f"  γ = {gamma[0]:.3f}")
        print(f"  h(w) = sech²(τ(w-1)/2) = {h_w[0]:.3f}")
        print(f"  w·h(w) = {(w * h_w)[0]:.3f}")
        print(f"  g(w,A) = {g[0]:.3f}")

        # Interpretation
        if C[0] > 0:
            status = "Policy CORRECT"
        else:
            status = "Policy WRONG"

        if A > 0:
            direction = "increase π"
        else:
            direction = "decrease π"

        print(f"  Status: {status}")
        print(f"  Gradient direction: {direction}")
        print(f"  Effective update magnitude: {g[0]:.3f}")


def analyze_extreme_cases():
    """Analyze extreme w values."""
    print("\n" + "=" * 70)
    print("Extreme Cases Analysis")
    print("=" * 70)

    extreme_cases = [
        ("Extreme w=10, A=+1", 10.0, 1.0),
        ("Extreme w=10, A=-1", 10.0, -1.0),
        ("Extreme w=0.1, A=+1", 0.1, 1.0),
        ("Extreme w=0.1, A=-1", 0.1, -1.0),
    ]

    for name, w, A in extreme_cases:
        g, gamma, C, h_w = compute_v5_weight(np.array([w]), np.array([A]))

        print(f"\n{name}")
        print(f"  w = {w:.1f}, A = {A:.1f}")
        print(f"  γ = {gamma[0]:.3f}, h(w) = {h_w[0]:.6f}")
        print(f"  g(w,A) = {g[0]:.3f}")

        # Compare with other methods
        g_sft = 1.0
        g_rl = w
        g_ppo_clip = np.clip(w, 0.8, 1.2)

        print(f"  Comparison: SFT={g_sft:.1f}, RL={g_rl:.1f}, PPO={g_ppo_clip:.1f}, v5={g[0]:.3f}")


def plot_weight_function():
    """Plot the gradient weight function."""
    w = np.linspace(0.1, 5, 200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: g(w) for different A values (fixed γ_base=0.1, γ_max=0.9)
    ax = axes[0, 0]
    for A_val in [1.0, 0.5, 0, -0.5, -1.0]:
        A = np.full_like(w, A_val)
        g, _, _, _ = compute_v5_weight(w, A)
        ax.plot(w, g, label=f'A={A_val}')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='SFT (g=1)')
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('w (IS ratio)')
    ax.set_ylabel('g(w, A) (gradient weight)')
    ax.set_title('Gradient Weight vs IS Ratio (v5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)

    # Plot 2: Compare with PPO clip and vanilla IS
    ax = axes[0, 1]
    A = np.ones_like(w)  # A = 1
    g_v5, _, _, _ = compute_v5_weight(w, A)
    g_sft = np.ones_like(w)
    g_rl = w
    g_ppo = np.clip(w, 0.8, 1.2)

    ax.plot(w, g_v5, label='v5 (A=1)', linewidth=2)
    ax.plot(w, g_sft, label='SFT (g=1)', linestyle='--')
    ax.plot(w, g_rl, label='RL (g=w)', linestyle='--')
    ax.plot(w, g_ppo, label='PPO clip', linestyle='--')
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('w (IS ratio)')
    ax.set_ylabel('g(w) (gradient weight)')
    ax.set_title('Comparison with Other Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)

    # Plot 3: Trust region h(w)
    ax = axes[1, 0]
    h_w = sech2((w - 1) / 2)  # tau_h = 1
    ax.plot(w, h_w, label='h(w) = sech²(τ(w-1)/2)', linewidth=2)
    ax.plot(w, w * h_w, label='w·h(w)', linewidth=2)
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('w (IS ratio)')
    ax.set_ylabel('Value')
    ax.set_title('Trust Region Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: γ(w, A) heatmap
    ax = axes[1, 1]
    w_grid = np.linspace(0.2, 4, 50)
    A_grid = np.linspace(-2, 2, 50)
    W, A_mesh = np.meshgrid(w_grid, A_grid)

    _, gamma_grid, _, _ = compute_v5_weight(W.flatten(), A_mesh.flatten())
    gamma_grid = gamma_grid.reshape(W.shape)

    im = ax.imshow(gamma_grid, extent=[0.2, 4, -2, 2], aspect='auto', origin='lower', cmap='viridis')
    ax.axvline(x=1, color='white', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.set_xlabel('w (IS ratio)')
    ax.set_ylabel('A (advantage)')
    ax.set_title('γ(w, A) Heatmap')
    plt.colorbar(im, ax=ax, label='γ')

    # Add quadrant labels
    ax.text(2.5, 1, 'Q1: C>0\nγ low', color='white', ha='center', fontsize=9)
    ax.text(2.5, -1, 'Q2: C<0\nγ high', color='white', ha='center', fontsize=9)
    ax.text(0.5, 1, 'Q3: C<0\nγ high', color='white', ha='center', fontsize=9)
    ax.text(0.5, -1, 'Q4: C>0\nγ low', color='white', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('is_reshape_v5_analysis.png', dpi=150)
    print("\nPlot saved to: is_reshape_v5_analysis.png")
    plt.show()


def parameter_sensitivity_analysis():
    """Analyze sensitivity to parameters."""
    print("\n" + "=" * 70)
    print("Parameter Sensitivity Analysis")
    print("=" * 70)

    w = np.array([2.0])  # Fixed w for analysis
    A = np.array([1.0])  # Fixed A for analysis

    print("\n1. Effect of tau_h (trust region width):")
    for tau_h in [0.5, 1.0, 1.5, 2.0]:
        g, _, _, h_w = compute_v5_weight(w, A, tau_h=tau_h)
        print(f"   tau_h={tau_h}: h(w)={h_w[0]:.3f}, g={g[0]:.3f}")

    print("\n2. Effect of tau_c (correctness sensitivity):")
    for tau_c in [0.5, 1.0, 2.0, 5.0]:
        g, gamma, _, _ = compute_v5_weight(w, A, tau_c=tau_c)
        print(f"   tau_c={tau_c}: γ={gamma[0]:.3f}, g={g[0]:.3f}")

    print("\n3. Effect of gamma_base/gamma_max:")
    configs = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
    for gamma_base, gamma_max in configs:
        g, gamma, _, _ = compute_v5_weight(w, A, gamma_base=gamma_base, gamma_max=gamma_max)
        print(f"   γ_base={gamma_base}, γ_max={gamma_max}: γ={gamma[0]:.3f}, g={g[0]:.3f}")


def compare_v5_vs_v31():
    """Compare v5 with v3.1 (w^γ)."""
    print("\n" + "=" * 70)
    print("v5 vs v3.1 Comparison")
    print("=" * 70)

    test_cases = [
        (1.0, 1.0, "on-policy, A>0"),
        (2.0, 1.0, "slight off-policy, A>0"),
        (5.0, 1.0, "off-policy, A>0"),
        (10.0, -1.0, "very off-policy, A<0"),
    ]

    print("\nw     | A   | v3.1 (w^0.5) | v5        | Ratio")
    print("-" * 55)

    for w_val, A_val, desc in test_cases:
        w = np.array([w_val])
        A = np.array([A_val])

        # v3.1 style: w^γ with fixed γ=0.5
        g_v31 = w_val ** 0.5

        # v5
        g_v5, _, _, _ = compute_v5_weight(w, A)

        ratio = g_v31 / g_v5[0] if g_v5[0] > 0 else float('inf')
        print(f"{w_val:5.1f} | {A_val:3.1f} | {g_v31:12.3f} | {g_v5[0]:9.3f} | {ratio:.2f}x")


def main():
    """Run all analyses."""
    print("IS Reshape v5 Numerical Analysis")
    print("=" * 70)

    analyze_four_quadrants()
    analyze_extreme_cases()
    parameter_sensitivity_analysis()
    compare_v5_vs_v31()

    # Only plot if matplotlib is available
    try:
        plot_weight_function()
    except Exception as e:
        print(f"\nPlotting skipped: {e}")


if __name__ == "__main__":
    main()
