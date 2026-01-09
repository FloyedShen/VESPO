"""
IS Reshape v6: Kernel Function Analysis

This script analyzes different localization kernels for IS Reshape.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple

# =============================================================================
# Kernel Functions
# =============================================================================

def kernel_gaussian(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian kernel: K(x) = exp(-x²/2σ²)"""
    return np.exp(-x**2 / (2 * sigma**2))


def kernel_sech2(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Logistic/sech² kernel: K(x) = sech²(x/τ)"""
    return 1.0 / np.cosh(x / tau)**2


def kernel_cauchy(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Cauchy kernel: K(x) = 1/(1 + x²/γ²)"""
    return 1.0 / (1.0 + x**2 / scale**2)


def kernel_laplace(x: np.ndarray, b: float = 1.0) -> np.ndarray:
    """Laplace kernel: K(x) = exp(-|x|/b)"""
    return np.exp(-np.abs(x) / b)


def kernel_epanechnikov(x: np.ndarray, h: float = 1.0) -> np.ndarray:
    """Epanechnikov kernel: K(x) = (1 - x²/h²)₊ (bounded support)"""
    return np.maximum(0, 1 - x**2 / h**2)


# =============================================================================
# Gradient Weight Functions
# =============================================================================

def gradient_weight(r: np.ndarray, kernel: Callable, gamma: float = 1.0, **kwargs) -> np.ndarray:
    """
    Compute gradient weight: g(r) = (1-γ) + γ·r·K(r-1)
    """
    K = kernel(r - 1, **kwargs)
    return (1 - gamma) + gamma * r * K


def compute_f_function(r: np.ndarray, kernel: Callable, gamma: float = 1.0, **kwargs) -> np.ndarray:
    """
    Numerically compute f(r) = (1-γ)log(r) + γ·∫₁ʳ K(t-1) dt
    """
    # The (1-γ)log(r) part
    sft_part = (1 - gamma) * np.log(r)

    # Numerical integration for the kernel part
    kernel_part = np.zeros_like(r)
    for i, r_val in enumerate(r):
        if r_val >= 1:
            t = np.linspace(1, r_val, 1000)
            K_vals = kernel(t - 1, **kwargs)
            kernel_part[i] = np.trapz(K_vals, t)
        else:
            t = np.linspace(r_val, 1, 1000)
            K_vals = kernel(t - 1, **kwargs)
            kernel_part[i] = -np.trapz(K_vals, t)

    return sft_part + gamma * kernel_part


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_kernels():
    """Compare different kernel functions."""
    print("=" * 70)
    print("Kernel Function Comparison")
    print("=" * 70)

    x = np.linspace(-3, 3, 100)

    kernels = {
        'Gaussian (σ=1)': (kernel_gaussian, {'sigma': 1.0}),
        'sech² (τ=1)': (kernel_sech2, {'tau': 1.0}),
        'Cauchy (γ=1)': (kernel_cauchy, {'scale': 1.0}),
        'Laplace (b=1)': (kernel_laplace, {'b': 1.0}),
        'Epanechnikov (h=2)': (kernel_epanechnikov, {'h': 2.0}),
    }

    print("\nKernel values at different distances from center:")
    print(f"{'Kernel':<20} | K(0) | K(1) | K(2) | K(3)")
    print("-" * 60)

    for name, (kernel, kwargs) in kernels.items():
        vals = [kernel(np.array([d]), **kwargs)[0] for d in [0, 1, 2, 3]]
        print(f"{name:<20} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f}")

    return kernels


def analyze_gradient_weights():
    """Analyze gradient weights for different kernels."""
    print("\n" + "=" * 70)
    print("Gradient Weight Analysis: g(r) = r·K(r-1)")
    print("=" * 70)

    r_vals = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

    kernels = {
        'Gaussian': (kernel_gaussian, {'sigma': 1.0}),
        'sech²': (kernel_sech2, {'tau': 1.0}),
        'Cauchy': (kernel_cauchy, {'scale': 1.0}),
        'Laplace': (kernel_laplace, {'b': 1.0}),
    }

    print(f"\nGradient weight g(r) = r·K(r-1) at different r:")
    header = f"{'r':<6} | " + " | ".join(f"{name:<10}" for name in kernels.keys())
    print(header)
    print("-" * len(header))

    for r in r_vals:
        r_arr = np.array([r])
        weights = []
        for name, (kernel, kwargs) in kernels.items():
            K = kernel(r_arr - 1, **kwargs)[0]
            g = r * K
            weights.append(f"{g:<10.3f}")
        print(f"{r:<6.1f} | " + " | ".join(weights))

    # Find peak for each kernel
    print("\nPeak of g(r) = r·K(r-1):")
    r_fine = np.linspace(0.1, 5, 1000)
    for name, (kernel, kwargs) in kernels.items():
        g = r_fine * kernel(r_fine - 1, **kwargs)
        peak_idx = np.argmax(g)
        print(f"  {name}: peak at r={r_fine[peak_idx]:.2f}, g_max={g[peak_idx]:.3f}")


def analyze_mixed_weights():
    """Analyze mixed weights with different γ."""
    print("\n" + "=" * 70)
    print("Mixed Weight Analysis: g(r) = (1-γ) + γ·r·K(r-1)")
    print("=" * 70)

    # Use sech² kernel
    kernel = kernel_sech2
    kwargs = {'tau': 1.0}

    r_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    gamma_vals = [0.0, 0.3, 0.5, 0.7, 1.0]

    print(f"\nGradient weight with sech² kernel (τ=1):")
    header = f"{'r':<6} | " + " | ".join(f"γ={g:<4}" for g in gamma_vals)
    print(header)
    print("-" * len(header))

    for r in r_vals:
        r_arr = np.array([r])
        weights = []
        for gamma in gamma_vals:
            g = gradient_weight(r_arr, kernel, gamma=gamma, **kwargs)[0]
            weights.append(f"{g:<6.3f}")
        print(f"{r:<6.1f} | " + " | ".join(weights))


def verify_localization_property():
    """Verify that all kernels give bounded gradient weights."""
    print("\n" + "=" * 70)
    print("Localization Property Verification")
    print("=" * 70)

    kernels = {
        'Gaussian': (kernel_gaussian, {'sigma': 1.0}),
        'sech²': (kernel_sech2, {'tau': 1.0}),
        'Cauchy': (kernel_cauchy, {'scale': 1.0}),
        'Laplace': (kernel_laplace, {'b': 1.0}),
    }

    # Test at extreme r values
    r_extreme = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

    print("\nGradient weight at extreme r values (γ=0.9):")
    print(f"{'r':<10} | " + " | ".join(f"{name:<10}" for name in kernels.keys()))
    print("-" * 70)

    for r in r_extreme:
        r_arr = np.array([r])
        weights = []
        for name, (kernel, kwargs) in kernels.items():
            g = gradient_weight(r_arr, kernel, gamma=0.9, **kwargs)[0]
            weights.append(f"{g:<10.4f}")
        print(f"{r:<10.2f} | " + " | ".join(weights))

    print("\nKey observation: All kernels give bounded weights!")
    print("As r → ∞: g(r) → (1-γ) = 0.1 (the SFT baseline)")


def compare_with_v31():
    """Compare v6 kernels with v3.1's w^γ."""
    print("\n" + "=" * 70)
    print("Comparison: v6 Kernels vs v3.1 (w^γ)")
    print("=" * 70)

    r_vals = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

    print(f"\nGradient weights at different r:")
    print(f"{'r':<8} | {'v3.1 (r^0.5)':<12} | {'Gaussian':<12} | {'sech²':<12} | {'Cauchy':<12}")
    print("-" * 70)

    for r in r_vals:
        r_arr = np.array([r])

        # v3.1: w^γ with γ=0.5
        g_v31 = r ** 0.5

        # v6 with different kernels (γ=0.9 for RL-like behavior)
        g_gaussian = gradient_weight(r_arr, kernel_gaussian, gamma=0.9, sigma=1.0)[0]
        g_sech2 = gradient_weight(r_arr, kernel_sech2, gamma=0.9, tau=1.0)[0]
        g_cauchy = gradient_weight(r_arr, kernel_cauchy, gamma=0.9, scale=1.0)[0]

        print(f"{r:<8.1f} | {g_v31:<12.3f} | {g_gaussian:<12.3f} | {g_sech2:<12.3f} | {g_cauchy:<12.3f}")

    print("\nKey insight: v3.1 grows unboundedly, v6 kernels stay bounded!")


def plot_analysis():
    """Create comprehensive plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Kernel functions K(x)
    ax = axes[0, 0]
    x = np.linspace(-3, 3, 200)

    ax.plot(x, kernel_gaussian(x, sigma=1.0), label='Gaussian (σ=1)', linewidth=2)
    ax.plot(x, kernel_sech2(x, tau=1.0), label='sech² (τ=1)', linewidth=2)
    ax.plot(x, kernel_cauchy(x, scale=1.0), label='Cauchy (γ=1)', linewidth=2)
    ax.plot(x, kernel_laplace(x, b=1.0), label='Laplace (b=1)', linewidth=2)

    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('x = r - 1')
    ax.set_ylabel('K(x)')
    ax.set_title('Localization Kernels')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gradient weights g(r) = r·K(r-1)
    ax = axes[0, 1]
    r = np.linspace(0.1, 5, 200)

    ax.plot(r, r * kernel_gaussian(r-1, sigma=1.0), label='Gaussian', linewidth=2)
    ax.plot(r, r * kernel_sech2(r-1, tau=1.0), label='sech²', linewidth=2)
    ax.plot(r, r * kernel_cauchy(r-1, scale=1.0), label='Cauchy', linewidth=2)
    ax.plot(r, r * kernel_laplace(r-1, b=1.0), label='Laplace', linewidth=2)
    ax.plot(r, r, '--', label='IS (g=r)', alpha=0.5)
    ax.axhline(y=1, linestyle=':', color='gray', alpha=0.5, label='SFT (g=1)')

    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('r (IS ratio)')
    ax.set_ylabel('g(r) = r·K(r-1)')
    ax.set_title('Gradient Weights (pure kernel, γ=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)

    # Plot 3: Mixed weights with different γ (sech² kernel)
    ax = axes[0, 2]
    r = np.linspace(0.1, 5, 200)

    for gamma in [0.0, 0.3, 0.5, 0.7, 1.0]:
        g = gradient_weight(r, kernel_sech2, gamma=gamma, tau=1.0)
        ax.plot(r, g, label=f'γ={gamma}', linewidth=2)

    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('r (IS ratio)')
    ax.set_ylabel('g(r)')
    ax.set_title('Mixed Weights: (1-γ) + γ·r·K(r-1) [sech²]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Comparison with v3.1
    ax = axes[1, 0]
    r = np.linspace(0.1, 10, 200)

    ax.plot(r, r**0.5, '--', label='v3.1: r^0.5', linewidth=2)
    ax.plot(r, r**0.3, '--', label='v3.1: r^0.3', linewidth=2)
    ax.plot(r, gradient_weight(r, kernel_sech2, gamma=0.9, tau=1.0),
            label='v6: sech² (γ=0.9)', linewidth=2)
    ax.plot(r, gradient_weight(r, kernel_gaussian, gamma=0.9, sigma=1.0),
            label='v6: Gaussian (γ=0.9)', linewidth=2)

    ax.axhline(y=0.1, linestyle=':', color='gray', alpha=0.5, label='Limit (1-γ)=0.1')
    ax.set_xlabel('r (IS ratio)')
    ax.set_ylabel('g(r)')
    ax.set_title('v6 vs v3.1: Gradient Weights')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4)

    # Plot 5: f(r) functions
    ax = axes[1, 1]
    r = np.linspace(0.2, 4, 100)

    # Compute f functions numerically
    f_sech2 = compute_f_function(r, kernel_sech2, gamma=0.9, tau=1.0)
    f_gaussian = compute_f_function(r, kernel_gaussian, gamma=0.9, sigma=1.0)
    f_log = np.log(r)  # SFT: log(r)
    f_linear = r - 1   # IS: r (shifted)

    ax.plot(r, f_sech2, label='v6: sech² (γ=0.9)', linewidth=2)
    ax.plot(r, f_gaussian, label='v6: Gaussian (γ=0.9)', linewidth=2)
    ax.plot(r, 0.1 * f_log, '--', label='SFT: 0.1·log(r)', alpha=0.7)

    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('r (IS ratio)')
    ax.set_ylabel('f(r)')
    ax.set_title('Objective Functions f(r)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Extreme behavior
    ax = axes[1, 2]
    r = np.logspace(-1, 3, 200)  # 0.1 to 1000

    ax.semilogx(r, r**0.5, '--', label='v3.1: r^0.5', linewidth=2)
    ax.semilogx(r, gradient_weight(r, kernel_sech2, gamma=0.9, tau=1.0),
                label='v6: sech² (γ=0.9)', linewidth=2)
    ax.semilogx(r, gradient_weight(r, kernel_cauchy, gamma=0.9, scale=1.0),
                label='v6: Cauchy (γ=0.9)', linewidth=2)

    ax.axhline(y=0.1, linestyle=':', color='gray', alpha=0.5, label='Limit (1-γ)')
    ax.set_xlabel('r (IS ratio, log scale)')
    ax.set_ylabel('g(r)')
    ax.set_title('Extreme Behavior (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 35)

    plt.tight_layout()
    plt.savefig('is_reshape_v6_kernel_analysis.png', dpi=150)
    print("\nPlot saved to: is_reshape_v6_kernel_analysis.png")
    plt.show()


def main():
    """Run all analyses."""
    print("IS Reshape v6: Kernel Function Analysis")
    print("=" * 70)

    analyze_kernels()
    analyze_gradient_weights()
    analyze_mixed_weights()
    verify_localization_property()
    compare_with_v31()

    try:
        plot_analysis()
    except Exception as e:
        print(f"\nPlotting skipped: {e}")


if __name__ == "__main__":
    main()
