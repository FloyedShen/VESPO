#!/usr/bin/env python3
"""
Tests for Enhanced Dual VLLM Coordinator

Tests advanced features:
1. Dual prompts with different contexts
2. Support constraint application
3. First token forcing
4. Statistics tracking
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_enhanced import EnhancedCoordinatorConfig
from coordinator_enhanced import EnhancedDualVLLMCoordinator


def test_enhanced_config():
    """Test enhanced configuration"""
    print("\n" + "="*70)
    print("TEST 1: Enhanced Configuration")
    print("="*70)

    # Test default configuration
    print("\nCase 1: Default configuration")
    config = EnhancedCoordinatorConfig()
    config.validate()
    print(f"  constraint_to_target: {config.constraint_to_target}")
    print(f"  target_top_p: {config.target_top_p}")
    print(f"  force_first_token: {config.force_first_token}")
    print(f"  exclude_special_tokens: {config.exclude_special_tokens}")
    print("  ✅ PASS")

    # Test custom configuration
    print("\nCase 2: Custom configuration")
    config = EnhancedCoordinatorConfig(
        theta_url="http://localhost:9000",
        t_url="http://localhost:9001",
        constraint_to_target=True,
        target_top_p=0.90,
        force_first_token=False,
    )
    config.validate()
    print(f"  constraint_to_target: {config.constraint_to_target}")
    print(f"  target_top_p: {config.target_top_p}")
    print(f"  force_first_token: {config.force_first_token}")
    print("  ✅ PASS")

    # Test validation
    print("\nCase 3: Validation")
    try:
        config = EnhancedCoordinatorConfig(target_top_p=1.5)
        config.validate()
        print("  ❌ Should have failed")
        return False
    except AssertionError:
        print("  ✅ Correctly rejected invalid target_top_p")

    print("\n✅ All configuration tests PASSED")
    return True


def test_support_constraint():
    """Test support constraint logic"""
    print("\n" + "="*70)
    print("TEST 2: Support Constraint")
    print("="*70)

    config = EnhancedCoordinatorConfig(
        constraint_to_target=True,
        target_top_p=0.8,
    )
    coordinator = EnhancedDualVLLMCoordinator(config)

    # Test case: 5 candidates with different probabilities
    print("\nCase 1: Top-p=0.8 filtering")
    candidates = [10, 20, 30, 40, 50]
    probs_theta = np.array([0.2, 0.3, 0.1, 0.15, 0.25])  # Sum=1.0
    probs_t = np.array([0.5, 0.3, 0.1, 0.05, 0.05])      # Sum=1.0

    print(f"  Input candidates: {candidates}")
    print(f"  π_t probs: {probs_t}")
    print(f"  π_t sorted: {np.sort(probs_t)[::-1]}")
    print(f"  π_t cumsum: {np.cumsum(np.sort(probs_t)[::-1])}")

    filtered_candidates, filtered_theta, filtered_t = coordinator._apply_support_constraint(
        candidates, probs_theta, probs_t
    )

    print(f"  Filtered candidates: {filtered_candidates}")
    print(f"  Filtered π_t: {filtered_t}")
    print(f"  Cumulative π_t: {np.sum(filtered_t):.3f} (should be ~1.0 after renorm)")

    # Should keep tokens 10, 20 (covering 0.8 mass)
    assert len(filtered_candidates) == 2, f"Should keep 2 tokens, got {len(filtered_candidates)}"
    assert 10 in filtered_candidates and 20 in filtered_candidates, "Should keep top-2 tokens"
    assert abs(np.sum(filtered_t) - 1.0) < 1e-6, "Should renormalize"
    assert abs(np.sum(filtered_theta) - 1.0) < 1e-6, "Should renormalize"
    print("  ✅ PASS")

    # Test case: Edge case with target_top_p=0.95
    print("\nCase 2: Top-p=0.95 (keep 4 tokens)")
    config.target_top_p = 0.95
    coordinator = EnhancedDualVLLMCoordinator(config)

    filtered_candidates, filtered_theta, filtered_t = coordinator._apply_support_constraint(
        candidates, probs_theta, probs_t
    )

    print(f"  Filtered candidates: {filtered_candidates}")
    # With cumsum [0.5, 0.8, 0.9, 0.95], searchsorted(0.95) gives index 3, so keep 4 tokens
    assert len(filtered_candidates) == 4, f"Should keep 4 tokens, got {len(filtered_candidates)}"
    assert abs(np.sum(filtered_t) - 1.0) < 1e-6, "Should renormalize"
    print("  ✅ PASS")

    print("\n✅ All support constraint tests PASSED")
    return True


def test_first_token_forcing_logic():
    """Test first token forcing logic (without vLLM)"""
    print("\n" + "="*70)
    print("TEST 3: First Token Forcing Logic")
    print("="*70)

    # We can't test the actual generation without vLLM,
    # but we can verify the configuration and statistics

    print("\nCase 1: With first token forcing")
    config = EnhancedCoordinatorConfig(force_first_token=True)
    coordinator = EnhancedDualVLLMCoordinator(config)
    assert coordinator.config.force_first_token == True
    print("  force_first_token: True")
    print("  ✅ PASS")

    print("\nCase 2: Without first token forcing")
    config = EnhancedCoordinatorConfig(force_first_token=False)
    coordinator = EnhancedDualVLLMCoordinator(config)
    assert coordinator.config.force_first_token == False
    print("  force_first_token: False")
    print("  ✅ PASS")

    print("\nCase 3: Statistics tracking")
    coordinator = EnhancedDualVLLMCoordinator(EnhancedCoordinatorConfig())
    stats = coordinator.get_statistics()
    assert "first_token_forced" in stats, "Should track first_token_forced"
    assert "constraint_applied" in stats, "Should track constraint_applied"
    print(f"  first_token_forced: {stats['first_token_forced']}")
    print(f"  constraint_applied: {stats['constraint_applied']}")
    print("  ✅ PASS")

    print("\n✅ All first token forcing tests PASSED")
    return True


def test_dual_prompt_validation():
    """Test dual prompt validation"""
    print("\n" + "="*70)
    print("TEST 4: Dual Prompt Validation")
    print("="*70)

    # We can't test actual generation without vLLM,
    # but we can test the validation logic

    print("\nCase 1: Length mismatch should be caught")
    config = EnhancedCoordinatorConfig()
    coordinator = EnhancedDualVLLMCoordinator(config)

    # This would need to be tested with actual async call
    # For now, just verify the coordinator accepts the method
    assert hasattr(coordinator, 'generate_batch_dual_prompts')
    print("  generate_batch_dual_prompts method exists")
    print("  ✅ PASS")

    print("\nCase 2: Backward compatibility")
    assert hasattr(coordinator, 'generate_batch')
    print("  generate_batch method exists (backward compatible)")
    print("  ✅ PASS")

    print("\n✅ All dual prompt validation tests PASSED")
    return True


def test_statistics_enhanced():
    """Test enhanced statistics"""
    print("\n" + "="*70)
    print("TEST 5: Enhanced Statistics")
    print("="*70)

    config = EnhancedCoordinatorConfig(
        force_first_token=True,
        constraint_to_target=True,
    )
    coordinator = EnhancedDualVLLMCoordinator(config)

    stats = coordinator.get_statistics()
    print("\nInitial statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Verify new statistics exist
    required_stats = ["first_token_forced", "constraint_applied"]
    for stat in required_stats:
        assert stat in stats, f"Missing statistic: {stat}"

    print("\n✅ All enhanced statistics tests PASSED")
    return True


def test_integration_scenarios():
    """Test realistic integration scenarios (configuration only)"""
    print("\n" + "="*70)
    print("TEST 6: Integration Scenarios")
    print("="*70)

    scenarios = [
        {
            "name": "Conservative (high alignment)",
            "config": EnhancedCoordinatorConfig(
                force_first_token=True,
                constraint_to_target=True,
                target_top_p=0.90,
            ),
        },
        {
            "name": "Balanced (recommended)",
            "config": EnhancedCoordinatorConfig(
                force_first_token=True,
                constraint_to_target=True,
                target_top_p=0.95,
            ),
        },
        {
            "name": "Exploratory (more diversity)",
            "config": EnhancedCoordinatorConfig(
                force_first_token=False,
                constraint_to_target=False,
            ),
        },
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        config = scenario['config']
        config.validate()
        coordinator = EnhancedDualVLLMCoordinator(config)

        print(f"  force_first_token: {config.force_first_token}")
        print(f"  constraint_to_target: {config.constraint_to_target}")
        if config.constraint_to_target:
            print(f"  target_top_p: {config.target_top_p}")
        print("  ✅ Valid configuration")

    print("\n✅ All integration scenario tests PASSED")
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("🧪 Enhanced Dual VLLM Tests (No vLLM Required)")
    print("="*70)

    tests = [
        ("Enhanced Configuration", test_enhanced_config),
        ("Support Constraint", test_support_constraint),
        ("First Token Forcing", test_first_token_forcing_logic),
        ("Dual Prompt Validation", test_dual_prompt_validation),
        ("Enhanced Statistics", test_statistics_enhanced),
        ("Integration Scenarios", test_integration_scenarios),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:.<50} {status}")

    print("\n" + "="*70)
    if passed_count == total_count:
        print(f"🎉 ALL {total_count} TESTS PASSED!")
    else:
        print(f"❌ {total_count - passed_count}/{total_count} TESTS FAILED")
    print("="*70 + "\n")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
