#!/usr/bin/env python3
"""
Quick test script to verify the pipeline works correctly.

This script:
1. Tests imports
2. Detects GPUs
3. Tests on a small subset of data
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all required imports."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")

        import transformers
        print(f"✓ Transformers {transformers.__version__}")

        import datasets
        print(f"✓ Datasets {datasets.__version__}")

        from PIL import Image
        print(f"✓ PIL")

        from utils import SEMANTIC_CLASSES, CLASS_WEIGHTS, compute_locatability_score
        print(f"✓ Utils module (150 classes, {len(CLASS_WEIGHTS)} weights)")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_gpu():
    """Test GPU availability."""
    print("\n" + "=" * 60)
    print("Testing GPU...")
    print("=" * 60)

    import torch

    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Will use CPU (slow).")
        return True

    n_gpus = torch.cuda.device_count()
    print(f"✓ Found {n_gpus} GPU(s):")

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {memory_gb:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")

    return True


def test_model_loading():
    """Test model loading."""
    print("\n" + "=" * 60)
    print("Testing model loading...")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        model_id = "facebook/mask2former-swin-large-ade-semantic"
        print(f"Loading model: {model_id}")

        processor = AutoImageProcessor.from_pretrained(model_id)
        print(f"✓ Processor loaded")

        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
        print(f"✓ Model loaded")

        # Test inference on dummy image
        from PIL import Image
        import numpy as np

        dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        print(f"✓ Created dummy image: {dummy_image.size}")

        inputs = processor(images=dummy_image, return_tensors="pt")
        print(f"✓ Preprocessed image")

        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✓ Model forward pass successful")

        semantic_map = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(480, 640)]
        )[0]
        print(f"✓ Post-processing successful, shape: {semantic_map.shape}")

        return True

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_computation():
    """Test score computation."""
    print("\n" + "=" * 60)
    print("Testing score computation...")
    print("=" * 60)

    try:
        import torch
        import numpy as np
        from utils import SEMANTIC_CLASSES, CLASS_WEIGHTS, compute_locatability_score

        # Create dummy semantic map
        semantic_map = torch.randint(0, 150, (480, 640))
        weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)

        score, mapping_json = compute_locatability_score(
            semantic_map,
            weights_tensor,
            SEMANTIC_CLASSES
        )

        print(f"✓ Computed locatability score: {score:.4f}")
        print(f"✓ Class mapping length: {len(mapping_json)} chars")

        import json
        mapping = json.loads(mapping_json)
        print(f"✓ Parsed class mapping: {len(mapping)} classes")

        if mapping:
            top_class = max(mapping.items(), key=lambda x: x[1])
            print(f"  Top class: {top_class[0]} ({top_class[1]:.2%})")

        return True

    except Exception as e:
        print(f"✗ Score computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_access():
    """Test dataset access."""
    print("\n" + "=" * 60)
    print("Testing dataset access...")
    print("=" * 60)

    try:
        import datasets

        # Test paths
        test_paths = [
            "/mnt/tidal-alsh-hilab/dataset/diandian/user/geogussr/processed/gaea/train",
            "/mnt/tidal-alsh-hilab/dataset/diandian/user/geogussr/processed/osv5m/train",
        ]

        for path in test_paths:
            if os.path.exists(path):
                print(f"✓ Found dataset: {path}")
                try:
                    ds = datasets.load_from_disk(path)
                    print(f"  Size: {len(ds):,} samples")
                    print(f"  Columns: {ds.column_names}")
                except Exception as e:
                    print(f"  ⚠ Could not load: {e}")
            else:
                print(f"⚠ Not found: {path}")

        return True

    except Exception as e:
        print(f"✗ Dataset access failed: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("  Locatability Score Computation Pipeline - Test Suite")
    print("=" * 70 + "\n")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("GPU Detection", test_gpu()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Score Computation", test_score_computation()))
    results.append(("Dataset Access", test_dataset_access()))

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print("=" * 70)
    print(f"  {passed}/{total} tests passed")
    print("=" * 70 + "\n")

    if passed == total:
        print("✓ All tests passed! Ready to process datasets.")
        print("\nQuick start:")
        print("  cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/compute_locatability_score")
        print("  ./run_batch.sh --gaea")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
