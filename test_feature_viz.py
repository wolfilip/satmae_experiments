#!/usr/bin/env python3
"""
Test script for feature visualization functions.
This verifies that the PCA-based feature visualization works correctly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import the visualization functions from engine_finetune
import sys

sys.path.insert(0, "/home/filip/satmae_experiments")


def test_visualize_features_conv():
    """Test visualization with convolutional features (B, C, H, W)"""
    print("Testing convolutional feature visualization...")

    # Simulate conv features: batch=2, channels=768, height=14, width=14
    B, C, H, W = 2, 768, 14, 14
    features = torch.randn(B, C, H, W)

    # Import the function
    from engine_finetune import visualize_features

    viz_1, viz_2 = visualize_features(features, is_conv=True)

    # Check output shapes
    assert viz_1.shape == (3, H, W), f"Expected shape (3, {H}, {W}), got {viz_1.shape}"
    assert viz_2.shape == (3, H, W), f"Expected shape (3, {H}, {W}), got {viz_2.shape}"

    # Check value range [0, 1]
    assert viz_1.min() >= 0 and viz_1.max() <= 1, "Values should be in [0, 1]"
    assert viz_2.min() >= 0 and viz_2.max() <= 1, "Values should be in [0, 1]"

    print("✓ Convolutional feature visualization passed!")
    return viz_1, viz_2


def test_visualize_features_vit():
    """Test visualization with ViT features (B, N, C)"""
    print("Testing ViT feature visualization...")

    # Simulate ViT features: batch=2, num_patches=196 (14x14), channels=768
    B, N, C = 2, 196, 768
    features = torch.randn(B, N, C)

    # Import the function
    from engine_finetune import visualize_features

    viz_1, viz_2 = visualize_features(features, is_conv=False)

    # Check output shapes
    H = W = int(np.sqrt(N))
    assert viz_1.shape == (3, H, W), f"Expected shape (3, {H}, {W}), got {viz_1.shape}"
    assert viz_2.shape == (3, H, W), f"Expected shape (3, {H}, {W}), got {viz_2.shape}"

    # Check value range [0, 1]
    assert viz_1.min() >= 0 and viz_1.max() <= 1, "Values should be in [0, 1]"
    assert viz_2.min() >= 0 and viz_2.max() <= 1, "Values should be in [0, 1]"

    print("✓ ViT feature visualization passed!")
    return viz_1, viz_2


def test_visualize_feature_quality():
    """Test feature quality metrics computation"""
    print("Testing feature quality metrics...")

    from engine_finetune import visualize_feature_quality

    # Test with ViT features
    B, N, C = 2, 196, 768
    features = torch.randn(B, N, C)

    metrics = visualize_feature_quality(features, is_conv=False)

    assert metrics is not None, "Should return metrics dictionary"
    assert "effective_rank_0" in metrics, "Should contain effective rank for batch 0"
    assert "spatial_smoothness" in metrics, "Should contain spatial smoothness"
    assert "mean_norm" in metrics, "Should contain mean norm"
    assert "std_norm" in metrics, "Should contain std norm"

    print(f"✓ Feature quality metrics computed:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    return metrics


def visualize_comparison():
    """Create a visual comparison of different feature qualities"""
    print("\nCreating visual comparison...")

    from engine_finetune import visualize_features

    # Simulate two different feature sets with different properties
    B, N, C = 1, 196, 384

    # High-quality features: structured patterns
    H = W = int(np.sqrt(N))
    x = torch.linspace(-1, 1, H)
    y = torch.linspace(-1, 1, W)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    features_good = torch.randn(B, N, C)
    # Add spatial structure to some channels
    for i in range(10):
        spatial_pattern = (torch.sin(5 * xx) * torch.cos(5 * yy)).reshape(-1)
        features_good[0, :, i] += spatial_pattern

    # Low-quality features: mostly noise
    features_bad = torch.randn(B, N, C) * 0.1

    # Visualize both
    viz_good_1, viz_good_2 = visualize_features(features_good, is_conv=False)
    viz_bad_1, viz_bad_2 = visualize_features(features_bad, is_conv=False)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(viz_good_1.permute(1, 2, 0).numpy())
    axes[0].set_title("Structured Features (Good)")
    axes[0].axis("off")

    axes[1].imshow(viz_bad_1.permute(1, 2, 0).numpy())
    axes[1].set_title("Noisy Features (Bad)")
    axes[1].axis("off")

    plt.tight_layout()
    output_path = "/home/filip/satmae_experiments/feature_viz_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved comparison visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Feature Visualization Test Suite")
    print("=" * 60)

    # Run tests
    viz_conv_1, viz_conv_2 = test_visualize_features_conv()
    print()

    viz_vit_1, viz_vit_2 = test_visualize_features_vit()
    print()

    metrics = test_visualize_feature_quality()
    print()

    visualize_comparison()
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nThe visualize_features function:")
    print("  1. Takes feature tensors (conv or ViT format)")
    print("  2. Applies PCA to reduce to 3 RGB components")
    print("  3. Enhances with feature norm intensity modulation")
    print("  4. Returns normalized [0,1] visualizations")
    print("\nThe visualize_feature_quality function:")
    print("  1. Computes effective rank (feature diversity)")
    print("  2. Measures spatial smoothness (local coherence)")
    print("  3. Calculates feature norm statistics")
    print("  4. Returns metrics dictionary for comparison")
