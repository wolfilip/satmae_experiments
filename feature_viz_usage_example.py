#!/usr/bin/env python3
"""
Simple example showing how to use the feature visualization functions.
This can be used as a reference for integration into other scripts.
"""

import torch
import numpy as np

# Example: Simulating model forward pass with feature extraction


def example_model_with_features():
    """
    Simulates a model that returns predictions and intermediate features.
    In your actual code, this would be your segmentation model.
    """
    # Simulate a batch of predictions
    batch_size = 4
    num_classes = 10
    height, width = 224, 224

    # Predictions: (B, C, H, W)
    predictions = torch.randn(batch_size, num_classes, height, width)

    # Features: typically returned as [conv_features, vit_features]
    # Conv features: (B, 256, 28, 28) - from conv encoder if present
    conv_features = torch.randn(batch_size, 256, 28, 28)

    # ViT features: (B, 196, 768) - from ViT backbone (14x14 patches, 768-dim)
    vit_features = torch.randn(batch_size, 196, 768)

    features = [conv_features, vit_features]

    return predictions, features


def visualize_batch_example():
    """
    Example showing how to visualize features for a batch of images.
    """
    print("=" * 60)
    print("Feature Visualization Usage Example")
    print("=" * 60)

    # Import the visualization functions
    # Note: In actual use, these are already in engine_finetune.py
    # from engine_finetune import visualize_features, visualize_feature_quality

    print("\n1. Simulating model forward pass...")
    predictions, features = example_model_with_features()
    conv_features, vit_features = features

    print(f"   Conv features shape: {conv_features.shape}")
    print(f"   ViT features shape: {vit_features.shape}")

    print("\n2. Generating PCA visualizations...")
    print("   Note: In save_images(), this is done like:")
    print("   ```python")
    print("   if args.visualize_features:")
    print("       if features[0] != 0:")
    print("           viz_conv_1, viz_conv_2 = visualize_features(features[0], True)")
    print("       viz_vit_1, viz_vit_2 = visualize_features(features[1], False)")
    print("   ```")

    print("\n3. Computing quality metrics...")
    print("   Note: Optional, use with --save_feature_metrics flag:")
    print("   ```python")
    print("   if args.save_feature_metrics:")
    print("       vit_metrics = visualize_feature_quality(features[1], is_conv=False)")
    print("       conv_metrics = visualize_feature_quality(features[0], is_conv=True)")
    print("   ```")

    print("\n4. Displaying in subplot grid...")
    print("   The save_images() function creates a figure with:")
    print("   - Row 0: Original input image")
    print("   - Row 1: Ground truth mask")
    print("   - Row 2: Predicted mask")
    print("   - Row 3: ViT feature visualization (PCA)")
    print("   - Row 4: Conv feature visualization (PCA) [if present]")

    print("\n" + "=" * 60)
    print("Integration Checklist:")
    print("=" * 60)
    print("✓ Import sklearn.decomposition.PCA at top of file")
    print("✓ Add visualize_features() function (already done)")
    print("✓ Add visualize_feature_quality() function (already done)")
    print("✓ Ensure model returns features in forward pass")
    print("✓ Call visualize_features() in save_images() (already done)")
    print("✓ Add --visualize_features flag to argument parser")
    print("✓ [Optional] Add --save_feature_metrics flag")
    print("✓ [Optional] Install scikit-learn: pip install scikit-learn")


def model_comparison_workflow():
    """
    Example workflow for comparing two models using feature visualizations.
    """
    print("\n" + "=" * 60)
    print("Model Comparison Workflow")
    print("=" * 60)

    print("\nStep 1: Run evaluation for Method A")
    print("---------------------------------------")
    print("$ python main_finetune.py \\")
    print("    --eval \\")
    print("    --visualize_features \\")
    print("    --save_images \\")
    print("    --save_feature_metrics \\")
    print("    --model vit_base_patch16 \\")
    print("    --finetune path/to/method_a_checkpoint.pth \\")
    print("    --dataset_type geobench_crop \\")
    print("    --method_name method_a")

    print("\nStep 2: Run evaluation for Method B")
    print("---------------------------------------")
    print("$ python main_finetune.py \\")
    print("    --eval \\")
    print("    --visualize_features \\")
    print("    --save_images \\")
    print("    --save_feature_metrics \\")
    print("    --model vit_large_patch16 \\")
    print("    --finetune path/to/method_b_checkpoint.pth \\")
    print("    --dataset_type geobench_crop \\")
    print("    --method_name method_b")

    print("\nStep 3: Visual comparison")
    print("-------------------------")
    print("Open side-by-side:")
    print("  outputs/images/geobench_crop_100pc_results/images/method_a/img_0.png")
    print("  outputs/images/geobench_crop_100pc_results/images/method_b/img_0.png")
    print("\nLook at rows 3-4 (feature visualizations):")
    print("  - More structured patterns = better features")
    print("  - Clearer object boundaries = better features")
    print("  - Less noise/randomness = better features")

    print("\nStep 4: Quantitative comparison")
    print("--------------------------------")
    print("Parse metrics files:")
    print("  outputs/.../per_image/method_a/feature_quality_metrics.txt")
    print("  outputs/.../per_image/method_b/feature_quality_metrics.txt")
    print("\nCompare average effective rank:")
    print("  Higher rank → more diverse/informative features")
    print("  (typically 100-300 for good features, <50 for poor)")

    print("\nStep 5: Create HTML comparison report")
    print("--------------------------------------")
    print("Use your existing comparison script to generate:")
    print("  outputs/images/geobench_crop_100pc_results/compare/")
    print("    method_a_vs_method_b.html")
    print("\nThis allows browsing all test images with IoU differences")
    print("and visual side-by-side comparison of features.")


def expected_feature_patterns():
    """
    Guide on interpreting feature visualizations.
    """
    print("\n" + "=" * 60)
    print("Interpreting Feature Visualizations")
    print("=" * 60)

    print("\n✅ GOOD FEATURES show:")
    print("  • Clear object boundaries with distinct colors")
    print("  • Smooth color transitions within semantic regions")
    print("  • Different colors for different object classes")
    print("  • Spatial structure matching input image layout")
    print("  • High effective rank (>150)")
    print("  • Low spatial smoothness (<0.01)")

    print("\n❌ POOR FEATURES show:")
    print("  • Random noise or speckled patterns")
    print("  • Uniform color across the image")
    print("  • No correspondence to input structure")
    print("  • Blocky artifacts or grid patterns")
    print("  • Low effective rank (<50)")
    print("  • High spatial smoothness (>0.1)")

    print("\n💡 WHAT TO LOOK FOR when comparing methods:")
    print("  1. Semantic Alignment:")
    print("     Do feature colors align with semantic regions?")
    print("     (e.g., all crops same color, all buildings another color)")

    print("\n  2. Spatial Coherence:")
    print("     Are nearby pixels with similar features similar in color?")
    print("     Smooth gradients within objects = good")

    print("\n  3. Discriminative Power:")
    print("     Can you distinguish different objects by color?")
    print("     More distinct colors = more discriminative")

    print("\n  4. Resolution:")
    print("     Are fine details preserved or blurred?")
    print("     Finer structure = better spatial resolution")

    print("\n  5. Consistency:")
    print("     Do similar images produce similar feature patterns?")
    print("     Consistency across dataset = robust features")


if __name__ == "__main__":
    # Run examples
    visualize_batch_example()
    model_comparison_workflow()
    expected_feature_patterns()

    print("\n" + "=" * 60)
    print("For full documentation, see:")
    print("  FEATURE_VISUALIZATION_README.md")
    print("=" * 60)
