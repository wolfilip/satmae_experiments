#!/usr/bin/env python3
"""
Generate an HTML comparison file for spacenet dataset predictions.
Compares three methods and sorts by IoU difference.
"""

import os
from pathlib import Path
import re

# Configuration
BASE_DIR = Path("/home/filip/satmae_experiments/outputs/images/spacenet_100pc_results")
INDIVIDUAL_IMAGES_DIR = BASE_DIR / "individual_images"
PER_IMAGE_DIR = BASE_DIR / "per_image"
OUTPUT_HTML = BASE_DIR / "comparison_miou_diff.html"

# Methods to compare
METHOD_NAMES = [
    "copernicusfm_vit_b",
    "dinov3_LS",
    "swin_bm_ms_fmow_hr_280k_100e_dinov3p34c_rgb_head_dconv_aug",
]


def find_method_folder(method_name):
    """Find the timestamped folder for a method."""
    pattern = re.compile(rf"{re.escape(method_name)}_(\d{{8}}_\d{{6}})$")
    for folder in INDIVIDUAL_IMAGES_DIR.iterdir():
        if folder.is_dir() and pattern.search(folder.name):
            return folder
    return None


def parse_iou_file(method_name):
    """Parse IoU values from the text file."""
    iou_file = PER_IMAGE_DIR / method_name / "image_results_iou_-1.txt"
    ious = {}
    with open(iou_file, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                img_id, iou_val = line.split(":")
                img_id = img_id.strip()
                iou_val = float(iou_val.strip())
                ious[img_id] = iou_val
    return ious


def get_iou_color(iou):
    """Return color class based on IoU value."""
    if iou >= 0.85:
        return "good"
    elif iou >= 0.70:
        return "mid"
    else:
        return "bad"


def generate_html():
    """Generate the HTML comparison file."""

    # Find timestamped folders for each method
    method_folder_map = {}
    for method in METHOD_NAMES:
        folder = find_method_folder(method)
        if folder:
            method_folder_map[method] = folder
            print(f"[INFO] Found folder for {method}: {folder.name}")
        else:
            print(f"[WARNING] No folder found for {method}")

    if not method_folder_map:
        print("[ERROR] No method folders found!")
        return

    # Determine reference folder for RGB and GT (prefer dinov3_LS)
    reference_folder = None
    if "dinov3_LS" in method_folder_map:
        reference_folder = method_folder_map["dinov3_LS"]
    else:
        # Fallback to first available method
        for m in METHOD_NAMES:
            if m in method_folder_map:
                reference_folder = method_folder_map[m]
                break

    print(f"[INFO] Using {reference_folder.name} for RGB and GT images")

    # Parse IoU files
    method_ious = {}
    for method in METHOD_NAMES:
        try:
            ious = parse_iou_file(method)
            method_ious[method] = ious
            print(f"[INFO] Parsed {len(ious)} IoUs for {method}")
        except Exception as e:
            print(f"[WARNING] Failed to parse IoUs for {method}: {e}")

    if not method_ious:
        print("[ERROR] No IoU data parsed!")
        return

    # Get common image IDs
    all_img_ids = set.intersection(*[set(ious.keys()) for ious in method_ious.values()])
    print(f"[INFO] Found {len(all_img_ids)} common images")

    # Calculate IoU difference (max - min) for each image
    img_data = []
    for img_id in all_img_ids:
        ious = [method_ious[method][img_id] for method in METHOD_NAMES]
        iou_diff = max(ious) - min(ious)
        img_data.append(
            {
                "img_id": img_id,
                "ious": {
                    method: method_ious[method][img_id] for method in METHOD_NAMES
                },
                "iou_diff": iou_diff,
            }
        )

    # Sort by IoU difference (descending)
    img_data.sort(key=lambda x: x["iou_diff"], reverse=True)

    # Generate HTML
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SpaceNet Prediction Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            text-align: center;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        th {
            background-color: #4CAF50;
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        img {
            max-width: 220px;
            height: auto;
            display: block;
            margin: 0 auto;
            image-rendering: pixelated;
        }
        .iou-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .iou-badge.good {
            background-color: #4CAF50;
            color: white;
        }
        .iou-badge.mid {
            background-color: #ff9800;
            color: white;
        }
        .iou-badge.bad {
            background-color: #f44336;
            color: white;
        }
    </style>
</head>
<body>
    <h1>SpaceNet 100% Results - Prediction Comparison</h1>
    <p style="text-align: center;">Sorted by IoU difference (max - min) across methods</p>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Image ID</th>
                <th>Δ IoU</th>
                <th>RGB</th>
                <th>Ground Truth</th>
"""

    # Add method columns
    for method in METHOD_NAMES:
        short_name = method.replace("_", " ").title()
        html_content += f"                <th>{short_name}<br/>Prediction</th>\n"

    html_content += """            </tr>
        </thead>
        <tbody>
"""

    # Add rows
    for rank, data in enumerate(img_data, 1):
        img_id = data["img_id"]
        iou_diff = data["iou_diff"]

        # RGB and GT from reference folder
        rgb_path = f"individual_images/{reference_folder.name}/rgb/{img_id}.png"
        gt_path = f"individual_images/{reference_folder.name}/gt_mask/{img_id}.png"

        html_content += f"""            <tr>
                <td>{rank}</td>
                <td>{img_id}</td>
                <td>{iou_diff:.4f}</td>
                <td><img src="{rgb_path}" loading="lazy" alt="RGB {img_id}"/></td>
                <td><img src="{gt_path}" loading="lazy" alt="GT {img_id}"/></td>
"""

        # Add prediction columns with IoU badges
        for method in METHOD_NAMES:
            if method in method_folder_map:
                folder = method_folder_map[method]
                pred_path = f"individual_images/{folder.name}/pred_mask/{img_id}.png"
                iou = data["ious"][method]
                color_class = get_iou_color(iou)
                html_content += f"""                <td>
                    <img src="{pred_path}" loading="lazy" alt="{method} {img_id}"/>
                    <div class="iou-badge {color_class}">{iou:.4f}</div>
                </td>
"""
            else:
                html_content += "                <td>N/A</td>\n"

        html_content += "            </tr>\n"

    html_content += """        </tbody>
    </table>
</body>
</html>
"""

    # Write HTML file
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)

    print(f"[INFO] Wrote HTML comparison to {OUTPUT_HTML}")


if __name__ == "__main__":
    generate_html()
