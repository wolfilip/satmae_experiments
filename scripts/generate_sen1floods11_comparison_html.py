import os, re, html
from pathlib import Path

BASE_DIR = Path(
    "/home/filip/satmae_experiments/outputs/images/sen1floods11_100pc_results"
)
PER_IMAGE_DIR = BASE_DIR / "per_image"
INDIV_IMAGES_DIR = BASE_DIR / "individual_images"
OUTPUT_HTML = BASE_DIR / "comparison_miou_diff.html"

# Methods we want to compare (present in individual_images)
method_names = [
    "copernicusfm_vit_b",
    "dinov3_LS",
    "swin_bm_ms_fmow_hr_280k_100e_dinov3p34c_rgb_head_dconv_aug",
]

TIMESTAMP_PATTERN = re.compile(r"(\d{8}_\d{6})$")

# Map method -> folder with timestamp
method_folder_map = {}
for entry in INDIV_IMAGES_DIR.iterdir():
    if not entry.is_dir():
        continue
    name = entry.name
    # Find prefix before the timestamp pattern
    m = TIMESTAMP_PATTERN.search(name)
    if m:
        ts = m.group(1)
        prefix = name[: -len(ts) - 1]  # remove '_' + timestamp
        if prefix in method_names:
            # Keep latest timestamp if multiple
            prev = method_folder_map.get(prefix)
            if prev is None or prev.name < name:
                method_folder_map[prefix] = entry

missing_methods = [m for m in method_names if m not in method_folder_map]
if missing_methods:
    print(f"[WARN] Missing individual_images folders for methods: {missing_methods}")

# Parse IoU files
ious = {}
for method in method_names:
    iou_file = PER_IMAGE_DIR / method / "image_results_iou_-1.txt"
    if not iou_file.exists():
        print(f"[WARN] IoU file missing for {method}: {iou_file}")
        continue
    method_ious = {}
    with iou_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            try:
                img_part, val_part = line.split(":", 1)
                img_id_str = img_part.strip().replace("img_", "")
                img_id = int(img_id_str)
                val = float(val_part.strip())
                method_ious[img_id] = val
            except Exception as e:
                print(f"[WARN] Failed to parse line for {method}: {line} ({e})")
    ious[method] = method_ious

# Aggregate image ids
all_img_ids = sorted(set().union(*[set(ious.get(m, {}).keys()) for m in method_names]))

# Compute differences
rows = []
for img_id in all_img_ids:
    vals = [(m, ious.get(m, {}).get(img_id, None)) for m in method_names]
    present_vals = [v for _, v in vals if v is not None]
    if len(present_vals) < 2:
        continue  # need at least two to compute difference
    diff = max(present_vals) - min(present_vals)
    rows.append((img_id, diff, vals))

rows.sort(key=lambda r: r[1], reverse=True)  # descending by difference

# Choose a reference folder for RGB & GT (prefer dinov3_LS)
reference_folder = None
if "dinov3_LS" in method_folder_map:
    reference_folder = method_folder_map["dinov3_LS"]
else:
    # Fallback to first available method
    for m in method_names:
        if m in method_folder_map:
            reference_folder = method_folder_map[m]
            break

# Build HTML
css = """
body { font-family: Arial, sans-serif; background:#fafafa; }
.table { border-collapse: collapse; width:100%; }
th, td { border:1px solid #ccc; padding:6px; text-align:center; vertical-align:top; }
th { background:#eee; }
.method-block { display:flex; flex-direction:column; align-items:center; }
img { max-width:220px; height:auto; image-rendering:pixelated; border:1px solid #666; }
.diff-cell { font-weight:bold; }
.iou-good { color:#0a7d00; }
.iou-mid { color:#b57700; }
.iou-bad { color:#c40000; }
.header { margin:20px 0; }
.sticky { position:sticky; top:0; background:#eee; }
"""

html_parts = [
    "<html><head><meta charset='utf-8'><title>sen1floods11 per-image IoU comparison</title>",
    f"<style>{css}</style>",
    "</head><body>",
]
html_parts.append(
    "<div class='header'><h1>sen1floods11 Per-Image IoU Comparison (Sorted by Max Difference)</h1>"
)
html_parts.append("<p>Methods compared: " + ", ".join(method_names) + "</p>")
html_parts.append("<p>Total images: " + str(len(rows)) + "</p></div>")

html_parts.append("<table class='table'>")
method_header_html = "".join(
    f"<th>{html.escape(m)}<br/>IoU & Pred Mask</th>" for m in method_names
)
extra_cols = "<th>RGB</th><th>GT Mask</th>" if reference_folder else ""
html_parts.append(
    f"<tr class='sticky'><th>Rank</th><th>Image ID</th><th>Δ IoU (max - min)</th>{extra_cols}{method_header_html}</tr>"
)

for rank, (img_id, diff, vals) in enumerate(rows, start=1):
    diff_cell = f"<td class='diff-cell'>{diff:.4f}</td>"
    io_cells = []
    # Add RGB & GT mask cells if reference folder available
    if reference_folder:
        rgb_rel = reference_folder.relative_to(BASE_DIR) / "rgb" / f"img_{img_id}.png"
        gt_rel = (
            reference_folder.relative_to(BASE_DIR) / "gt_mask" / f"img_{img_id}.png"
        )
        rgb_cell = f"<td><div class='method-block'><div>RGB</div><img loading='lazy' src='{html.escape(str(rgb_rel))}' alt='rgb img {img_id}' /></div></td>"
        gt_cell = f"<td><div class='method-block'><div>GT</div><img loading='lazy' src='{html.escape(str(gt_rel))}' alt='gt mask img {img_id}' /></div></td>"
        io_cells.extend([rgb_cell, gt_cell])
    for method, score in vals:
        folder = method_folder_map.get(method)
        if folder is None or score is None:
            io_cells.append("<td>Missing</td>")
            continue
        # image path relative to HTML file
        img_rel = folder.relative_to(BASE_DIR) / "pred_mask" / f"img_{img_id}.png"
        # Color class by score
        if score >= 0.95:
            cls = "iou-good"
        elif score >= 0.80:
            cls = "iou-mid"
        else:
            cls = "iou-bad"
        cell_html = (
            f"<div class='method-block'><div class='{cls}'>IoU: {score:.4f}</div>"
            f"<img loading='lazy' src='{html.escape(str(img_rel))}' alt='{method} img {img_id}' />"
            f"</div>"
        )
        io_cells.append(f"<td>{cell_html}</td>")
    html_parts.append(
        f"<tr><td>{rank}</td><td>{img_id}</td>{diff_cell}{''.join(io_cells)}</tr>"
    )

html_parts.append("</table>")
html_parts.append(
    "<p style='margin-top:30px;font-size:12px;color:#666;'>Generated automatically.</p>"
)
html_parts.append("</body></html>")

OUTPUT_HTML.write_text("\n".join(html_parts))
print(f"[INFO] Wrote HTML comparison to {OUTPUT_HTML}")
