#!/usr/bin/env bash
set -euo pipefail

ROOT1="/home/filip/datasets/fMoW-Sentinel/fmow-sentinel/train"
ROOT2="/storage/datasets/fMoW/train"
OUT="/home/filip/paired_fmow_baseline_exact.txt"
STATS="/home/filip/paired_fmow_baseline_exact_stats.txt"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"
: > "$STATS"

python3 - <<'PY'
import os
from pathlib import Path

root1 = Path("/home/filip/datasets/fMoW-Sentinel/fmow-sentinel/train")
root2 = Path("/storage/datasets/fMoW/train")
out_path = Path("/home/filip/paired_fmow_baseline_exact.txt")

# helpers
def list_dir_sorted(p: Path):
    return sorted([x for x in p.iterdir() if x.is_dir()])

def list_images_by_basename(p: Path):
    imgs = {}
    for ext in ("*.tif",):
        for f in p.glob(ext):
            imgs[f.stem] = str(f.resolve())
    return imgs

def list_images_by_basename_jpg(p: Path):
    """Return dict mapping normalized stem -> path for jpg/png files that end with '_rgb'.

    Normalization removes the trailing '_rgb' suffix (case-insensitive). Files like
    'tile_001_msrgb.jpg' will be ignored.
    """
    imgs = {}
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
        for f in p.glob(ext):
            stem = f.stem
            if stem.lower().endswith('_rgb'):
                norm = stem[:-4]
                if norm not in imgs:
                    imgs[norm] = str(f.resolve())
    return imgs

written = 0
with out_path.open("w") as fout, open(out_path.with_name(out_path.name).parent.joinpath('paired_fmow_baseline_exact_stats.txt'), 'w') as stats_f:
    stats_f.write("class\tinstance\tmatched_pairs\n")
    for class_dir in list_dir_sorted(root1):
        class_name = class_dir.name
        other_class_dir = root2 / class_name
        if not other_class_dir.exists():
            matches = [d for d in list_dir_sorted(root2) if d.name.lower() == class_name.lower()]
            if matches:
                other_class_dir = matches[0]
            else:
                continue

        print(f"Processing class: {class_name}")
        for inst_dir in list_dir_sorted(class_dir):
            inst_name = inst_dir.name
            other_inst_dir = other_class_dir / inst_name
            if not other_inst_dir.exists():
                matches = [d for d in list_dir_sorted(other_class_dir) if d.name.startswith(inst_name) or inst_name.startswith(d.name)]
                if matches:
                    other_inst_dir = matches[0]
                else:
                    continue

            print(f"  Instance: {inst_name}")
            imgs1 = list_images_by_basename(inst_dir)
            imgs2 = list_images_by_basename_jpg(other_inst_dir)

            # print available basenames for debugging
            print(f"    Available tif basenames: {sorted(imgs1.keys())}")
            print(f"    Available jpg basenames: {sorted(imgs2.keys())}")

            # show each attempted matching candidate
            all_keys = sorted(set(imgs1.keys()) | set(imgs2.keys()))
            for stem in all_keys:
                t = imgs1.get(stem)
                j = imgs2.get(stem)
                print(f"      Trying: {stem} -> tif: {t if t else 'MISSING'} | jpg: {j if j else 'MISSING'}")

            common = set(imgs1.keys()) & set(imgs2.keys())
            matched = 0
            for stem in sorted(common):
                print(f"      Match found: {stem} -> {imgs1[stem]} | {imgs2[stem]}")
                fout.write(f"{imgs1[stem]}\t{imgs2[stem]}\n")
                written += 1
                matched += 1

            stats_f.write(f"{class_name}\t{inst_name}\t{matched}\n")
            stats_f.flush()
            print(f"    Matched {matched} exact pairs")

print(f"Wrote {written} exact pairs to {out_path}")
PY

echo "Done. Exact pairs written to: $OUT"
chmod +x "$OUT" || true
