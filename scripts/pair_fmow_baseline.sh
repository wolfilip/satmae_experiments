#!/usr/bin/env bash
set -euo pipefail

ROOT1="/home/filip/datasets/fMoW-Sentinel/fmow-sentinel/train"
ROOT2="/storage/datasets/fMoW/train"
OUT="/home/filip/paired_fmow_baseline.txt"
STATS="/home/filip/paired_fmow_baseline_stats.txt"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"
: > "$STATS"

python3 - <<'PY'
import os
from pathlib import Path

root1 = Path("/home/filip/datasets/fMoW-Sentinel/fmow-sentinel/train")
root2 = Path("/storage/datasets/fMoW/train")
out_path = Path("/home/filip/paired_fmow_baseline.txt")

# safe helpers
def list_dir_sorted(p: Path):
    return sorted([x for x in p.iterdir() if x.is_dir()])

def list_images_tif(p: Path):
    return sorted([str(x.resolve()) for x in p.glob("*.tif")])

def list_images_jpg(p: Path):
    imgs = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
        imgs.extend(list(p.glob(ext)))
    return sorted([str(x.resolve()) for x in imgs])

written = 0
with out_path.open("w") as fout, open(out_path.with_name(out_path.name).parent.joinpath('paired_fmow_baseline_stats.txt'), 'w') as stats_f:
    stats_f.write("class\tinstance\ttifs\tjpgs\tpairs_written\n")
    for class_dir in list_dir_sorted(root1):
        class_name = class_dir.name
        other_class_dir = root2 / class_name
        if not other_class_dir.exists():
            # try case-insensitive match
            matches = [d for d in list_dir_sorted(root2) if d.name.lower() == class_name.lower()]
            if matches:
                other_class_dir = matches[0]
            else:
                continue

        # echo which class folder we're working on
        print(f"Processing class: {class_name} (folder: {class_dir})")

        for inst_dir in list_dir_sorted(class_dir):
            inst_name = inst_dir.name
            other_inst_dir = other_class_dir / inst_name
            if not other_inst_dir.exists():
                matches = [d for d in list_dir_sorted(other_class_dir) if d.name.startswith(inst_name) or inst_name.startswith(d.name)]
                if matches:
                    other_inst_dir = matches[0]
                else:
                    continue

            # echo instance folder being processed
            print(f"  Instance: {inst_name} -> {inst_dir}  with partner {other_inst_dir}")

            imgs1 = list_images_tif(inst_dir)
            imgs2 = list_images_jpg(other_inst_dir)

            if not imgs1 or not imgs2:
                print(f"    Skipping (no images): tifs={len(imgs1)} jpgs={len(imgs2)}")
                continue

            # write how many images we found for this instance
            pairs_written_local = 0
            n = max(len(imgs1), len(imgs2))
            for i in range(n):
                img1 = imgs1[i % len(imgs1)]
                img2 = imgs2[i % len(imgs2)]
                fout.write(f"{img1}\t{img2}\n")
                written += 1
                pairs_written_local += 1

            stats_f.write(f"{class_name}\t{inst_name}\t{len(imgs1)}\t{len(imgs2)}\t{pairs_written_local}\n")
            # flush so stats appear even if script runs long
            stats_f.flush()
            print(f"    Matched tifs={len(imgs1)} jpgs={len(imgs2)} -> wrote {pairs_written_local} pairs")

print(f"Wrote {written} pairs to {out_path}")
PY

echo "Done. Pairs written to: $OUT"
chmod +x "$OUT" || true
