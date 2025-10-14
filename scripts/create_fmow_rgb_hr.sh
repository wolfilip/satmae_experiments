#!/usr/bin/env bash
set -euo pipefail

PAIRS_FILE="/home/filip/paired_fmow_baseline.txt"
DEST_ROOT="/home/filip/datasets/fmow_rgb_hr"

if [ ! -f "$PAIRS_FILE" ]; then
  echo "Pairs file not found: $PAIRS_FILE"
  exit 1
fi

mkdir -p "$DEST_ROOT"

idx=0
while IFS=$'\t' read -r tif jpg; do
  idx=$((idx+1))
  # try to infer class and instance from tif path: assume .../<class>/<instance>/file.tif
  class_dir=$(echo "$tif" | awk -F'/' '{n=NF; print $(n-2)}')
  inst_dir=$(echo "$tif" | awk -F'/' '{n=NF; print $(n-1)}')

  dest_dir="$DEST_ROOT/$class_dir/$inst_dir"
  mkdir -p "$dest_dir"

  # copy with clear filenames
  src1_dest="$dest_dir/pair_$(printf "%06d" "$idx")_fmow.tif"
  src2_dest="$dest_dir/pair_$(printf "%06d" "$idx")_baseline.jpg"

  cp "$tif" "$src1_dest"
  cp "$jpg" "$src2_dest"

  echo "Copied pair $idx -> $src1_dest , $src2_dest"

done < "$PAIRS_FILE"

echo "All pairs copied to $DEST_ROOT"