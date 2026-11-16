"""End-to-end microscopy pipeline demo: detection -> segmentation -> classification.

One command that runs all three agent skills composed, writes every artifact,
and prints a per-class summary. This is the meeting "wow" command.

    python run_pipeline.py --image ../sample.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Make the classification skill (which itself pulls in segmentation + detection)
# importable regardless of cwd.
_CLS_DIR = Path(__file__).resolve().parent / "classification"
if str(_CLS_DIR) not in sys.path:
    sys.path.insert(0, str(_CLS_DIR))

import classify_skill  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end microscopy pipeline (3 skills)")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model-size", default="base")
    p.add_argument("--outdir", type=Path, default=Path("pipeline_out"))
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] detection  -> [2/3] segmentation -> [3/3] classification on {args.image}")
    contract = classify_skill.classify(
        args.image, threshold=args.threshold, model_size=args.model_size
    )

    json_path = args.outdir / "pipeline_result.json"
    csv_path = args.outdir / "cells.csv"
    overlay_path = args.outdir / "pipeline_overlay.png"

    json_path.write_text(json.dumps(contract, indent=2))
    classify_skill.write_csv(contract, csv_path)
    classify_skill.draw_overlay(args.image, contract, overlay_path)

    n = contract["output"]["num_instances"]
    counts = Counter(inst["label"] for inst in contract["instances"])
    print("\n=== pipeline summary ===")
    print(f"cells analyzed : {n}")
    for label, c in counts.most_common():
        print(f"  {label:<12} {c}")
    print(f"\nartifacts in {args.outdir}/:")
    print(f"  {json_path.name}   (full contract)")
    print(f"  {csv_path.name}            (dataframe)")
    print(f"  {overlay_path.name}  (labeled image)")


if __name__ == "__main__":
    main()
