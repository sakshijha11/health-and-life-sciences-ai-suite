"""Prepare an RGB tile from an S-BIAD800 OME-Zarr (multi-channel IMC) image.

Why this exists: S-BIAD800 (the public reference set) is Imaging Mass Cytometry
in OME-Zarr — 5D (t, c, z, y, x), 27 uint8 channels, NOT RGB. RF-DETR (and the
interim skill) consume 8-bit RGB. This step makes the channel->RGB decision
explicit so it is part of the pipeline's documented input contract:

    multichannel (c=27) --pick 3 channels--> per-channel min-max stretch --> RGB tile

When Geti supplies OME-Zarr-native models, this maps to their reader skills
(ITEP-96803/96804). For the interim test we collapse to RGB here.

Run:
    python prep_sbiad800_tile.py --zarr "sbiad800/Tonsil 1.ome.zarr/0" \
        --channels 0 1 2 --level 1 --out tonsil1_rgb.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S-BIAD800 OME-Zarr -> RGB tile")
    p.add_argument("--zarr", required=True, help="Path to the OME-Zarr image group")
    p.add_argument(
        "--channels",
        type=int,
        nargs=3,
        default=[0, 1, 2],
        help="Three channel indices mapped to R,G,B",
    )
    p.add_argument("--level", type=int, default=1, help="Multiscale pyramid level")
    p.add_argument("--out", type=Path, default=Path("sbiad800_rgb.png"))
    return p.parse_args()


def stretch(plane: np.ndarray) -> np.ndarray:
    """Per-channel min-max contrast stretch to 0-255 uint8."""
    lo, hi = np.percentile(plane, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1
    out = (np.clip(plane, lo, hi) - lo) / (hi - lo)
    return (out * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    g = zarr.open(args.zarr, mode="r")
    arr = g[str(args.level)]  # (t, c, z, y, x)
    r, gch, b = args.channels

    # take t=0, z=0; select the 3 channels -> (y, x) each
    planes = [stretch(np.asarray(arr[0, c, 0])) for c in (r, gch, b)]
    rgb = np.stack(planes, axis=-1)  # (y, x, 3)

    Image.fromarray(rgb, mode="RGB").save(args.out)
    print(f"Wrote {args.out}  shape={rgb.shape}  from channels R={r} G={gch} B={b}")
    print(
        "Input contract note: source is 27-channel uint8 IMC (OME-Zarr); "
        "this tile is a 3-channel RGB projection for the interim RF-DETR test."
    )


if __name__ == "__main__":
    main()
