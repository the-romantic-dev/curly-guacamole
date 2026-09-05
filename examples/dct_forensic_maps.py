"""Compute DCT forensic map shapes for one image.

Usage:
    python examples/dct_forensic_maps.py path/to/image.jpg --image-size 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.forensic.dct import forensic_maps, luma_qtable, resize_fmaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bgr = cv2.imread(str(args.image_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {args.image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qtable = luma_qtable(args.image_path)
    maps = forensic_maps(rgb, qtable)
    resized = resize_fmaps(maps, args.image_size)

    print(f"maps: {maps.shape}")
    print(f"resized: {resized.shape}")


if __name__ == "__main__":
    main()

