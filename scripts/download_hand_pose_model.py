#!/usr/bin/env python3
"""Fetch the desktop ONNX hand-pose weights used by auto/onnx backends.

Usage:
    uv run python scripts/download_hand_pose_model.py
    uv run python scripts/download_hand_pose_model.py --output models/hand_pose/hand_pose_resnet18.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("download_hand_pose_model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/hand_pose/hand_pose_resnet18.onnx"),
    )
    args = parser.parse_args()

    from chaihuo_reachy.hand_pose import ensure_hand_pose_onnx_model

    path = ensure_hand_pose_onnx_model(args.output)
    logger.info("✅ 手势 ONNX 模型就绪: %s (%d bytes)", path, path.stat().st_size)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)
