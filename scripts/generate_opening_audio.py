#!/usr/bin/env python3
"""Generate the event opening WAV once; runtime playback is fully offline."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from chaihuo_reachy.config import load_config
from chaihuo_reachy.opening import (
    DEFAULT_OPENING_AUDIO_PATH,
    DEFAULT_OPENING_TEXT_PATH,
    synthesize_show_audio,
)


async def generate(text_path: Path, output_path: Path) -> None:
    text = text_path.read_text(encoding="utf-8").strip()
    audio = await synthesize_show_audio(text, output_path, load_config())
    print(
        f"saved={output_path} sample_rate={audio.sample_rate} "
        f"duration={audio.duration_s:.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, default=DEFAULT_OPENING_TEXT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OPENING_AUDIO_PATH)
    args = parser.parse_args()
    asyncio.run(generate(args.text, args.output))


if __name__ == "__main__":
    main()
