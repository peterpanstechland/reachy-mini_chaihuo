"""Offline opening-show assets and WAV loading helpers."""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENING_AUDIO_PATH = PROJECT_ROOT / "assets/opening/chaihuo_opening_zh.wav"
DEFAULT_OPENING_TEXT_PATH = PROJECT_ROOT / "assets/opening/chaihuo_opening_zh.txt"
DEFAULT_TRAINING_AUDIO_PATH = PROJECT_ROOT / "assets/training/wio_terminal.wav"
DEFAULT_TRAINING_TEXT_PATH = PROJECT_ROOT / "assets/training/wio_terminal.txt"
DEFAULT_TRAINING_FAQ_PATH = PROJECT_ROOT / "assets/training/wio_terminal_faq.txt"

_FAQ_PAIR = re.compile(
    r"^Q[:：]\s*(.+?)\nA[:：]\s*(.+?)(?=\nQ[:：]|\Z)",
    re.MULTILINE | re.DOTALL,
)

_DIALECT_HEADING = re.compile(
    r"\n\s*(?:【[^】]*话[^】]*】|\[[^\]\n]*话[^\]\n]*\])\s*\n",
)
_DIALECT_LABEL = re.compile(r"[【\[]\s*([^】\]/\s]*话)")

# Official Qwen3-TTS-Flash dialect voices. Hangzhou is Wu; there is no
# Hangzhou system voice, so it shares Shanghai Jada.
_DIALECT_VOICES = (
    ("天津", "Peter"),
    ("上海", "Jada"),
    ("杭州", "Jada"),
    ("北京", "Dylan"),
    ("南京", "Li"),
    ("陕西", "Marcus"),
    ("四川", "Eric"),
    ("粤", "Rocky"),
    ("广东", "Rocky"),
)


@dataclass(frozen=True)
class OpeningScript:
    body: str
    dialect_text: str
    dialect_label: str


def _dialect_label(heading: str) -> str:
    match = _DIALECT_LABEL.search(heading)
    return match.group(1) if match else heading.strip("【】[] \n")


def parse_opening_script(text: str) -> OpeningScript:
    """Split Mandarin body, dialect tail, and the heading label."""
    match = _DIALECT_HEADING.search(text)
    if not match:
        return OpeningScript(text.strip(), "", "")
    return OpeningScript(
        body=text[: match.start()].strip(),
        dialect_text=text[match.end() :].strip(),
        dialect_label=_dialect_label(match.group(0)),
    )


def split_opening_text(text: str) -> tuple[str, str]:
    """Split the Mandarin body from an optional dialect tail.

    A heading such as ``[天津话]`` or ``【杭州话 / 杭州味口语版】`` starts
    the spoken dialect segment and is not itself synthesized.
    """
    script = parse_opening_script(text)
    return script.body, script.dialect_text


def resolve_dialect_voice(label: str) -> str:
    """Map a script heading like ``杭州话`` to a Qwen dialect voice."""
    if not label:
        raise ValueError("方言标题为空")
    for keyword, voice in _DIALECT_VOICES:
        if keyword in label:
            return voice
    raise ValueError(f"没有对应的方言音色: {label}")


@dataclass(frozen=True)
class OpeningAudio:
    pcm: bytes
    sample_rate: int
    duration_s: float


@dataclass(frozen=True)
class LocalShowSpec:
    kind: str
    event: str
    status: str
    audio_path: Path
    label: str
    busy_message: str
    missing_message: str


LOCAL_SHOWS: dict[str, LocalShowSpec] = {
    "opening": LocalShowSpec(
        kind="opening",
        event="opening_show",
        status="opening_status",
        audio_path=DEFAULT_OPENING_AUDIO_PATH,
        label="开场白",
        busy_message="开场白正在播放",
        missing_message="开场音频不可用",
    ),
    "training": LocalShowSpec(
        kind="training",
        event="training_show",
        status="training_status",
        audio_path=DEFAULT_TRAINING_AUDIO_PATH,
        label="Wio Terminal 培训",
        busy_message="培训讲解正在播放",
        missing_message="培训音频不可用",
    ),
}


def load_opening_audio(path: str | Path = DEFAULT_OPENING_AUDIO_PATH) -> OpeningAudio:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"本地讲解音频不存在: {audio_path}")

    with wave.open(str(audio_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        if channels != 1 or sample_width != 2:
            raise ValueError("本地讲解音频必须是 16-bit mono WAV")
        pcm = wav.readframes(frame_count)

    return OpeningAudio(
        pcm=pcm,
        sample_rate=sample_rate,
        duration_s=frame_count / sample_rate,
    )


def parse_faq_items(text: str) -> list[dict[str, str]]:
    """Parse ``Q:`` / ``A:`` blocks used by the Wio Terminal FAQ file."""
    items: list[dict[str, str]] = []
    for match in _FAQ_PAIR.finditer(text.strip()):
        question = " ".join(match.group(1).split())
        answer = " ".join(match.group(2).split())
        if question and answer:
            items.append({"question": question, "answer": answer})
    return items


def load_training_faq_text(path: str | Path = DEFAULT_TRAINING_FAQ_PATH) -> str:
    faq_path = Path(path)
    if not faq_path.is_file():
        return ""
    return faq_path.read_text(encoding="utf-8").strip()


def load_training_faq_items(path: str | Path = DEFAULT_TRAINING_FAQ_PATH) -> list[dict[str, str]]:
    return parse_faq_items(load_training_faq_text(path))
