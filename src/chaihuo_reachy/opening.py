"""Offline opening-show assets, dialect catalog, and WAV helpers."""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENING_AUDIO_PATH = PROJECT_ROOT / "assets/opening/chaihuo_opening_zh.wav"
DEFAULT_OPENING_TEXT_PATH = PROJECT_ROOT / "assets/opening/chaihuo_opening_zh.txt"
DEFAULT_TRAINING_AUDIO_PATH = PROJECT_ROOT / "assets/training/wio_terminal.wav"
DEFAULT_TRAINING_TEXT_PATH = PROJECT_ROOT / "assets/training/wio_terminal.txt"
DEFAULT_TRAINING_FAQ_PATH = PROJECT_ROOT / "assets/training/wio_terminal_faq.txt"
# File synthesis must use HTTP flash, not the conversation realtime WS.
# Realtime is shared with live replies and hard-times-out at 60s.
SHOW_TTS_MODEL = "qwen3-tts-flash"
DIALECT_TTS_MODEL = SHOW_TTS_MODEL
SHOW_TTS_CHUNK_CHARS = 500

_FAQ_PAIR = re.compile(
    r"^Q[:：]\s*(.+?)\nA[:：]\s*(.+?)(?=\nQ[:：]|\Z)",
    re.MULTILINE | re.DOTALL,
)

_DIALECT_HEADING = re.compile(
    r"\n\s*(?:【[^】]*话[^】]*】|\[[^\]\n]*话[^\]\n]*\])\s*\n",
)
_DIALECT_LABEL = re.compile(r"[【\[]\s*([^】\]/\s]*话)")


@dataclass(frozen=True)
class DialectVoice:
    id: str
    label: str
    voice: str
    keywords: tuple[str, ...]
    spoken_tail: bool = True


# Official Qwen3-TTS-Flash dialect voices. Hangzhou is Wu; there is no
# Hangzhou system voice, so it shares Shanghai Jada. Mandarin uses the
# conversation default Cherry and does not write a dialect tail.
SHOW_DIALECTS: tuple[DialectVoice, ...] = (
    DialectVoice("mandarin", "普通话", "Cherry", ("普通话", "国语"), spoken_tail=False),
    DialectVoice("tianjin", "天津话", "Peter", ("天津",)),
    DialectVoice("shanghai", "上海话", "Jada", ("上海", "沪")),
    DialectVoice("hangzhou", "杭州话", "Jada", ("杭州",)),
    DialectVoice("beijing", "北京话", "Dylan", ("北京",)),
    DialectVoice("nanjing", "南京话", "Li", ("南京",)),
    DialectVoice("shaanxi", "陕西话", "Marcus", ("陕西", "西安")),
    DialectVoice("sichuan", "四川话", "Eric", ("四川", "成都")),
    DialectVoice("cantonese", "粤语", "Rocky", ("粤", "广东", "广州")),
)
DEFAULT_SHOW_DIALECT = SHOW_DIALECTS[0]
_DIALECT_BY_ID = {item.id: item for item in SHOW_DIALECTS}


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


def list_show_dialects() -> list[dict[str, Any]]:
    """UI/API catalog for the show-content dialect dropdown."""
    return [
        {
            "id": item.id,
            "label": item.label,
            "voice": item.voice,
            "keywords": list(item.keywords),
            "spoken_tail": item.spoken_tail,
        }
        for item in SHOW_DIALECTS
    ]


def get_show_dialect(dialect_id: str) -> DialectVoice:
    dialect = _DIALECT_BY_ID.get(str(dialect_id or "").strip())
    if dialect is None:
        raise ValueError(f"没有对应的方言: {dialect_id}")
    return dialect


def _match_dialect(text: str, *, include_mandarin: bool) -> DialectVoice | None:
    hits: list[tuple[int, DialectVoice]] = []
    for dialect in SHOW_DIALECTS:
        if not include_mandarin and not dialect.spoken_tail:
            continue
        matched = next((keyword for keyword in dialect.keywords if keyword in text), "")
        if matched:
            hits.append((len(matched), dialect))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0], reverse=True)
    return hits[0][1]


def match_show_dialect(brief: str, location_text: str = "") -> DialectVoice:
    """Pick a dialect from the brief first, then live/session location."""
    for source in (brief or "", location_text or ""):
        hit = _match_dialect(source, include_mandarin=False)
        if hit is not None:
            return hit
    if _match_dialect(brief or "", include_mandarin=True) is DEFAULT_SHOW_DIALECT:
        return DEFAULT_SHOW_DIALECT
    return DEFAULT_SHOW_DIALECT


def resolve_show_dialect(dialect_id: str, brief: str, location_text: str = "") -> DialectVoice:
    if dialect_id:
        try:
            return get_show_dialect(dialect_id)
        except ValueError:
            pass
    return match_show_dialect(brief, location_text)


def resolve_dialect_voice(label: str) -> str:
    """Map a script heading like ``杭州话`` to a Qwen dialect voice."""
    if not label:
        raise ValueError("方言标题为空")
    dialect = _match_dialect(label, include_mandarin=True)
    if dialect is None:
        raise ValueError(f"没有对应的方言音色: {label}")
    if dialect is DEFAULT_SHOW_DIALECT and not any(
        keyword in label for keyword in dialect.keywords
    ):
        raise ValueError(f"没有对应的方言音色: {label}")
    return dialect.voice


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
    text_path: Path
    label: str
    busy_message: str
    missing_message: str


LOCAL_SHOWS: dict[str, LocalShowSpec] = {
    "opening": LocalShowSpec(
        kind="opening",
        event="opening_show",
        status="opening_status",
        audio_path=DEFAULT_OPENING_AUDIO_PATH,
        text_path=DEFAULT_OPENING_TEXT_PATH,
        label="开场白",
        busy_message="开场白正在播放",
        missing_message="开场音频不可用",
    ),
    "training": LocalShowSpec(
        kind="training",
        event="training_show",
        status="training_status",
        audio_path=DEFAULT_TRAINING_AUDIO_PATH,
        text_path=DEFAULT_TRAINING_TEXT_PATH,
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


_SENTENCE_BREAK = re.compile(r"(?<=[。！？!?；;])")


def split_show_tts_chunks(text: str, max_chars: int = SHOW_TTS_CHUNK_CHARS) -> list[str]:
    """Split a spoken script into HTTP-flash-sized pieces."""
    script = text.strip()
    if not script:
        return []
    if len(script) <= max_chars:
        return [script]
    units: list[str] = []
    for block in (part.strip() for part in re.split(r"\n{2,}", script) if part.strip()):
        if len(block) <= max_chars:
            units.append(block)
            continue
        sentences = [part.strip() for part in _SENTENCE_BREAK.split(block) if part.strip()]
        if not sentences:
            sentences = [block[index : index + max_chars] for index in range(0, len(block), max_chars)]
        for sentence in sentences:
            if len(sentence) <= max_chars:
                units.append(sentence)
                continue
            units.extend(
                sentence[index : index + max_chars]
                for index in range(0, len(sentence), max_chars)
            )
    chunks: list[str] = []
    buf = ""
    for unit in units:
        candidate = unit if not buf else f"{buf}\n{unit}"
        if buf and len(candidate) > max_chars:
            chunks.append(buf)
            buf = unit
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def _resample_show_pcm(samples: Any, source_rate: int, target_rate: int) -> Any:
    import numpy as np

    if source_rate == target_rate or samples.size == 0:
        return samples
    target_size = max(1, round(samples.size * target_rate / source_rate))
    source_positions = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, target_size, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


async def synthesize_show_segment(
    text: str,
    config: Any,
    *,
    tts_client_cls: Callable[..., Any] | None = None,
) -> tuple[Any, int]:
    import numpy as np

    if tts_client_cls is None:
        from chaihuo_reachy.bailian.tts_client import BailianTTSClient

        tts_client_cls = BailianTTSClient

    chunks: list[bytes] = []
    sample_rates: set[int] = set()

    def on_audio(pcm: bytes, sample_rate: int) -> None:
        chunks.append(bytes(pcm))
        sample_rates.add(int(sample_rate))

    client = tts_client_cls(config, on_audio=on_audio)
    await client.open()
    try:
        await client.synthesize(text)
    finally:
        await client.close()

    if not chunks or len(sample_rates) != 1:
        raise RuntimeError("TTS 未返回有效的单采样率 PCM")
    return np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32), sample_rates.pop()


async def synthesize_show_audio(
    text: str,
    output_path: str | Path,
    config: Any,
    *,
    tts_client_cls: Callable[..., Any] | None = None,
) -> OpeningAudio:
    """Synthesize a show script to 16-bit mono WAV, including an optional dialect tail."""
    import numpy as np

    output = Path(output_path)
    script = parse_opening_script(text)
    if not script.body:
        raise ValueError("口播正文为空")

    show_config = replace(
        config,
        tts_volume=100,
        tts_speech_rate=0.96,
        bailian_tts_model=SHOW_TTS_MODEL,
    )
    segments: list[Any] = []
    sample_rate: int | None = None
    for piece in split_show_tts_chunks(script.body):
        piece_samples, piece_rate = await synthesize_show_segment(
            piece, show_config, tts_client_cls=tts_client_cls
        )
        if sample_rate is None:
            sample_rate = piece_rate
        elif piece_rate != sample_rate:
            piece_samples = _resample_show_pcm(piece_samples, piece_rate, sample_rate)
        segments.append(piece_samples)
    if sample_rate is None:
        raise RuntimeError("TTS 未返回有效的单采样率 PCM")
    if script.dialect_text:
        dialect_config = replace(
            show_config,
            bailian_tts_voice=resolve_dialect_voice(script.dialect_label),
        )
        for piece in split_show_tts_chunks(script.dialect_text):
            dialect_samples, dialect_rate = await synthesize_show_segment(
                piece, dialect_config, tts_client_cls=tts_client_cls
            )
            silence = np.zeros(round(sample_rate * 0.35), dtype=np.float32)
            segments.extend(
                [silence, _resample_show_pcm(dialect_samples, dialect_rate, sample_rate)]
            )

    samples = np.concatenate(segments)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0:
        raise RuntimeError("TTS 返回了静音音频")
    samples *= min(4.0, (32767.0 * 0.92) / peak)
    pcm = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return OpeningAudio(
        pcm=pcm,
        sample_rate=sample_rate,
        duration_s=len(pcm) / 2 / sample_rate,
    )
