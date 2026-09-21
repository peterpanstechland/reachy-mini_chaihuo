from __future__ import annotations

import wave

import pytest

from chaihuo_reachy.config import Config
from chaihuo_reachy.opening import (
    LOCAL_SHOWS,
    SHOW_TTS_MODEL,
    list_show_dialects,
    load_opening_audio,
    load_training_faq_items,
    match_show_dialect,
    parse_faq_items,
    parse_opening_script,
    resolve_dialect_voice,
    split_opening_text,
    split_show_tts_chunks,
    synthesize_show_audio,
)


def test_load_opening_audio_reads_mono_pcm(tmp_path) -> None:
    path = tmp_path / "opening.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x00" * 2400)

    audio = load_opening_audio(path)

    assert audio.sample_rate == 24000
    assert audio.duration_s == pytest.approx(0.1)
    assert audio.pcm == b"\x01\x00" * 2400


def test_load_opening_audio_rejects_stereo(tmp_path) -> None:
    path = tmp_path / "opening.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 20)

    with pytest.raises(ValueError, match="16-bit mono"):
        load_opening_audio(path)


def test_split_opening_text_keeps_hangzhou_and_tianjin_tails() -> None:
    hangzhou = "普通话正文。\n\n【杭州话 / 杭州味口语版】\n\n今朝一道搞点好玩的！\n"
    tianjin = "普通话正文。\n\n[天津话]\n原山小学的同学们，走着！\n"
    body, tail = split_opening_text(hangzhou)
    assert body == "普通话正文。"
    assert tail == "今朝一道搞点好玩的！"
    body, tail = split_opening_text(tianjin)
    assert body == "普通话正文。"
    assert tail == "原山小学的同学们，走着！"
    assert split_opening_text("只有普通话") == ("只有普通话", "")


def test_dialect_heading_selects_qwen_voice() -> None:
    hangzhou = parse_opening_script(
        "普通话正文。\n\n【杭州话 / 杭州味口语版】\n\n今朝一道搞点好玩的！\n"
    )
    assert hangzhou.dialect_label == "杭州话"
    assert resolve_dialect_voice(hangzhou.dialect_label) == "Jada"
    tianjin = parse_opening_script("普通话正文。\n\n[天津话]\n走着！\n")
    assert tianjin.dialect_label == "天津话"
    assert resolve_dialect_voice(tianjin.dialect_label) == "Peter"


def test_parse_faq_items_reads_qa_blocks() -> None:
    items = parse_faq_items(
        "前言可以忽略。\n\nQ: 端口找不到？\nA: 换一根能传数据的 USB-C 线。\n\n"
        "Q：Grove 怎么接？\nA：插到侧面 Grove 口，不用焊接。\n"
    )
    assert items == [
        {"question": "端口找不到？", "answer": "换一根能传数据的 USB-C 线。"},
        {"question": "Grove 怎么接？", "answer": "插到侧面 Grove 口，不用焊接。"},
    ]


def test_training_assets_are_registered_as_local_show() -> None:
    spec = LOCAL_SHOWS["training"]
    assert spec.event == "training_show"
    assert spec.status == "training_status"
    assert spec.text_path.name == "wio_terminal.txt"
    items = load_training_faq_items()
    assert len(items) >= 4
    assert all(item["question"] and item["answer"] for item in items)
    assert any("Wio Terminal" in item["question"] for item in items)


def test_show_dialect_catalog_lists_mandarin_and_hangzhou() -> None:
    catalog = list_show_dialects()
    labels = {item["label"]: item for item in catalog}
    assert labels["普通话"]["voice"] == "Cherry"
    assert labels["普通话"]["spoken_tail"] is False
    assert labels["杭州话"]["voice"] == "Jada"
    assert match_show_dialect("杭州杭创营").label == "杭州话"
    assert match_show_dialect("", "浙江省杭州市西湖区").id == "hangzhou"
    assert match_show_dialect("普通介绍，没有城市").id == "mandarin"
    assert resolve_dialect_voice("普通话") == "Cherry"


class _FakeShowTTS:
    def __init__(self, cfg, on_audio=None) -> None:
        self.cfg = cfg
        self._on_audio = on_audio

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def synthesize(self, text: str) -> None:
        assert self._on_audio is not None
        self._on_audio(b"\x10\x00" * max(16, len(text)), 24000)


@pytest.mark.asyncio
async def test_synthesize_show_audio_joins_dialect_tail(tmp_path) -> None:
    output = tmp_path / "show.wav"
    audio = await synthesize_show_audio(
        "普通话正文。\n\n【杭州话】\n\n今朝一道开搞！\n",
        output,
        Config(),
        tts_client_cls=_FakeShowTTS,
    )
    assert output.is_file()
    assert audio.sample_rate == 24000
    assert audio.duration_s > 0
    loaded = load_opening_audio(output)
    assert loaded.pcm == audio.pcm


def test_show_tts_chunks_stay_under_http_limit() -> None:
    body = "。".join(f"这是第{index}句很长的口播内容" for index in range(40)) + "。"
    chunks = split_show_tts_chunks(body, max_chars=80)
    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert "第0句" in chunks[0]
    assert "第39句" in chunks[-1]


@pytest.mark.asyncio
async def test_show_synth_uses_http_flash_not_realtime(tmp_path) -> None:
    seen: list[str] = []

    class _RecordTTS(_FakeShowTTS):
        def __init__(self, cfg, on_audio=None) -> None:
            seen.append(cfg.bailian_tts_model)
            super().__init__(cfg, on_audio)

    await synthesize_show_audio(
        "短稿。",
        tmp_path / "flash.wav",
        Config(bailian_tts_model="qwen3-tts-flash-realtime"),
        tts_client_cls=_RecordTTS,
    )
    assert seen
    assert all(model == SHOW_TTS_MODEL for model in seen)
    assert all("realtime" not in model for model in seen)
