from __future__ import annotations

import wave

import pytest

from chaihuo_reachy.opening import (
    LOCAL_SHOWS,
    load_opening_audio,
    load_training_faq_items,
    parse_faq_items,
    parse_opening_script,
    resolve_dialect_voice,
    split_opening_text,
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
    items = load_training_faq_items()
    assert len(items) >= 4
    assert all(item["question"] and item["answer"] for item in items)
    assert any("Wio Terminal" in item["question"] for item in items)
