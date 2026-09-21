from __future__ import annotations

import wave
from pathlib import Path


import pytest

from chaihuo_reachy.config import Config
from chaihuo_reachy.opening import LocalShowSpec
from chaihuo_reachy.show_content import (
    SHOW_DURATION_PRESETS,
    ShowLibrary,
    handle_show_content_action,
    location_text_from_engine,
)


def _write_wav(path: Path, frames: int = 2400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x00" * frames)


def _show_specs(tmp_path: Path) -> dict[str, LocalShowSpec]:
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    opening_txt = asset_dir / "opening.txt"
    opening_wav = asset_dir / "opening.wav"
    training_txt = asset_dir / "training.txt"
    training_wav = asset_dir / "training.wav"
    opening_txt.write_text(
        "欢迎来到杭州杭创营。\n\n【杭州话】\n\n今朝一道开搞！\n", encoding="utf-8"
    )
    training_txt.write_text("欢迎来参加 Wio Terminal 培训。\n", encoding="utf-8")
    _write_wav(opening_wav)
    _write_wav(training_wav, frames=4800)
    return {
        "opening": LocalShowSpec(
            kind="opening",
            event="opening_show",
            status="opening_status",
            audio_path=opening_wav,
            text_path=opening_txt,
            label="开场白",
            busy_message="busy",
            missing_message="missing",
        ),
        "training": LocalShowSpec(
            kind="training",
            event="training_show",
            status="training_status",
            audio_path=training_wav,
            text_path=training_txt,
            label="培训",
            busy_message="busy",
            missing_message="missing",
        ),
    }


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[list[dict]] = []

    async def chat_stream(self, messages: list[dict]) -> object:
        self.messages.append(messages)
        yield self.text


class _FakeTTS:
    def __init__(self, cfg, on_audio=None) -> None:
        self.cfg = cfg
        self._on_audio = on_audio

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def synthesize(self, text: str) -> None:
        assert self._on_audio is not None
        self._on_audio(b"\x22\x00" * max(24, len(text)), 24000)


def test_location_text_joins_live_and_session() -> None:
    class _Pos:
        province = "浙江省"
        city = "杭州市"
        district = "西湖区"
        address = "杭创营"

    class _Loc:
        latest_position = _Pos()

    class _Engine:
        _location = _Loc()
        _session_location = {"place": "OPC 社区"}

    assert "杭州" in location_text_from_engine(_Engine())
    assert "OPC" in location_text_from_engine(_Engine())


def test_seed_imports_current_assets_as_first_history(tmp_path) -> None:
    specs = _show_specs(tmp_path)
    library = ShowLibrary(tmp_path / "shows", specs)
    payload = library.list_payload("opening")
    assert payload["versions"]
    assert payload["active_id"] == payload["versions"][0]["id"]
    assert payload["versions"][0]["dialect"] == "hangzhou"
    assert payload["durations"] == list(SHOW_DURATION_PRESETS)
    seeded = library.get_payload("opening", payload["active_id"])
    assert "杭州杭创营" in seeded["script"]
    assert seeded["version"]["has_audio"] is True


@pytest.mark.asyncio
async def test_generate_draft_does_not_change_playable_wav(tmp_path) -> None:
    specs = _show_specs(tmp_path)
    original = specs["opening"].audio_path.read_bytes()
    original_text = specs["opening"].text_path.read_text(encoding="utf-8")
    library = ShowLibrary(tmp_path / "shows", specs)
    seeded = library.list_payload("opening")
    llm = _FakeLLM("大家好，这是新草稿。")
    payload = await library.generate_payload(
        Config(),
        kind="opening",
        brief="杭州杭创营",
        duration_s=30,
        llm_client=llm,
    )
    assert payload["action"] == "generate"
    assert payload["id"] != seeded["active_id"]
    assert payload["active_id"] == seeded["active_id"]
    assert payload["script"] == "大家好，这是新草稿。"
    assert specs["opening"].audio_path.read_bytes() == original
    assert specs["opening"].text_path.read_text(encoding="utf-8") == original_text
    assert not (tmp_path / "shows/opening/versions" / payload["id"] / "audio.wav").exists()
    assert llm.messages
    assert "30" in llm.messages[0][1]["content"]


@pytest.mark.asyncio
async def test_save_tts_switches_active_and_copies_local_show(tmp_path) -> None:
    specs = _show_specs(tmp_path)
    original = specs["opening"].audio_path.read_bytes()
    library = ShowLibrary(tmp_path / "shows", specs)
    seeded = library.list_payload("opening")
    script = "大家好，这是确认播出的稿。\n\n【杭州话】\n\n今朝开搞！\n"
    payload = await library.save_tts_payload(
        Config(),
        kind="opening",
        script=script,
        brief="杭州杭创营",
        duration_s=60,
        dialect_id="hangzhou",
        tts_client_cls=_FakeTTS,
    )
    assert payload["action"] == "save_tts"
    assert payload["active_id"] == payload["id"]
    assert payload["active_id"] != seeded["active_id"]
    assert specs["opening"].text_path.read_text(encoding="utf-8").startswith("大家好，这是确认播出的稿")
    assert specs["opening"].audio_path.read_bytes() != original
    assert (tmp_path / "shows/opening/versions" / payload["id"] / "audio.wav").is_file()


@pytest.mark.asyncio
async def test_handle_show_content_action_routes_list_and_generate(tmp_path) -> None:
    specs = _show_specs(tmp_path)
    library = ShowLibrary(tmp_path / "shows", specs)
    listed = await handle_show_content_action(
        library, Config(), {"action": "list", "kind": "training"}
    )
    assert listed["kind"] == "training"
    assert listed["versions"]
    generated = await handle_show_content_action(
        library,
        Config(),
        {
            "action": "generate",
            "kind": "training",
            "brief": "把流程讲清楚",
            "duration_s": 90,
            "dialect": "mandarin",
        },
        llm_client=_FakeLLM("培训草稿。"),
    )
    assert generated["script"] == "培训草稿。"
    assert generated["active_id"] == listed["active_id"]


def test_opening_audio_cli_uses_shared_synth() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_opening_audio.py"
    ).read_text(encoding="utf-8")
    assert "synthesize_show_audio" in source
    assert "BailianTTSClient" not in source
