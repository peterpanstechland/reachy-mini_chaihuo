from __future__ import annotations

import asyncio
import inspect

import pytest

from chaihuo_reachy.dashboard import (
    ChatMessageStore,
    DashboardHub,
    run_websocket_session,
)
from chaihuo_reachy.main import DASHBOARD_HTML, run_dashboard


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.controls: asyncio.Queue[dict | None] = asyncio.Queue()

    async def send_json(self, message: dict) -> None:
        self.sent.append(message)

    async def receive_json(self) -> dict:
        message = await self.controls.get()
        if message is None:
            raise RuntimeError("disconnected")
        return message


@pytest.mark.asyncio
async def test_websocket_pushes_live_partial_without_client_polling() -> None:
    hub = DashboardHub()
    websocket = FakeWebSocket()

    async def handle_control(_ws: FakeWebSocket, _message: dict) -> None:
        return None

    task = asyncio.create_task(
        run_websocket_session(
            websocket,
            hub,
            lambda: [{"type": "state", "state": "listening"}],
            handle_control,
        )
    )
    await asyncio.sleep(0)
    event = hub.publish({"type": "transcript", "text": "正在实时识别", "final": False})
    await asyncio.sleep(0.01)
    assert event in websocket.sent
    await websocket.controls.put(None)
    with pytest.raises(RuntimeError, match="disconnected"):
        await task
    assert hub.subscriber_count == 0


@pytest.mark.asyncio
async def test_websocket_receives_controls_while_sender_is_waiting() -> None:
    hub = DashboardHub()
    websocket = FakeWebSocket()
    handled = asyncio.Event()

    async def handle_control(_ws: FakeWebSocket, message: dict) -> None:
        if message["type"] == "get_state":
            handled.set()

    task = asyncio.create_task(
        run_websocket_session(websocket, hub, lambda: [], handle_control)
    )
    await websocket.controls.put({"type": "get_state"})
    await asyncio.wait_for(handled.wait(), timeout=0.2)
    await websocket.controls.put(None)
    with pytest.raises(RuntimeError):
        await task


def test_history_keeps_final_not_stale_partial_trail() -> None:
    hub = DashboardHub()
    hub.publish({"type": "transcript", "text": "你", "final": False})
    hub.publish({"type": "transcript", "text": "你好", "final": False})
    final = hub.publish({"type": "transcript", "text": "你好小柴", "final": True})
    replayed = hub.replay()
    # Final transcript is included
    assert final in replayed
    # ASR history replay contains the final text
    history_events = [e for e in replayed if e.get("type") == "asr_history_replay"]
    assert len(history_events) == 1
    items = history_events[0].get("items", [])
    assert any(item["text"] == "你好小柴" for item in items)
    # No stale partials
    partial_in_replay = [
        e for e in replayed
        if e.get("type") == "transcript" and not e.get("final")
    ]
    assert not partial_in_replay


def test_browser_uses_page_host_and_normalized_chat_protocol() -> None:
    assert 'window.location.host+"/ws"' in DASHBOARD_HTML
    assert 'window.location.protocol==="https:"?"wss:":"ws:"' in DASHBOARD_HTML
    assert 'type:"chat_send"' in DASHBOARD_HTML
    assert 'typeof globalThis.crypto.randomUUID==="function"' in DASHBOARD_HTML
    assert 'client_message_id:clientMessageId()' in DASHBOARD_HTML
    assert 'm.type==="chat_history"' in DASHBOARD_HTML
    assert 'm.type==="chat_message_upsert"' in DASHBOARD_HTML
    assert 'm.type==="chat_message_delta"' in DASHBOARD_HTML
    # The WeChat-style Dashboard must retain the original device controls.
    assert 'id="wakeBtn"' in DASHBOARD_HTML
    assert 'id="energyListenBtn"' in DASHBOARD_HTML
    assert 'id="openingBtn"' in DASHBOARD_HTML
    assert 'id="trainingBtn"' in DASHBOARD_HTML
    assert 'id="trainingFaq"' in DASHBOARD_HTML
    assert 'id="volumeRange"' in DASHBOARD_HTML
    assert 'type:"get_wake_word"' in DASHBOARD_HTML
    assert 'type:"set_wake_word"' in DASHBOARD_HTML
    assert 'type:"energy_listen"' in DASHBOARD_HTML
    assert 'type:"opening_show"' in DASHBOARD_HTML
    assert 'type:"training_show"' in DASHBOARD_HTML
    assert "Wio Terminal 常见问题" in DASHBOARD_HTML
    assert "renderTrainingFaq" in DASHBOARD_HTML
    assert 'type:"get_volume"' in DASHBOARD_HTML
    assert 'type:"set_volume"' in DASHBOARD_HTML
    # Opening the HTML file directly must not issue file:// API requests.
    assert 'window.location.protocol==="file:"' in DASHBOARD_HTML
    assert 'http://localhost:8640/' in DASHBOARD_HTML
    assert "dbCam" not in DASHBOARD_HTML
    assert 'm.type==="asr_status"' in DASHBOARD_HTML
    assert 'id="modelInfo"' in DASHBOARD_HTML
    assert 'id="searchInfo"' in DASHBOARD_HTML
    assert 'id="audioFrontend"' in DASHBOARD_HTML
    assert 'id="locationInfo"' in DASHBOARD_HTML
    assert "已联网核验" in DASHBOARD_HTML
    assert "navigator.geolocation.watchPosition" in DASHBOARD_HTML
    assert 'type:"browser_location"' in DASHBOARD_HTML
    assert 'id="beatDanceBtn"' in DASHBOARD_HTML
    assert "大东北我的家乡" in DASHBOARD_HTML
    assert 'type:"motion_dance_loop"' in DASHBOARD_HTML
    assert "setBeatDanceState(!!m.dance_loop_active" in DASHBOARD_HTML
    assert 'id="bgmSelect"' in DASHBOARD_HTML
    assert 'id="bgmBtn"' in DASHBOARD_HTML
    assert 'type:"get_bgm"' in DASHBOARD_HTML
    assert 'type:"bgm_control"' in DASHBOARD_HTML
    assert "播放音乐" in DASHBOARD_HTML
    assert 'id="reachyCard"' in DASHBOARD_HTML
    assert 'id="reachyPower"' in DASHBOARD_HTML
    assert 'id="movePlayBtn"' in DASHBOARD_HTML
    assert 'type:"reachy_link"' in DASHBOARD_HTML
    assert 'action:on?"connect":"disconnect"' in DASHBOARD_HTML
    assert 'type:"reachy_move"' in DASHBOARD_HTML
    assert "robotLink(m.robot_link)" in DASHBOARD_HTML
    assert 'id="gesturePreview"' in DASHBOARD_HTML
    assert 'id="gesturePreviewCamera"' in DASHBOARD_HTML
    assert 'id="gesturePreviewOverlay"' in DASHBOARD_HTML
    assert "setGesturePreview(gestureActive" in DASHBOARD_HTML
    assert 'cam.src="/camera/stream"' in DASHBOARD_HTML
    assert 'id="showEditor"' in DASHBOARD_HTML
    assert 'id="showEditorBtn"' in DASHBOARD_HTML
    assert 'id="showHistory"' in DASHBOARD_HTML
    assert 'id="showScript"' in DASHBOARD_HTML
    assert 'id="showBrief"' in DASHBOARD_HTML
    assert 'id="showDuration"' in DASHBOARD_HTML
    assert 'id="showDialect"' in DASHBOARD_HTML
    assert "文案配置" in DASHBOARD_HTML
    assert "function setShowEditor" in DASHBOARD_HTML
    assert 'type:"show_content"' in DASHBOARD_HTML
    assert 'm.type==="show_content"' in DASHBOARD_HTML
    assert 'm.type==="show_content_status"' in DASHBOARD_HTML
    assert "保存并转语音" in DASHBOARD_HTML


def test_dashboard_embeds_daemon_link_controls() -> None:
    source = inspect.getsource(run_dashboard)
    assert "daemon_http.fetch_daemon_status" in source
    assert "_refresh_robot_link" in source
    assert "event_type == \"reachy_move\"" in source
    assert "event_type == \"reachy_link\"" in source
    assert "await _try_connect_daemon(cfg)" in source
    assert "await _close_reachy_runtime(current)" in source
    assert "await _stop_recorded_move()" in source
    assert 'event_type == "show_content"' in source
    assert "handle_show_content_action" in source
    assert "show_content_busy" in source


def test_opening_show_respects_live_dashboard_volume() -> None:
    source = inspect.getsource(run_dashboard)
    opening_start = source.index("async def _run_opening_show")
    opening_end = source.index("engine.on_state_change", opening_start)
    opening_source = source[opening_start:opening_end]
    assert "engine._audio.volume =" not in opening_source

    opening_gate_start = source.index("if show_active and event_type not in")
    opening_gate_end = source.index("if event_type == \"gesture_mode\"", opening_gate_start)
    opening_gate_source = source[opening_gate_start:opening_gate_end]
    assert '"set_volume"' in opening_gate_source
    assert '"motion_pose"' not in opening_gate_source
    assert '"reachy_link"' not in opening_gate_source
    assert '"reachy_move"' not in opening_gate_source


def test_chat_store_replays_both_sides_sources_and_capture() -> None:
    store = ChatMessageStore(message_limit=10, capture_limit=2)
    user, assistant = store.begin_turn(
        "turn-1", "外面有什么？", source="dashboard", client_message_id="client-1"
    )
    assert user["role"] == "user"
    assert assistant["role"] == "assistant"
    capture_id, updated = store.attach_capture(
        "turn-1", b"\xff\xd8photo", label="车外后视"
    )
    assert updated is not None
    final = store.finalize(
        "turn-1",
        "画面中清晰可见一辆车。",
        sources=[{"title": "日记", "url": "https://example.test"}],
    )
    assert final and final["attachments"][0]["capture_id"] == capture_id
    history = store.history_event()
    assert [message["role"] for message in history["messages"]] == ["user", "assistant"]
    assert history["messages"][1]["sources"]
    assert store.get_capture(capture_id) == b"\xff\xd8photo"


def test_chat_store_limits_captures_and_clear() -> None:
    store = ChatMessageStore(message_limit=4, capture_limit=1)
    store.begin_turn("turn-1", "一", source="voice")
    first, _ = store.attach_capture("turn-1", b"first", label="Reachy 前置")
    second, _ = store.attach_capture("turn-1", b"second", label="Reachy 前置")
    assert store.get_capture(first) is None
    assert store.get_capture(second) == b"second"
    store.clear()
    assert store.history_event()["messages"] == []
