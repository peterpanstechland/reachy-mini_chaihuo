from __future__ import annotations

import httpx
import pytest

from chaihuo_reachy import daemon_http


def test_daemon_url_keeps_absolute_paths() -> None:
    assert daemon_http.daemon_url("127.0.0.1", 8000, "/api/daemon/status") == (
        "http://127.0.0.1:8000/api/daemon/status"
    )
    assert daemon_http.daemon_url("localhost", 8000, "api/move/running") == (
        "http://localhost:8000/api/move/running"
    )


def test_summarize_robot_link_disconnected() -> None:
    link = daemon_http.summarize_robot_link(
        None,
        {
            "daemon_error": "本机 daemon 端口已被占用",
            "daemon_host": "localhost",
            "daemon_port": 8000,
            "serial_port": "/dev/ttyACM0",
        },
    )
    assert link["available"] is False
    assert link["power_on"] is False
    assert link["label"] == "已关闭"
    assert link["detail"] == "本机 daemon 端口已被占用"
    assert [item["label"] for item in link["datasets"]] == ["Dances", "Emotions"]


def test_summarize_robot_link_ready_and_sleeping() -> None:
    ready = daemon_http.summarize_robot_link(
        {
            "state": "running",
            "simulation_enabled": False,
            "backend_status": {"motor_control_mode": "enabled"},
        },
        {
            "robot_ready": True,
            "daemon_owner": "owned",
            "daemon_host": "localhost",
            "daemon_port": 8000,
            "serial_port": "/dev/ttyACM0",
        },
    )
    assert ready["available"] is True
    assert ready["power_on"] is True
    assert ready["label"] == "已就绪"
    assert "localhost:8000" in ready["detail"]
    assert "ttyACM0" in ready["detail"]

    asleep = daemon_http.summarize_robot_link(
        {
            "state": "running",
            "backend_status": {"motor_control_mode": "disabled"},
        },
        {"robot_ready": True, "daemon_owner": "owned"},
    )
    assert asleep["power_on"] is True
    assert asleep["label"] == "已连接"


def test_summarize_robot_link_keeps_toggle_on_while_daemon_is_running() -> None:
    link = daemon_http.summarize_robot_link(
        {
            "state": "running",
            "backend_status": {"motor_control_mode": "disabled"},
        },
        {"robot_ready": True},
    )
    assert link["power_on"] is True
    assert link["label"] == "已连接"


@pytest.mark.asyncio
async def test_fetch_daemon_status_uses_official_route(monkeypatch) -> None:
    seen: list[str] = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"state": "running"}

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, url: str):
            seen.append(url)
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    payload = await daemon_http.fetch_daemon_status("localhost", 8000)
    assert payload == {"state": "running"}
    assert seen == ["http://localhost:8000/api/daemon/status"]


@pytest.mark.asyncio
async def test_fetch_daemon_status_rejects_non_json_object(monkeypatch) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return ["not", "a", "status"]

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, _url: str):
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    with pytest.raises(daemon_http.DaemonHttpError, match="无效数据"):
        await daemon_http.fetch_daemon_status("localhost", 8000)
