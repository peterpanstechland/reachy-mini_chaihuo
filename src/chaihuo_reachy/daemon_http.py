"""HTTP client for the Reachy Mini daemon control API.

The official daemon dashboard talks to these same routes.  皮皮虾 reuses
them for live connection status and recorded-move playback.  Start/stop
of the owned daemon stays in ``main._try_connect_daemon`` /
``main._close_reachy_runtime``.
"""

from __future__ import annotations

from typing import Any

import httpx

MOVE_DATASETS: tuple[tuple[str, str], ...] = (
    ("Dances", "pollen-robotics/reachy-mini-dances-library"),
    ("Emotions", "pollen-robotics/reachy-mini-emotions-library"),
)

_STATUS_LABELS = {
    "starting": "连接中",
    "stopping": "关闭中",
    "stopped": "已关闭",
    "not_initialized": "未初始化",
    "error": "连接失败",
}


class DaemonHttpError(RuntimeError):
    """The daemon HTTP API returned an unusable response."""


def daemon_url(host: str, port: int, path: str) -> str:
    """Build a daemon HTTP URL without adding extra configuration."""
    prefix = "/" if not path.startswith("/") else ""
    return f"http://{host}:{port}{prefix}{path}"


def dataset_catalog() -> list[dict[str, str]]:
    """Official recorded-move libraries shown in the daemon Move player."""
    return [{"id": repo, "label": label} for label, repo in MOVE_DATASETS]


async def fetch_daemon_status(
    host: str, port: int, *, timeout: float = 0.5
) -> dict[str, Any]:
    """Return ``/api/daemon/status`` or raise ``DaemonHttpError``."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(daemon_url(host, port, "/api/daemon/status"))
    if response.status_code != 200:
        raise DaemonHttpError(f"daemon 健康检查返回 HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise DaemonHttpError("daemon 健康检查返回了无效数据")
    return payload


def summarize_robot_link(
    payload: dict[str, Any] | None, sdk_status: dict[str, Any]
) -> dict[str, Any]:
    """Map daemon status + connect-time SDK fields into one dashboard card."""
    serial = sdk_status.get("serial_port")
    owner = str(sdk_status.get("daemon_owner") or "none")
    host = sdk_status.get("daemon_host")
    port = sdk_status.get("daemon_port")
    datasets = dataset_catalog()
    if payload is None:
        return {
            "available": False,
            "power_on": False,
            "busy": False,
            "state": "unavailable",
            "label": "已关闭",
            "detail": str(sdk_status.get("daemon_error") or "daemon 未连接"),
            "daemon_error": sdk_status.get("daemon_error"),
            "motor_control_mode": None,
            "serial_port": serial,
            "simulation": False,
            "datasets": datasets,
        }

    state = str(payload.get("state") or "").lower()
    backend = payload.get("backend_status")
    backend = backend if isinstance(backend, dict) else {}
    motor = str(backend.get("motor_control_mode") or "").lower() or None
    error = payload.get("error") or backend.get("error") or sdk_status.get("daemon_error")
    simulation = bool(payload.get("simulation_enabled"))
    busy = state in {"starting", "stopping"}

    if state == "running":
        power_on = True
        if motor == "disabled":
            label = "已连接"
        elif motor == "enabled":
            label = "已就绪"
        else:
            label = "已就绪" if sdk_status.get("robot_ready") else "已连接"
    elif state == "starting":
        power_on = True
        label = _STATUS_LABELS[state]
    elif state == "error":
        power_on = False
        label = _STATUS_LABELS[state]
    else:
        power_on = False
        label = _STATUS_LABELS.get(state, state or "未知")

    parts: list[str] = []
    if host:
        parts.append(f"{host}:{port}" if port else str(host))
    if owner and owner != "none":
        parts.append(owner)
    if serial:
        parts.append(str(serial).rsplit("/", 1)[-1])
    if motor:
        parts.append(f"电机 {motor}")
    if simulation:
        parts.append("模拟")
    detail = str(error) if (error and not power_on and state == "error") else (
        " · ".join(parts) or "Reachy Mini"
    )

    return {
        "available": True,
        "power_on": power_on,
        "busy": busy,
        "state": state or "unknown",
        "label": label,
        "detail": detail,
        "daemon_error": error,
        "motor_control_mode": motor,
        "serial_port": serial,
        "simulation": simulation,
        "datasets": datasets,
    }


async def list_recorded_moves(
    host: str, port: int, dataset: str, *, timeout: float = 20.0
) -> list[str]:
    """List recorded moves in an official Hugging Face dataset."""
    path = f"/api/move/recorded-move-datasets/list/{dataset}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(daemon_url(host, port, path))
    if response.status_code != 200:
        raise DaemonHttpError(f"读取动作库失败 HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, list):
        raise DaemonHttpError("动作库返回了无效数据")
    return [str(name) for name in payload]


async def play_recorded_move(
    host: str, port: int, dataset: str, move: str, *, timeout: float = 20.0
) -> str:
    """Start a recorded move on the daemon backend and return its UUID."""
    path = f"/api/move/play/recorded-move-dataset/{dataset}/{move}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(daemon_url(host, port, path))
    if response.status_code != 200:
        raise DaemonHttpError(f"播放动作失败 HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("uuid"):
        raise DaemonHttpError("播放动作未返回任务编号")
    return str(payload["uuid"])


async def stop_recorded_move(
    host: str, port: int, move_uuid: str, *, timeout: float = 5.0
) -> None:
    """Cancel a running recorded-move task."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            daemon_url(host, port, "/api/move/stop"),
            json={"uuid": move_uuid},
        )
    if response.status_code != 200:
        raise DaemonHttpError(f"停止动作失败 HTTP {response.status_code}")


async def list_running_moves(
    host: str, port: int, *, timeout: float = 2.0
) -> list[str]:
    """Return UUIDs of recorded moves the daemon is currently playing."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(daemon_url(host, port, "/api/move/running"))
    if response.status_code != 200:
        raise DaemonHttpError(f"查询动作状态失败 HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, list):
        return []
    running: list[str] = []
    for item in payload:
        if isinstance(item, dict) and item.get("uuid"):
            running.append(str(item["uuid"]))
        elif item:
            running.append(str(item))
    return running
