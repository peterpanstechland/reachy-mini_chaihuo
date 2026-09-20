"""Camera module — captures frames from Reachy Mini's built-in USB camera.

The Reachy Mini head contains a Lite camera (USB Video Class device) that
appears as a standard UVC camera when connected via USB.

Detection strategy:
  - **macOS:** Uses AVFoundation. Search camera names for "Reachy".
    Falls back to index 0 if not found.
  - **Jetson:** Uses V4L2. Searches /dev/video* for "Reachy" in the device
    name. Falls back to /dev/video0.

Frame capture: OpenCV (cv2.VideoCapture) → single JPEG frame → bytes.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("chaihuo_reachy.camera")


def _reap_subprocess(process: subprocess.Popen[bytes]) -> None:
    """Wait for a terminated subprocess, escalating to SIGKILL if it lingers.

    Runs on a daemon thread from ``close()`` so the event loop is never
    blocked; guarantees the child is reaped (or killed) so a camera ffmpeg
    can never survive as an orphan holding the device.
    """
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg 未在 5 秒内退出，发送强制终止")
        process.kill()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg 强制终止也超时（进程可能已失控）")


def visual_quality_issue(jpeg: bytes) -> str | None:
    """Return a truthful user-facing issue when a frame is unusable."""
    if not jpeg:
        return "没有拿到画面。"
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return "拿到的画面数据无效。"
    height, width = frame.shape[:2]
    if height < 120 or width < 160:
        return "拿到的画面尺寸太小，我看不清。"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    dark_ratio = float((gray < 25).mean())
    bright_ratio = float((gray > 235).mean())
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if mean < 25 or dark_ratio > 0.65:
        return "画面太暗了，我现在看不清。"
    if mean > 238 or bright_ratio > 0.85:
        return "画面过曝了，我现在看不清。"
    # Blur gate: Laplacian variance is naturally low (<15) for static/low
    # texture scenes like a car interior — only report blur when BOTH
    # contrast and focus are extremely low, otherwise the VLM judges.
    if std < 4 and focus < 4:
        return "镜头可能被挡住或没有对准场景，画面太模糊了，我看不清。"
    return None


_REACHY_AVFOUNDATION_NAME = "Reachy Mini Camera"
_AVFOUNDATION_PREFIX = "avfoundation:"


def _avfoundation_video_devices() -> dict[str, int]:
    """Return exact AVFoundation video names mapped to their current indexes."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {}
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    import re

    devices: dict[str, int] = {}
    in_video_section = False
    for line in result.stderr.splitlines():
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if not in_video_section:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line.strip())
        if match:
            devices[match.group(2).strip()] = int(match.group(1))
    return devices


def _find_named_reachy_camera_macos() -> str | None:
    """Return an exact AVFoundation selector for Reachy Mini.

    OpenCV and FFmpeg do not use the same camera indexes on macOS.  In
    particular, FFmpeg reported Reachy as index 1 on the development Mac
    while OpenCV exposed it as index 0.  Selecting by the AVFoundation name
    avoids silently opening the MacBook camera after a reconnect.

    Detection strategy (tried in order):
      1. system_profiler SPCameraDataType — most reliable, no external deps
      2. ffmpeg -list_devices — works even when system_profiler doesn't show
         the camera (e.g. privacy settings block it)
    """
    # Strategy 1: system_profiler (fast, no extra deps)
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        result = None

    if result is not None and f"{_REACHY_AVFOUNDATION_NAME}:" in result.stdout:
        selector = f"{_AVFOUNDATION_PREFIX}{_REACHY_AVFOUNDATION_NAME}"
        logger.info("Found Reachy camera by AVFoundation name: %s", selector)
        return selector

    # Strategy 2: ffmpeg -list_devices (fallback when system_profiler misses it)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        try:
            list_result = subprocess.run(
                [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", '""'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # ffmpeg writes device list to stderr
            output = list_result.stderr
            for line in output.split("\n"):
                if "Reachy" in line:
                    # Extract the device name from ffmpeg output:
                    # [AVFoundation indev @ ...] [0] FaceTime HD Camera
                    # [AVFoundation indev @ ...] [1] Reachy Mini Camera
                    import re
                    m = re.search(r"\]\s*(.+)$", line.strip())
                    if m:
                        name = m.group(1).strip()
                        selector = f"{_AVFOUNDATION_PREFIX}{name}"
                        logger.info(
                            "Found Reachy camera via ffmpeg: %s", selector
                        )
                        return selector
        except (OSError, subprocess.SubprocessError):
            pass

    return None


def _find_reachy_camera_index_linux() -> int | str | None:
    """Find the Reachy Mini camera on Linux.

    Video node numbers change after replug. Prefer the named capture node
    (index=0) and never fall back to the laptop webcam.
    """
    import glob
    import os

    capture_nodes: list[str] = []
    metadata_nodes: list[str] = []
    for node in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            name = Path(node, "name").read_text(encoding="utf-8").strip().lower()
            if "reachy" not in name:
                continue
            dev = f"/dev/{Path(node).name}"
            index_path = Path(node, "index")
            index = (
                index_path.read_text(encoding="utf-8").strip()
                if index_path.exists()
                else "0"
            )
            if index == "0":
                capture_nodes.append(dev)
            else:
                metadata_nodes.append(dev)
        except Exception:
            pass
    if len(capture_nodes) == 1:
        logger.info("Found Reachy camera (sysfs capture): %s", capture_nodes[0])
        return capture_nodes[0]
    if len(capture_nodes) > 1:
        logger.warning(
            "Multiple Reachy capture nodes %s; using %s",
            capture_nodes,
            capture_nodes[0],
        )
        return capture_nodes[0]
    if metadata_nodes:
        logger.warning(
            "Reachy camera metadata-only nodes %s; using %s",
            metadata_nodes,
            metadata_nodes[0],
        )
        return metadata_nodes[0]

    # v4l2-ctl probe (when installed)
    for video_dev in sorted(glob.glob("/dev/video*")):
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", video_dev, "--all"],
                capture_output=True, text=True, timeout=5,
            )
            if "reachy" in result.stdout.lower():
                logger.info("Found Reachy camera: %s", video_dev)
                return video_dev
        except Exception:
            pass

    # Check by-name symlinks
    by_id = "/dev/v4l/by-id/"
    if os.path.exists(by_id):
        for entry in os.listdir(by_id):
            if "reachy" in entry.lower():
                path = os.path.join(by_id, entry)
                logger.info("Found Reachy camera by-id: %s", path)
                return path

    return None


def find_reachy_camera(config_value: int | str = "auto") -> int | str:
    """Resolve the Reachy Mini camera device.

    Args:
        config_value: "auto" | int index | "/dev/video0" path.

    Returns:
        OpenCV-compatible camera index (int) or device path (str).

    On macOS, this function NEVER silently falls back to camera index 0
    (which is typically the Mac's built-in FaceTime camera). Instead it
    returns a named AVFoundation selector that will produce a clear error
    if FFmpeg is missing or the camera isn't connected.
    """
    if config_value != "auto":
        return config_value

    system = platform.system()
    if system == "Darwin":
        named = _find_named_reachy_camera_macos()
        if named is not None:
            return named
        # Camera indexes are different between OpenCV and FFmpeg and can also
        # be reordered by Continuity Camera. Never infer the robot camera from
        # FPS or image content: those heuristics can select the MacBook camera.
        raise RuntimeError(
            "未检测到名为 'Reachy Mini Camera' 的 AVFoundation 设备；"
            "拒绝回退到 Mac 前置摄像头"
        )
    elif system == "Linux":
        dev = _find_reachy_camera_index_linux()
        if dev is not None:
            return dev
        raise RuntimeError(
            "未检测到名为 Reachy Mini Camera 的 USB 摄像头；拒绝回退到笔记本摄像头"
        )

    logger.warning("Reachy camera not found — falling back to camera 0")
    return 0


def v4l2_open_candidates(device: int | str) -> list[int | str]:
    """Return the sysfs path first, then the numeric index OpenCV 5 accepts."""
    candidates: list[int | str] = [device]
    if isinstance(device, str):
        name = Path(device).name
        if name.startswith("video") and name[5:].isdigit():
            index = int(name[5:])
            if index != device:
                candidates.append(index)
    return candidates


class Camera:
    """Capture still frames from Reachy Mini's USB camera.

    Usage::

        cam = Camera(device=find_reachy_camera())
        if cam.open():
            jpeg_bytes = cam.capture_jpeg()
            cam.close()
    """

    def __init__(
        self,
        device: int | str = "auto",
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._device = find_reachy_camera(device)
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._ffmpeg: subprocess.Popen[bytes] | None = None
        self._ffmpeg_buffer = bytearray()
        self._prefetched_jpeg: bytes | None = None
        self._recover_lock = threading.Lock()
        self._last_recover_at = 0.0
        self._read_failures = 0

    def open(self) -> bool:
        """Open the camera. Returns True on success."""
        if (
            platform.system() == "Darwin"
            and isinstance(self._device, str)
            and self._device.startswith(_AVFOUNDATION_PREFIX)
        ):
            return self._open_named_avfoundation()

        if platform.system() == "Linux":
            if not self._open_v4l2():
                logger.error("Cannot open camera device: %s", self._device)
                return False
        else:
            self._cap = cv2.VideoCapture(self._device)
            if not self._cap.isOpened():
                logger.error("Cannot open camera device: %s", self._device)
                self._cap = None
                return False
        assert self._cap is not None

        if platform.system() == "Linux":
            self._cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if platform.system() == "Linux":
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Warm up: discard first few frames (auto-exposure settling)
        for _ in range(5):
            self._cap.read()
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        fourcc_value = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
        )
        logger.info(
            "Camera opened: device=%s requested=%dx%d negotiated=%dx%d@%.1f format=%s",
            self._device,
            self._width,
            self._height,
            actual_width,
            actual_height,
            actual_fps,
            actual_fourcc,
        )
        return True

    def _open_v4l2(self) -> bool:
        """Open the Reachy node, falling back to the numeric V4L2 index.

        OpenCV 5 can refuse ``/dev/videoN`` as a capture name even when the
        node is valid. The sysfs path is still the stable identity; index is
        only the open handle.
        """
        for device in v4l2_open_candidates(self._device):
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if cap.isOpened():
                if device != self._device:
                    logger.info(
                        "V4L2 无法按路径打开 %s，改用索引 %s",
                        self._device,
                        device,
                    )
                    self._device = device
                self._cap = cap
                return True
            cap.release()
        self._cap = None
        return False

    def _open_named_avfoundation(self) -> bool:
        """Resolve the exact Reachy name, then open its current AVFoundation index."""
        name = self._device.removeprefix(_AVFOUNDATION_PREFIX)
        index = _avfoundation_video_devices().get(name)
        if index is None:
            logger.error("Exact AVFoundation device not found: %s", name)
            return False
        self._cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            logger.error("Cannot open Reachy AVFoundation index %d", index)
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        for _ in range(5):
            self._cap.read()

        logger.info(
            "Camera opened by exact name: device=%s index=%d resolution=%dx%d",
            name,
            index,
            self._width,
            self._height,
        )
        return True

    def _read_ffmpeg_jpeg(self) -> bytes | None:
        process = self._ffmpeg
        if process is None or process.stdout is None or process.poll() is not None:
            return None
        while True:
            start = self._ffmpeg_buffer.find(b"\xff\xd8")
            if start >= 0:
                end = self._ffmpeg_buffer.find(b"\xff\xd9", start + 2)
                if end >= 0:
                    frame = bytes(self._ffmpeg_buffer[start : end + 2])
                    del self._ffmpeg_buffer[: end + 2]
                    return frame
            chunk = process.stdout.read(8192)
            if not chunk:
                return None
            self._ffmpeg_buffer.extend(chunk)
            if len(self._ffmpeg_buffer) > 8 * 1024 * 1024:
                del self._ffmpeg_buffer[:-2]

    def close(self) -> None:
        """Close the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera closed")
        if self._ffmpeg is not None:
            process = self._ffmpeg
            self._ffmpeg = None
            if process.poll() is None:
                process.terminate()
                # Don't block the event loop — let the OS reap the process.
                # The ffmpeg process reads from a pipe that will close, so it
                # will exit on its own shortly after terminate().  The reaper
                # thread guarantees wait() still runs, so a lingerer is
                # SIGKILLed instead of surviving as an orphan holding the
                # camera device.
                threading.Thread(
                    target=_reap_subprocess,
                    args=(process,),
                    name="ffmpeg-reaper",
                    daemon=True,
                ).start()
            self._ffmpeg_buffer.clear()
            self._prefetched_jpeg = None
            logger.info("Named AVFoundation camera closed")

    def _device_missing(self) -> bool:
        return (
            isinstance(self._device, str)
            and self._device.startswith("/dev/")
            and not Path(self._device).exists()
        )

    def _recover(self) -> bool:
        """Re-resolve the Reachy node after USB re-enumeration."""
        now = time.monotonic()
        with self._recover_lock:
            if now - self._last_recover_at < 1.5:
                return False
            self._last_recover_at = now
            previous = self._device
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            try:
                self._device = find_reachy_camera("auto")
            except RuntimeError as exc:
                logger.warning("Camera recover: %s", exc)
                return False
            logger.warning("Camera recovering: %s → %s", previous, self._device)
            if not self.open():
                return False
            self._read_failures = 0
            return True

    def _read_bgr(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened() or self._device_missing():
            if not self._recover():
                return None
        assert self._cap is not None
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._read_failures = 0
            return frame
        self._read_failures += 1
        if self._read_failures >= 3 or self._device_missing():
            if self._recover() and self._cap is not None:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    self._read_failures = 0
                    return frame
        return None

    def capture_jpeg(self, quality: int = 85) -> bytes | None:
        """Capture a single frame and encode as JPEG.

        Returns:
            JPEG bytes, or None if capture failed.
        """
        if self._cap is None and self._ffmpeg is not None:
            if self._prefetched_jpeg is not None:
                frame = self._prefetched_jpeg
                self._prefetched_jpeg = None
                return frame
            return self._read_ffmpeg_jpeg()

        frame = self._read_bgr()
        if frame is None:
            return None

        success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            return None

        return jpeg.tobytes()

    def capture_frame(self) -> np.ndarray | None:
        """Capture a single frame as a numpy array (BGR)."""
        if self._ffmpeg is not None:
            jpeg = self.capture_jpeg()
            if not jpeg:
                return None
            array = np.frombuffer(jpeg, dtype=np.uint8)
            return cv2.imdecode(array, cv2.IMREAD_COLOR)
        return self._read_bgr()

    @property
    def is_open(self) -> bool:
        return (
            self._cap is not None
            and self._cap.isOpened()
        ) or (
            self._ffmpeg is not None
            and self._ffmpeg.poll() is None
        )

    # ── CameraBackend interface compat ─────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True if camera is connected and delivering frames."""
        return self.is_open

    @property
    def backend_name(self) -> str:
        if (
            platform.system() == "Darwin"
            and isinstance(self._device, str)
            and self._device.startswith(_AVFOUNDATION_PREFIX)
        ):
            return "opencv_avfoundation_exact_name"
        return "opencv_direct"

    def read(self) -> np.ndarray | None:
        """Return latest BGR frame (CameraBackend compat)."""
        return self.capture_frame()
