"""Dance backing-track helpers — track resolution, PCM reading, BPM detection.

The engine plays a music file while the robot dances and paces the
choreography to the track's beat.  ``detect_bpm`` estimates the tempo from
the audio energy envelope via autocorrelation (pure numpy, no extra
dependencies); when it fails (silence, speech, odd format) the caller falls
back to a default beat length and the dance simply runs at that tempo.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

import numpy as np

from chaihuo_reachy.audio_frontend import pcm16_rms

logger = logging.getLogger("chaihuo_reachy.music")

_MIN_BPM = 70.0
_MAX_BPM = 180.0
_ENVELOPE_FRAME_S = 0.02

# BPM of the tracks synthesized by scripts/gen_dance_music.py.  The
# autocorrelation detector misreads these synthetic patterns (offbeat
# hats win over the kick), so known styles use the table and unknown
# tracks fall back to ``detect_bpm``.
STYLE_BPM: dict[str, float] = {
    "happy": 120.0,
    "swing": 100.0,
    "robot": 90.0,
    "elegant": 80.0,
    "funky": 132.0,
    "silly": 112.0,
}


def resolve_track(music_dir: Path | str, style: str) -> Path | None:
    """Pick the backing track for a dance style.

    ``<music_dir>/<style>.wav`` wins; otherwise any ``*.wav`` in the dir.
    Returns None when the directory is missing or empty.
    """
    music_dir = Path(music_dir)
    track = music_dir / f"{style}.wav"
    if track.is_file():
        return track
    candidates = sorted(music_dir.glob("*.wav")) if music_dir.is_dir() else []
    return candidates[0] if candidates else None


def read_track(path: Path) -> tuple[int, bytes] | None:
    """Read a 16-bit mono WAV as (sample_rate, int16 PCM bytes), or None.

    Non-16-bit/stereo files are rejected with a warning — the engine feeds
    the bytes straight to the speaker.
    """
    return _read_wav_mono16(path)


def _read_wav_mono16(path: Path) -> tuple[int, bytes] | None:
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                logger.warning("伴奏 %s 不是 16-bit 单声道 WAV，跳过", path)
                return None
            frames = wf.readframes(wf.getnframes())
    except Exception:
        logger.warning("伴奏 %s 读取失败，跳过", path, exc_info=True)
        return None
    if not frames:
        logger.warning("伴奏 %s 为空，跳过", path)
        return None
    return wf.getframerate(), frames


def resample_pcm(pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
    """Simple linear resampling for int16 PCM (no-op when rates match)."""
    if src_sr == dst_sr or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(len(samples) * dst_sr / src_sr)
    indices = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.int16).tobytes()


def load_audio(path: Path, target_sr: int | None = None) -> tuple[int, bytes] | None:
    """Read any supported track as (sample_rate, int16 MONO PCM bytes).

    ``.wav`` uses the existing wave reader; ``.mp3`` is decoded via
    miniaudio and downmixed stereo→mono.  When miniaudio is unavailable the
    caller's sibling ``.wav`` (same stem) is tried.  Returns None on a
    missing or undecodable file.
    """
    path = Path(path)
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".wav":
        info = _read_wav_mono16(path)
    elif suffix == ".mp3":
        info = None
        try:
            import miniaudio

            decoded = miniaudio.decode_file(
                str(path), output_format=miniaudio.SampleFormat.SIGNED16
            )
            x = np.frombuffer(decoded.samples, dtype=np.int16)
            if decoded.nchannels > 1:
                x = x.reshape(-1, decoded.nchannels).mean(axis=1).astype(np.int16)
            if x.size:
                info = (decoded.sample_rate, x.tobytes())
        except ImportError:
            logger.warning("miniaudio 不可用，尝试同目录 WAV 版本: %s", path)
            info = _read_wav_mono16(path.with_suffix(".wav"))
        except Exception:
            logger.warning("MP3 %s 解码失败，尝试同目录 WAV 版本", path, exc_info=True)
            info = _read_wav_mono16(path.with_suffix(".wav"))
        if info is None:
            return None
    else:
        return None
    if target_sr is not None and info[0] != target_sr:
        return target_sr, resample_pcm(info[1], info[0], target_sr)
    return info


def _energy_envelope(pcm: bytes, sr: int) -> tuple[np.ndarray, float] | None:
    """RMS frames for BPM and buildup. Same window ``detect_bpm`` uses."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    frame = max(1, int(sr * _ENVELOPE_FRAME_S))
    n_frames = len(x) // frame
    if n_frames < 60:  # need ~1.2 s minimum for a meaningful envelope
        return None
    env = np.sqrt(
        np.mean(x[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1)
    )
    return env, sr / frame


def _bpm_from_envelope(env: np.ndarray, env_fps: float) -> float | None:
    centered = env - env.mean()
    ac = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    # Ignore the first 0.1 s (self-lag) — we want the first beat period.
    ac[: max(1, int(0.1 * env_fps))] = 0.0

    lag_min = int(env_fps * 60.0 / _MAX_BPM)
    lag_max = max(lag_min + 1, int(env_fps * 60.0 / _MIN_BPM))
    if lag_max >= len(ac):
        return None
    seg = ac[lag_min : lag_max + 1]
    if seg.size == 0 or float(seg.max()) <= 0.0:
        return None
    lag = lag_min + int(np.argmax(seg))
    return round(60.0 / (lag / env_fps), 1)


def detect_bpm(pcm: bytes, sr: int) -> float | None:
    """Estimate the tempo of a 16-bit mono PCM buffer (energy autocorrelation).

    Returns BPM in [70, 180] or None when no stable beat is found.
    """
    info = _energy_envelope(pcm, sr)
    if info is None:
        return None
    return _bpm_from_envelope(*info)


class AmbientBeatTracker:
    """Rolling mic window → BPM + beat phase for live dance.

    Reuses ``detect_bpm`` on a few seconds of capture. Motion stays on a
    tempo grid snapped to the latest onset so the robot can follow room
    music without playing a local file.
    """

    def __init__(self, sample_rate: int = 16000, window_s: float = 6.0) -> None:
        self.sample_rate = max(8000, int(sample_rate or 16000))
        self._window_bytes = self.sample_rate * 2 * max(3, int(window_s))
        self._pcm = bytearray()
        self._lock = threading.Lock()
        self._bpm = 120.0
        self._bpm_locked = False
        self._last_bpm_at = 0.0
        self._last_onset = 0.0
        self._noise = 0.012
        self._rms = 0.0
        self._buildup = 0.0

    @property
    def bpm(self) -> float:
        with self._lock:
            return self._bpm

    @property
    def rms(self) -> float:
        with self._lock:
            return self._rms

    @property
    def buildup(self) -> float:
        with self._lock:
            return self._buildup

    def push(self, pcm: bytes, *, now: float | None = None) -> None:
        if not pcm:
            return
        now = time.monotonic() if now is None else now
        rms = pcm16_rms(pcm)
        with self._lock:
            self._pcm.extend(pcm)
            overflow = len(self._pcm) - self._window_bytes
            if overflow > 0:
                del self._pcm[:overflow]
            self._rms = rms
            if rms < self._noise * 1.5:
                self._noise = 0.95 * self._noise + 0.05 * max(rms, 1e-4)
            min_gap = 60.0 / _MAX_BPM
            if rms >= max(self._noise * 2.4, 0.018) and now - self._last_onset >= min_gap:
                self._last_onset = now
            if now - self._last_bpm_at < 1.5:
                return
            if len(self._pcm) < self.sample_rate * 2 * 3:
                return
            info = _energy_envelope(bytes(self._pcm), self.sample_rate)
            self._last_bpm_at = now
            if info is None:
                return
            env, env_fps = info
            half = max(1, env.size // 2)
            early = float(np.mean(env[:half]))
            late = float(np.mean(env[half:]))
            climb = min(1.0, max(0.0, (late / max(early, 1e-4) - 1.0) / 0.7))
            if self._buildup == 0.0:
                self._buildup = climb
            else:
                self._buildup = 0.7 * self._buildup + 0.3 * climb
            estimated = _bpm_from_envelope(env, env_fps)
            if estimated is None:
                return
            if not self._bpm_locked:
                self._bpm = estimated
                self._bpm_locked = True
            else:
                self._bpm = round(0.7 * self._bpm + 0.3 * estimated, 1)

    def beat_for_elapsed(self, elapsed: float, origin: float) -> tuple[dict[str, float], float]:
        with self._lock:
            bpm = self._bpm
            rms = self._rms
            last_onset = self._last_onset
            buildup = self._buildup
        interval = 60.0 / bpm if bpm > 0 else 0.5
        phase0 = ((last_onset - origin) % interval) if last_onset > 0 else 0.0
        beat_index = math.floor((elapsed - phase0) / interval)
        beat_time = phase0 + max(0, beat_index) * interval
        strength = min(1.0, max(0.25, rms / 0.08))
        return {"strength": strength, "rms": rms, "buildup": buildup}, beat_time
