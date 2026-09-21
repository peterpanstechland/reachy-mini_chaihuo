"""Show-script library, Qwen drafting, and apply-to-LOCAL_SHOWS flow."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from chaihuo_reachy.opening import (
    DEFAULT_SHOW_DIALECT,
    LOCAL_SHOWS,
    PROJECT_ROOT,
    DialectVoice,
    LocalShowSpec,
    list_show_dialects,
    load_opening_audio,
    match_show_dialect,
    parse_opening_script,
    resolve_show_dialect,
    synthesize_show_audio,
)

logger = logging.getLogger("chaihuo_reachy.show_content")

DEFAULT_SHOWS_ROOT = PROJECT_ROOT / "data" / "shows"
SHOW_DURATION_PRESETS = (30, 60, 90, 120)
SHOW_CHARS_PER_SECOND = 4.5
SHOW_CONTENT_ACTIONS = frozenset({"list", "get", "generate", "save_tts"})

_SHOW_WRITER_RULES = (
    "现在你在写一段可以离线播放的口播稿，不是在和观众对话。\n"
    "不要使用情绪标签，不要调用工具，不要提日记检索、摄像头或系统提示。\n"
    "用皮皮虾第一人称口语，完整讲完，不要写成 1-3 句闲聊。"
)


def normalize_show_kind(kind: str) -> str:
    key = str(kind or "").strip()
    if key not in LOCAL_SHOWS:
        raise ValueError("文案类型必须是 opening 或 training")
    return key


def normalize_show_duration(value: Any) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        seconds = 60
    return min(SHOW_DURATION_PRESETS, key=lambda preset: abs(preset - seconds))


def location_text_from_engine(engine: Any) -> str:
    """Join live position fields and the session note for dialect matching."""
    parts: list[str] = []
    location = getattr(engine, "_location", None)
    position = getattr(location, "latest_position", None) if location is not None else None
    if position is not None:
        for key in ("province", "city", "district", "address"):
            value = getattr(position, key, "") or ""
            if value:
                parts.append(str(value))
    session = getattr(engine, "_session_location", None) or {}
    place = str(session.get("place") or "").strip()
    if place:
        parts.append(place)
    return " ".join(parts)


def show_writer_system_prompt(cfg: Any) -> str:
    persona = ""
    loader = getattr(cfg, "system_prompt", None)
    if callable(loader):
        persona = str(loader() or "").strip()
    first = persona.split("\n\n", 1)[0].strip() if persona else ""
    if not first:
        first = "你是柴火基地车上的皮皮虾。"
    return f"{first}\n\n{_SHOW_WRITER_RULES}"


def build_show_generate_messages(
    cfg: Any,
    *,
    kind: str,
    brief: str,
    duration_s: int,
    dialect: DialectVoice,
    location_text: str = "",
) -> list[dict[str, str]]:
    spec = LOCAL_SHOWS[normalize_show_kind(kind)]
    seconds = normalize_show_duration(duration_s)
    target_chars = max(80, int(seconds * SHOW_CHARS_PER_SECOND))
    if dialect.spoken_tail:
        dialect_rule = (
            f"正文用普通话。正文结束后空一行，单独写【{dialect.label}】，"
            f"再空一行写一段{dialect.label}口语收尾。"
            "不要把标题写进正文句子里。"
        )
    else:
        dialect_rule = "只写普通话正文，不要出现【xx话】标题或方言段落。"
    location_line = (
        f"可参考的现场地点：{location_text}" if location_text.strip() else "没有额外定位。"
    )
    brief_text = brief.strip() or "按现有场合写一版得体的口播。"
    user = (
        f"请写一份「{spec.label}」口播稿。\n"
        f"目标时长约 {seconds} 秒，按每秒 {SHOW_CHARS_PER_SECOND} 字，大约 {target_chars} 字。\n"
        f"{dialect_rule}\n"
        f"{location_line}\n"
        f"需求：\n{brief_text}\n"
        "只输出口播正文本身，不要解释，不要 markdown。"
    )
    return [
        {"role": "system", "content": show_writer_system_prompt(cfg)},
        {"role": "user", "content": user},
    ]


def _strip_generated_script(text: str) -> str:
    script = text.strip()
    if script.startswith("```"):
        lines = script.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        script = "\n".join(lines).strip()
    return script


async def generate_show_script(
    cfg: Any,
    *,
    kind: str,
    brief: str,
    duration_s: int,
    dialect: DialectVoice,
    location_text: str = "",
    llm_client: Any | None = None,
) -> str:
    messages = build_show_generate_messages(
        cfg,
        kind=kind,
        brief=brief,
        duration_s=duration_s,
        dialect=dialect,
        location_text=location_text,
    )
    chunks: list[str] = []
    if llm_client is not None:
        async for token in llm_client.chat_stream(messages):
            chunks.append(token)
    else:
        from chaihuo_reachy.bailian.llm_client import BailianLLMClient

        async with BailianLLMClient(cfg) as client:
            async for token in client.chat_stream(messages):
                chunks.append(token)
    script = _strip_generated_script("".join(chunks))
    if not script:
        raise RuntimeError("模型没有返回文案")
    return script


def _new_version_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _version_title(script: str, dialect: DialectVoice, duration_s: int, *, draft: bool) -> str:
    first = next((line.strip() for line in script.splitlines() if line.strip()), "")
    if first:
        short = first[:24] + ("…" if len(first) > 24 else "")
        return short
    prefix = "草稿" if draft else "播出"
    return f"{prefix} · {dialect.label} · {duration_s}s"


def _copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


class ShowLibrary:
    """Manifest + version files under data/shows/{opening,training}/."""

    def __init__(
        self,
        root: Path | None = None,
        shows: dict[str, LocalShowSpec] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else DEFAULT_SHOWS_ROOT
        self.shows = shows or LOCAL_SHOWS

    def kind_dir(self, kind: str) -> Path:
        return self.root / normalize_show_kind(kind)

    def version_dir(self, kind: str, version_id: str) -> Path:
        return self.kind_dir(kind) / "versions" / version_id

    def ensure_kind(self, kind: str) -> dict[str, Any]:
        spec = self.shows[normalize_show_kind(kind)]
        manifest = self._read_manifest(kind)
        if manifest["versions"]:
            return manifest
        return self._seed_from_spec(kind, spec)

    def list_payload(
        self,
        kind: str,
        *,
        brief: str = "",
        location_text: str = "",
        selected_id: str = "",
    ) -> dict[str, Any]:
        manifest = self.ensure_kind(kind)
        versions = self._public_versions(kind, manifest)
        selected = selected_id or versions[0]["id"] if versions else ""
        return self._envelope(
            "list",
            kind,
            manifest,
            selected_id=selected,
            brief=brief,
            location_text=location_text,
        )

    def get_payload(
        self,
        kind: str,
        version_id: str,
        *,
        location_text: str = "",
    ) -> dict[str, Any]:
        manifest = self.ensure_kind(kind)
        item = self._require_version(manifest, version_id)
        script = self._read_script(kind, version_id)
        return self._envelope(
            "get",
            kind,
            manifest,
            selected_id=version_id,
            script=script,
            brief=str(item.get("brief") or ""),
            location_text=location_text,
            extra={"id": version_id, "version": self._public_item(kind, item)},
        )

    async def generate_payload(
        self,
        cfg: Any,
        *,
        kind: str,
        brief: str,
        duration_s: int,
        dialect_id: str = "",
        location_text: str = "",
        llm_client: Any | None = None,
    ) -> dict[str, Any]:
        manifest = self.ensure_kind(kind)
        dialect = resolve_show_dialect(dialect_id, brief, location_text)
        seconds = normalize_show_duration(duration_s)
        script = await generate_show_script(
            cfg,
            kind=kind,
            brief=brief,
            duration_s=seconds,
            dialect=dialect,
            location_text=location_text,
            llm_client=llm_client,
        )
        item = self._write_version(
            kind,
            manifest,
            script=script,
            brief=brief,
            duration_s=seconds,
            dialect=dialect,
            draft=True,
        )
        return self._envelope(
            "generate",
            kind,
            manifest,
            selected_id=item["id"],
            script=script,
            brief=brief,
            location_text=location_text,
            extra={
                "id": item["id"],
                "dialect": dialect.id,
                "version": self._public_item(kind, item),
            },
        )

    async def save_tts_payload(
        self,
        cfg: Any,
        *,
        kind: str,
        script: str,
        brief: str = "",
        duration_s: int = 60,
        dialect_id: str = "",
        version_id: str = "",
        location_text: str = "",
        tts_client_cls: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        spec = self.shows[normalize_show_kind(kind)]
        manifest = self.ensure_kind(kind)
        text = script.strip()
        if not text:
            raise ValueError("口播稿为空")
        dialect = resolve_show_dialect(
            dialect_id, brief or parse_opening_script(text).dialect_label, location_text
        )
        seconds = normalize_show_duration(duration_s)
        if version_id:
            item = self._require_version(manifest, version_id)
        else:
            item = self._new_item(
                brief=brief,
                duration_s=seconds,
                dialect=dialect,
                script=text,
                draft=False,
            )
            manifest["versions"].insert(0, item)
        item["brief"] = brief
        item["duration_s"] = seconds
        item["dialect"] = dialect.id
        item["title"] = _version_title(text, dialect, seconds, draft=False)
        folder = self.version_dir(kind, item["id"])
        folder.mkdir(parents=True, exist_ok=True)
        script_path = folder / "script.txt"
        audio_path = folder / "audio.wav"
        script_path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        audio = await synthesize_show_audio(
            text, audio_path, cfg, tts_client_cls=tts_client_cls
        )
        item["duration_s"] = max(seconds, int(round(audio.duration_s)))
        manifest["active_id"] = item["id"]
        self._write_manifest(kind, manifest)
        self._apply_active(spec, script_path, audio_path)
        return self._envelope(
            "save_tts",
            kind,
            manifest,
            selected_id=item["id"],
            script=text,
            brief=brief,
            location_text=location_text,
            extra={"id": item["id"], "version": self._public_item(kind, item)},
        )

    def _apply_active(self, spec: LocalShowSpec, script_path: Path, audio_path: Path) -> None:
        _copy_file(script_path, spec.text_path)
        _copy_file(audio_path, spec.audio_path)

    def _seed_from_spec(self, kind: str, spec: LocalShowSpec) -> dict[str, Any]:
        script = (
            spec.text_path.read_text(encoding="utf-8").strip()
            if spec.text_path.is_file()
            else ""
        )
        parsed = parse_opening_script(script)
        dialect = (
            match_show_dialect(parsed.dialect_label or script)
            if script
            else DEFAULT_SHOW_DIALECT
        )
        duration_s = 60
        if spec.audio_path.is_file():
            try:
                duration_s = max(1, int(round(load_opening_audio(spec.audio_path).duration_s)))
            except Exception:
                logger.exception("无法读取现有 %s 音频时长", kind)
        item = self._new_item(
            brief="",
            duration_s=normalize_show_duration(duration_s)
            if duration_s in SHOW_DURATION_PRESETS
            else duration_s,
            dialect=dialect,
            script=script or spec.label,
            draft=False,
        )
        item["title"] = f"导入 · {spec.label}"
        folder = self.version_dir(kind, item["id"])
        folder.mkdir(parents=True, exist_ok=True)
        script_path = folder / "script.txt"
        script_path.write_text(
            (script or spec.label) + "\n", encoding="utf-8"
        )
        if spec.audio_path.is_file():
            _copy_file(spec.audio_path, folder / "audio.wav")
        manifest = {"active_id": item["id"], "versions": [item]}
        self._write_manifest(kind, manifest)
        return manifest

    def _public_versions(self, kind: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        return [self._public_item(kind, item) for item in manifest["versions"]]

    def _public_item(self, kind: str, item: dict[str, Any]) -> dict[str, Any]:
        audio_path = self.version_dir(kind, item["id"]) / "audio.wav"
        return {
            "id": item["id"],
            "title": item.get("title") or item["id"],
            "created_at": item.get("created_at") or "",
            "duration_s": int(item.get("duration_s") or 60),
            "dialect": item.get("dialect") or "mandarin",
            "brief": item.get("brief") or "",
            "has_audio": audio_path.is_file(),
        }

    def _envelope(
        self,
        action: str,
        kind: str,
        manifest: dict[str, Any],
        *,
        selected_id: str,
        script: str = "",
        brief: str = "",
        location_text: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        versions = self._public_versions(kind, manifest)
        selected = selected_id or (versions[0]["id"] if versions else "")
        dialect = resolve_show_dialect("", brief, location_text)
        payload = {
            "type": "show_content",
            "action": action,
            "kind": kind,
            "active_id": manifest.get("active_id") or "",
            "selected_id": selected,
            "script": script,
            "brief": brief,
            "versions": versions,
            "dialects": list_show_dialects(),
            "durations": list(SHOW_DURATION_PRESETS),
            "matched_dialect": dialect.id,
        }
        if extra:
            payload.update(extra)
        return payload

    def _require_version(self, manifest: dict[str, Any], version_id: str) -> dict[str, Any]:
        for item in manifest["versions"]:
            if item["id"] == version_id:
                return item
        raise ValueError(f"找不到历史版本: {version_id}")

    def _read_script(self, kind: str, version_id: str) -> str:
        path = self.version_dir(kind, version_id) / "script.txt"
        if not path.is_file():
            raise ValueError(f"找不到文案文件: {version_id}")
        return path.read_text(encoding="utf-8")

    def _new_item(
        self,
        *,
        brief: str,
        duration_s: int,
        dialect: DialectVoice,
        script: str,
        draft: bool,
    ) -> dict[str, Any]:
        return {
            "id": _new_version_id(),
            "title": _version_title(script, dialect, duration_s, draft=draft),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": duration_s,
            "dialect": dialect.id,
            "brief": brief,
        }

    def _write_version(
        self,
        kind: str,
        manifest: dict[str, Any],
        *,
        script: str,
        brief: str,
        duration_s: int,
        dialect: DialectVoice,
        draft: bool,
    ) -> dict[str, Any]:
        item = self._new_item(
            brief=brief,
            duration_s=duration_s,
            dialect=dialect,
            script=script,
            draft=draft,
        )
        folder = self.version_dir(kind, item["id"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "script.txt").write_text(
            script + ("" if script.endswith("\n") else "\n"), encoding="utf-8"
        )
        manifest["versions"].insert(0, item)
        self._write_manifest(kind, manifest)
        return item

    def _read_manifest(self, kind: str) -> dict[str, Any]:
        path = self.kind_dir(kind) / "manifest.json"
        if not path.is_file():
            return {"active_id": "", "versions": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("损坏的文案 manifest，将重新导入: %s", path)
            return {"active_id": "", "versions": []}
        versions = data.get("versions")
        if not isinstance(versions, list):
            versions = []
        return {
            "active_id": str(data.get("active_id") or ""),
            "versions": [item for item in versions if isinstance(item, dict) and item.get("id")],
        }

    def _write_manifest(self, kind: str, manifest: dict[str, Any]) -> None:
        path = self.kind_dir(kind) / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


async def handle_show_content_action(
    library: ShowLibrary,
    cfg: Any,
    data: dict[str, Any],
    *,
    location_text: str = "",
    llm_client: Any | None = None,
    tts_client_cls: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    action = str(data.get("action") or "list").strip()
    if action not in SHOW_CONTENT_ACTIONS:
        raise ValueError("show_content action 必须是 list、get、generate 或 save_tts")
    kind = normalize_show_kind(str(data.get("kind") or "opening"))
    brief = str(data.get("brief") or "")
    if action == "list":
        return library.list_payload(
            kind,
            brief=brief,
            location_text=location_text,
            selected_id=str(data.get("id") or ""),
        )
    if action == "get":
        version_id = str(data.get("id") or "")
        if not version_id:
            listed = library.list_payload(kind, brief=brief, location_text=location_text)
            version_id = str(listed.get("selected_id") or "")
        if not version_id:
            raise ValueError("还没有可编辑的文案版本")
        return library.get_payload(kind, version_id, location_text=location_text)
    if action == "generate":
        return await library.generate_payload(
            cfg,
            kind=kind,
            brief=brief,
            duration_s=data.get("duration_s") or 60,
            dialect_id=str(data.get("dialect") or ""),
            location_text=location_text,
            llm_client=llm_client,
        )
    return await library.save_tts_payload(
        cfg,
        kind=kind,
        script=str(data.get("script") or ""),
        brief=brief,
        duration_s=data.get("duration_s") or 60,
        dialect_id=str(data.get("dialect") or ""),
        version_id=str(data.get("id") or ""),
        location_text=location_text,
        tts_client_cls=tts_client_cls,
    )
