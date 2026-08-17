"""Thread-safe in-memory call sessions with per-call JSON persistence."""

from __future__ import annotations

import copy
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CALLS_DIR = ROOT / "outputs" / "calls"
_CALL_SID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_PATH_FIELDS = {"recording_path", "transcript_path", "report_path"}
_UPDATABLE_FIELDS = {
    "scenario_id",
    "scenario_name",
    "turn_count",
    "status",
    "end_reason",
    "controller_state",
    *_PATH_FIELDS,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Own call state and serialize every mutation while holding one lock."""

    def __init__(self, output_dir: str | Path = CALLS_DIR) -> None:
        self.output_dir = Path(output_dir)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_call_sid(call_sid: str) -> str:
        if not isinstance(call_sid, str) or not _CALL_SID.fullmatch(call_sid):
            raise ValueError("Invalid CallSid")
        return call_sid

    def _persist(self, session: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{session['call_sid']}.json"
        # Direct write is safe here because all mutations hold self._lock.
        # Avoid os.replace temp-file races that Windows antivirus often locks.
        destination.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def create(
        self,
        call_sid: str,
        scenario_id: str,
        scenario_name: str,
    ) -> dict[str, Any]:
        """Create an active session; repeated creates return the existing one."""
        call_sid = self._validate_call_sid(call_sid)
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        if not isinstance(scenario_name, str) or not scenario_name.strip():
            raise ValueError("scenario_name must be a non-empty string")

        with self._lock:
            existing = self._sessions.get(call_sid)
            if existing is not None:
                return copy.deepcopy(existing)

            timestamp = _now()
            session: dict[str, Any] = {
                "call_sid": call_sid,
                "scenario_id": scenario_id.strip(),
                "scenario_name": scenario_name.strip(),
                "started_at": timestamp,
                "updated_at": timestamp,
                "history": [],
                "turn_count": 0,
                "status": "active",
                "end_reason": None,
                "controller_state": None,
                "recording_path": None,
                "transcript_path": None,
                "report_path": None,
            }
            self._persist(session)
            self._sessions[call_sid] = session
            return copy.deepcopy(session)

    def get(self, call_sid: str) -> dict[str, Any] | None:
        """Return a defensive copy of a session, if present."""
        call_sid = self._validate_call_sid(call_sid)
        with self._lock:
            session = self._sessions.get(call_sid)
            return copy.deepcopy(session) if session is not None else None

    def _require(self, call_sid: str) -> dict[str, Any]:
        call_sid = self._validate_call_sid(call_sid)
        try:
            return self._sessions[call_sid]
        except KeyError:
            raise KeyError(f"Unknown CallSid: {call_sid}") from None

    def add_turn(
        self,
        call_sid: str,
        speaker: str,
        text: str,
        confidence: float | str | None = None,
    ) -> dict[str, Any]:
        """Append one history entry and increment the session turn count."""
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("speaker must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if confidence is not None and not isinstance(confidence, (int, float, str)):
            raise TypeError("confidence must be a number, string, or None")

        with self._lock:
            session = self._require(call_sid)
            session["history"].append(
                {
                    "speaker": speaker.strip(),
                    "text": text.strip(),
                    "confidence": confidence,
                    "timestamp": _now(),
                }
            )
            session["turn_count"] += 1
            session["updated_at"] = _now()
            self._persist(session)
            return copy.deepcopy(session)

    def update(self, call_sid: str, **changes: Any) -> dict[str, Any]:
        """Update supported session metadata and persist it."""
        unknown = set(changes) - _UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Unsupported session fields: {', '.join(sorted(unknown))}")
        if "turn_count" in changes and (
            not isinstance(changes["turn_count"], int)
            or isinstance(changes["turn_count"], bool)
            or changes["turn_count"] < 0
        ):
            raise ValueError("turn_count must be a non-negative integer")
        for field in ("scenario_id", "scenario_name", "status", "end_reason"):
            if field in changes and (
                changes[field] is not None
                and (
                    not isinstance(changes[field], str)
                    or not changes[field].strip()
                )
            ):
                raise ValueError(f"{field} must be a non-empty string")
        if "controller_state" in changes and not isinstance(
            changes["controller_state"],
            (dict, type(None)),
        ):
            raise ValueError("controller_state must be a dictionary or None")
        for field in _PATH_FIELDS:
            if field in changes and changes[field] is not None:
                changes[field] = str(changes[field])

        with self._lock:
            session = self._require(call_sid)
            for field, value in changes.items():
                session[field] = value.strip() if isinstance(value, str) else value
            session["updated_at"] = _now()
            self._persist(session)
            return copy.deepcopy(session)

    def finish(
        self,
        call_sid: str,
        status: str = "completed",
        *,
        recording_path: str | Path | None = None,
        transcript_path: str | Path | None = None,
        report_path: str | Path | None = None,
        end_reason: str | None = None,
    ) -> dict[str, Any]:
        """Mark a call finished and optionally attach generated artifact paths."""
        changes: dict[str, Any] = {"status": status}
        supplied_paths = {
            "recording_path": recording_path,
            "transcript_path": transcript_path,
            "report_path": report_path,
        }
        changes.update(
            {field: value for field, value in supplied_paths.items() if value is not None}
        )
        if end_reason is not None:
            changes["end_reason"] = end_reason
        return self.update(call_sid, **changes)


_store = SessionStore()


def create(call_sid: str, scenario_id: str, scenario_name: str) -> dict[str, Any]:
    return _store.create(call_sid, scenario_id, scenario_name)


def get(call_sid: str) -> dict[str, Any] | None:
    return _store.get(call_sid)


def add_turn(
    call_sid: str,
    speaker: str,
    text: str,
    confidence: float | str | None = None,
) -> dict[str, Any]:
    return _store.add_turn(call_sid, speaker, text, confidence)


def update(call_sid: str, **changes: Any) -> dict[str, Any]:
    return _store.update(call_sid, **changes)


def finish(
    call_sid: str,
    status: str = "completed",
    *,
    recording_path: str | Path | None = None,
    transcript_path: str | Path | None = None,
    report_path: str | Path | None = None,
    end_reason: str | None = None,
) -> dict[str, Any]:
    return _store.finish(
        call_sid,
        status,
        recording_path=recording_path,
        transcript_path=transcript_path,
        report_path=report_path,
        end_reason=end_reason,
    )
