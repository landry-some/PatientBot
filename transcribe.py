"""Transcribe call recordings and persist transcript artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
TRANSCRIPT_DIR = ROOT / "outputs" / "transcripts"


def _safe_call_sid(call_sid: str | None, recording_path: Path) -> str:
    candidate = (call_sid or recording_path.stem).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
    if not safe:
        raise ValueError("Call SID is empty or cannot be used as a filename.")
    return safe


def _load_json_object(value: str | None, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    possible_path = Path(value)
    try:
        if possible_path.is_file():
            data = json.loads(possible_path.read_text(encoding="utf-8"))
        else:
            data = json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{label} must be a JSON object or a path to a JSON object: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return data


def transcribe_recording(
    recording_path: Path | str,
    call_sid: str | None = None,
    scenario: Mapping[str, Any] | None = None,
    call_metadata: Mapping[str, Any] | None = None,
    output_dir: Path | str = TRANSCRIPT_DIR,
) -> dict[str, Any]:
    """Transcribe one recording and write matching text and JSON artifacts."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the environment or .env."
        )

    recording = Path(recording_path).expanduser().resolve()
    if not recording.is_file():
        raise FileNotFoundError(f"Recording file not found: {recording}")

    model = os.getenv("WHISPER_MODEL", "whisper-1").strip() or "whisper-1"
    sid = _safe_call_sid(call_sid, recording)

    try:
        with recording.open("rb") as audio:
            response = OpenAI(api_key=api_key).audio.transcriptions.create(
                model=model,
                file=audio,
                response_format="json",
            )
    except Exception as exc:
        raise RuntimeError(f"OpenAI transcription failed for {recording}: {exc}") from exc

    if isinstance(response, str):
        text = response.strip()
    else:
        text = str(getattr(response, "text", "") or "").strip()
    if not text:
        raise RuntimeError("OpenAI transcription returned no transcript text.")

    artifact: dict[str, Any] = {
        "call_sid": call_sid or sid,
        "recording_path": str(recording),
        "model": model,
        "transcript": text,
        "transcribed_at": datetime.now(timezone.utc).isoformat(),
    }
    if scenario is not None:
        artifact["scenario"] = dict(scenario)
    if call_metadata is not None:
        artifact["call_metadata"] = dict(call_metadata)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    text_path = destination / f"{sid}.txt"
    json_path = destination / f"{sid}.json"
    text_path.write_text(text + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    artifact["text_path"] = str(text_path.resolve())
    artifact["json_path"] = str(json_path.resolve())
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a patient-call recording with OpenAI."
    )
    parser.add_argument("recording", type=Path, help="Path to the audio recording.")
    parser.add_argument("--call-sid", help="Call SID used for output filenames.")
    parser.add_argument(
        "--scenario",
        help="Scenario as inline JSON or a path to a JSON file.",
    )
    parser.add_argument(
        "--metadata",
        help="Call metadata as inline JSON or a path to a JSON file.",
    )
    args = parser.parse_args()

    try:
        result = transcribe_recording(
            args.recording,
            call_sid=args.call_sid,
            scenario=_load_json_object(args.scenario, "scenario"),
            call_metadata=_load_json_object(args.metadata, "metadata"),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Transcript: {result['text_path']}")
    print(f"Metadata:   {result['json_path']}")


if __name__ == "__main__":
    main()
