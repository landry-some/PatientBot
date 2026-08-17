"""Twilio Media Streams ↔ OpenAI Realtime bridge helpers."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def realtime_model() -> str:
    return os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1-mini").strip()


def realtime_voice() -> str:
    return os.getenv("OPENAI_REALTIME_VOICE", "alloy").strip() or "alloy"


def build_patient_instructions(scenario: dict[str, Any], primed: dict[str, Any] | None = None) -> str:
    """Build Realtime session instructions from locally prepared scenario JSON."""
    if not primed or not primed.get("trained"):
        raise RuntimeError(
            "Realtime patient requires prepared scenario JSON first. "
            "Run: python make_call.py --scenario <id> (preparation is automatic), "
            "or: python make_call.py --scenario <id> --train-only"
        )

    exact = primed.get("exact_answer_sheet") or {}
    exact_answers = primed.get("exact_answers") or {}
    name = primed.get("patient_name") or (scenario.get("persona") or {}).get("name") or "the patient"
    opening = primed.get("opening_line") or scenario.get("opening") or ""
    dob = primed.get("date_of_birth") or (scenario.get("persona") or {}).get("date_of_birth") or ""

    return f"""
You are role-playing a realistic patient on a phone call with a clinic AI receptionist.
The scenario was validated locally before this call. Give EXACT answers from it only.

EXACT IDENTITY
- Name: {name}
- Date of birth: {dob}
- Opening line (use this after their greeting): {opening}

EXACT ANSWER SHEET (source of truth):
{json.dumps(exact, ensure_ascii=False, indent=2)}

EXACT ANSWERS MAP:
{json.dumps(exact_answers, ensure_ascii=False, indent=2)}

FULL SCENARIO JSON:
{json.dumps(scenario, ensure_ascii=False, indent=2)}

Refusal rules:
{json.dumps(primed.get('refusal_rules') or [], ensure_ascii=False, indent=2)}

Conversation rules:
1. Remain silent until you hear the clinic's first spoken greeting.
2. Then say the opening line exactly or with only tiny natural spoken variation that keeps every fact identical.
3. When asked for name, DOB, balance, insurance, pharmacy, injury details, etc., answer ONLY with values from the exact answer sheet/map.
4. If the clinic asks for something not in the JSON, say you do not have that information.
5. Never invent numbers, dates, clinics, clinicians, diagnoses, or card details.
6. Never give medical advice.
7. Keep replies short (1-3 sentences).
8. When goals are met or the clinic finishes with a clear next step, thank them and say goodbye.
""".strip()


def session_update_event(instructions: str) -> dict[str, Any]:
    """OpenAI Realtime session configured for Twilio PCMU and turn transcripts."""
    transcription_model = os.getenv(
        "OPENAI_INPUT_TRANSCRIPTION_MODEL",
        "gpt-4o-mini-transcribe",
    ).strip()
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": realtime_model(),
            "output_modalities": ["audio"],
            "instructions": instructions,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "transcription": {"model": transcription_model},
                    "turn_detection": {
                        "type": "server_vad",
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": realtime_voice(),
                },
            },
        },
    }
