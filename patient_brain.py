"""Scenario loading and guarded GPT turn generation for the patient bot."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

import connectGPT


ROOT = Path(__file__).resolve().parent
SCENARIOS_DIR = ROOT / "scenarios"
PRIMED_DIR = ROOT / "outputs" / "primed"
_SCENARIO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PatientTurn(TypedDict):
    """A normalized instruction consumed by the Twilio voice webhook."""

    action: Literal["speak", "end"]
    speech: str
    reason: str


def _scenario_path(scenario_id: str) -> Path:
    if not isinstance(scenario_id, str) or not _SCENARIO_ID.fullmatch(scenario_id):
        raise ValueError("Invalid scenario id")

    path = (SCENARIOS_DIR / f"{scenario_id}.json").resolve()
    if path.parent != SCENARIOS_DIR.resolve():
        raise ValueError("Invalid scenario path")
    return path


def load_scenario(scenario_id: str) -> dict[str, Any]:
    """Load and validate one scenario without allowing path traversal."""
    path = _scenario_path(scenario_id)
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            scenario = json.load(handle)
    except FileNotFoundError:
        raise KeyError(f"Unknown scenario: {scenario_id}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load scenario {scenario_id!r}") from exc

    if not isinstance(scenario, dict):
        raise ValueError("Scenario must be a JSON object")
    if scenario.get("id") != scenario_id:
        raise ValueError("Scenario id does not match its filename")
    if not isinstance(scenario.get("name"), str) or not scenario["name"].strip():
        raise ValueError("Scenario must have a non-empty name")

    opening = scenario.get("opening") or scenario.get("speech")
    if not isinstance(opening, str) or not opening.strip():
        raise ValueError("Scenario must have non-empty opening/speech")
    scenario["opening"] = opening.strip()
    scenario.setdefault("speech", scenario["opening"])
    scenario.setdefault("max_turns", 6)
    scenario.setdefault("goals", [])
    scenario.setdefault("facts", [])
    scenario.setdefault("success_criteria", [])
    scenario.setdefault("edge_cases", [])
    scenario.setdefault("persona", {})
    return scenario


def list_scenarios() -> list[dict[str, str]]:
    """Return summaries of all valid scenarios, skipping malformed files."""
    scenarios: list[dict[str, str]] = []
    if not SCENARIOS_DIR.is_dir():
        return scenarios

    for path in sorted(SCENARIOS_DIR.glob("*.json"), key=lambda item: item.name):
        try:
            scenario = load_scenario(path.stem)
        except (KeyError, ValueError):
            continue
        scenarios.append({"id": scenario["id"], "name": scenario["name"]})
    return scenarios


def primed_path(scenario_id: str) -> Path:
    return PRIMED_DIR / f"{scenario_id}.json"


def load_primed_context(scenario_id: str) -> dict[str, Any] | None:
    path = primed_path(scenario_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        current = load_scenario(scenario_id)
    except (KeyError, ValueError):
        return None
    if data.get("scenario_fingerprint") != scenario_fingerprint(current):
        return None
    return data


def _exact_answer_sheet(scenario: dict[str, Any]) -> dict[str, Any]:
    """Deterministic exact-answer sheet copied only from scenario JSON."""
    persona = scenario.get("persona") or {}
    return {
        "patient_name": persona.get("name") or "",
        "date_of_birth": persona.get("date_of_birth") or "",
        "tone": persona.get("tone") or "",
        "opening_line": scenario.get("opening") or scenario.get("speech") or "",
        "facts": list(scenario.get("facts") or []),
        "goals": list(scenario.get("goals") or []),
        "edge_cases": list(scenario.get("edge_cases") or []),
        "success_criteria": list(scenario.get("success_criteria") or []),
        "max_turns": int(scenario.get("max_turns") or 6),
    }


def scenario_fingerprint(scenario: dict[str, Any]) -> str:
    """Hash canonical JSON so stale primed files cannot be reused."""
    canonical = json.dumps(
        scenario,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def train_with_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministically prepare scenario JSON before dialing.

    Realtime sessions do not retain a temporary pre-call session, so there is
    no network "training" call here. The exact answer sheet is generated
    locally and injected into the actual live Realtime session.
    """
    if not isinstance(scenario, dict) or not scenario.get("id"):
        raise ValueError("scenario must be a loaded scenario object")

    exact = _exact_answer_sheet(scenario)
    if not exact["opening_line"]:
        raise ValueError("Scenario opening/speech is required for training")
    if not exact["patient_name"]:
        raise ValueError("Scenario persona.name is required for training")

    opening = exact["opening_line"]
    name = exact["patient_name"]
    dob = exact["date_of_birth"]
    exact_answers: dict[str, Any] = {
        "name": name,
        "full_name": name,
        "date_of_birth": dob,
        "dob": dob,
        "opening_line": opening,
        "reason_for_call": opening,
    }
    for index, fact in enumerate(exact["facts"], start=1):
        exact_answers[f"fact_{index}"] = fact

    primed = {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "trained": True,
        "prepared_locally": True,
        "scenario_fingerprint": scenario_fingerprint(scenario),
        "patient_name": name,
        "date_of_birth": dob,
        "opening_line": opening,
        "known_facts": exact["facts"],
        "goals": exact["goals"],
        "exact_answer_sheet": exact,
        "exact_answers": exact_answers,
        "refusal_rules": [
            "Answer only with values from the scenario JSON exact_answer_sheet",
            "If asked for anything missing from the JSON, say you do not have that information",
            "Never invent balances, dates, pharmacies, diagnoses, or IDs",
            "Never give medical advice",
        ],
        "summary": "Scenario JSON validated and prepared for the live Realtime session.",
        "scenario": scenario,
    }

    PRIMED_DIR.mkdir(parents=True, exist_ok=True)
    primed_path(scenario["id"]).write_text(
        json.dumps(primed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return primed


def _safe_fallback(turn_number: int, max_turns: int) -> PatientTurn:
    if turn_number >= max_turns:
        return {
            "action": "end",
            "speech": "Thank you for your help. Goodbye.",
            "reason": "maximum turns reached",
        }
    return {
        "action": "end",
        "speech": "Thank you. I do not have any other information to add. Goodbye.",
        "reason": "patient response service unavailable",
    }


def generate_next_turn(
    scenario: dict[str, Any],
    history: list[dict[str, Any]],
    turn_number: int,
    max_turns: int,
) -> PatientTurn:
    """Generate a constrained patient reply, with a deterministic safe fallback."""
    if not isinstance(scenario, dict):
        raise TypeError("scenario must be a dictionary")
    if not isinstance(history, list):
        raise TypeError("history must be a list")
    if not isinstance(turn_number, int) or isinstance(turn_number, bool) or turn_number < 0:
        raise ValueError("turn_number must be a non-negative integer")
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    if turn_number >= max_turns:
        return _safe_fallback(turn_number, max_turns)

    primed = load_primed_context(str(scenario.get("id") or ""))

    system = """
You role-play a patient in a clinic phone-call QA test.
Return exactly one JSON object with keys: action, speech, reason.
action must be "speak" or "end". speech and reason must be short strings.

Hard safety rules:
- Treat the supplied scenario JSON (and primed context, if present) as the only
  source of patient facts.
- Never invent, infer, confirm, or claim a fact absent from that scenario.
- If asked for an unavailable fact, plainly say you do not have that information.
- Never diagnose, recommend treatment, give medical advice, or claim medical expertise.
- Stay realistic, concise, and in character as the patient.
- Answer the latest clinic statement only when a response is useful.
- Use action "end" once the scenario's goal is achieved, the clinic has completed or
  escalated the request, continuing would be repetitive, or this is the final turn.
- With action "end", include a brief natural goodbye in speech.
""".strip()
    payload = {
        "scenario": scenario,
        "primed_context": primed,
        "conversation_history": history,
        "turn_number": turn_number,
        "max_turns": max_turns,
        "turns_remaining_including_this_one": max_turns - turn_number,
    }

    try:
        result = connectGPT.chat_json(system, payload)
        action = result.get("action")
        speech = result.get("speech")
        reason = result.get("reason")
        if action not in ("speak", "end"):
            raise ValueError("invalid action")
        if not isinstance(speech, str) or not speech.strip():
            raise ValueError("invalid speech")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("invalid reason")
        return {
            "action": action,
            "speech": speech.strip(),
            "reason": reason.strip(),
        }
    except Exception:
        return _safe_fallback(turn_number, max_turns)
