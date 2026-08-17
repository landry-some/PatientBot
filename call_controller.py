"""Deterministic per-call guardrails around the generative voice model."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


GOODBYE_MARKERS = (
    "goodbye",
    "good bye",
    "bye now",
    "thank you, bye",
    "thanks, bye",
)
RISK_MARKERS = {
    "possible_medical_advice": ("you should take", "increase your dose", "diagnosis is"),
    "unsafe_guarantee": ("guaranteed refill", "definitely approved"),
    "sensitive_payment_request": ("credit card number", "security code", "cvv"),
}


@dataclass
class CallState:
    call_sid: str
    scenario_id: str
    max_patient_turns: int
    max_seconds: int
    max_idle_seconds: int
    completion_signals: list[str] = field(default_factory=list)
    scenario_facts: list[str] = field(default_factory=list)
    started_monotonic: float = field(default_factory=time.monotonic)
    last_activity_monotonic: float = field(default_factory=time.monotonic)
    clinic_turns: int = 0
    patient_turns: int = 0
    greeting_seen: bool = False
    opening_sent: bool = False
    goal_signal_seen: bool = False
    goals_completed: bool = False
    facts_disclosed: list[int] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    ended: bool = False
    end_reason: str | None = None

    def elapsed_seconds(self) -> int:
        return max(0, int(time.monotonic() - self.started_monotonic))

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("started_monotonic", None)
        data.pop("last_activity_monotonic", None)
        data["elapsed_seconds"] = self.elapsed_seconds()
        data["idle_seconds"] = max(
            0,
            int(time.monotonic() - self.last_activity_monotonic),
        )
        return data


_states: dict[str, CallState] = {}
_lock = threading.RLock()


def create(
    call_sid: str,
    scenario: dict[str, Any],
    max_seconds: int,
    max_idle_seconds: int,
) -> CallState:
    with _lock:
        existing = _states.get(call_sid)
        if existing:
            return existing
        state = CallState(
            call_sid=call_sid,
            scenario_id=str(scenario["id"]),
            max_patient_turns=int(scenario.get("max_turns") or 6),
            max_seconds=max_seconds,
            max_idle_seconds=max_idle_seconds,
            completion_signals=[
                str(value).lower()
                for value in scenario.get("completion_signals", [])
                if str(value).strip()
            ],
            scenario_facts=[str(value) for value in scenario.get("facts", [])],
        )
        _states[call_sid] = state
        return state


def get(call_sid: str) -> CallState | None:
    with _lock:
        return _states.get(call_sid)


def touch_activity(call_sid: str) -> None:
    with _lock:
        state = _states.get(call_sid)
        if state and not state.ended:
            state.last_activity_monotonic = time.monotonic()


def on_clinic_turn(call_sid: str, text: str) -> str | None:
    with _lock:
        state = _states.get(call_sid)
        if not state or state.ended:
            return state.end_reason if state else None
        state.clinic_turns += 1
        state.last_activity_monotonic = time.monotonic()
        if state.clinic_turns == 1:
            state.greeting_seen = True
        lower = text.lower()
        if state.completion_signals and any(
            signal in lower for signal in state.completion_signals
        ):
            state.goal_signal_seen = True
            state.goals_completed = True
        for flag, markers in RISK_MARKERS.items():
            if flag not in state.quality_flags and any(marker in lower for marker in markers):
                state.quality_flags.append(flag)
        return _limit_reason(state)


def on_patient_turn(call_sid: str, text: str) -> str | None:
    with _lock:
        state = _states.get(call_sid)
        if not state or state.ended:
            return state.end_reason if state else None
        state.patient_turns += 1
        state.last_activity_monotonic = time.monotonic()
        state.opening_sent = True
        lower = text.lower()
        patient_words = {word.strip(".,!?;:()") for word in lower.split()}
        for index, fact in enumerate(state.scenario_facts):
            if index in state.facts_disclosed:
                continue
            fact_words = {
                word.strip(".,!?;:()")
                for word in fact.lower().split()
                if len(word.strip(".,!?;:()")) >= 4
            }
            if fact_words and len(patient_words & fact_words) / len(fact_words) >= 0.5:
                state.facts_disclosed.append(index)
        if any(marker in lower for marker in GOODBYE_MARKERS):
            return mark_ended(call_sid, "patient_goodbye")
        if state.goal_signal_seen:
            return mark_ended(call_sid, "goal_completed")
        return _limit_reason(state)


def check_limits(call_sid: str) -> str | None:
    with _lock:
        state = _states.get(call_sid)
        return _limit_reason(state) if state else None


def _limit_reason(state: CallState) -> str | None:
    if state.ended:
        return state.end_reason
    if state.patient_turns >= state.max_patient_turns:
        return mark_ended(state.call_sid, "max_patient_turns")
    if state.elapsed_seconds() >= state.max_seconds:
        return mark_ended(state.call_sid, "max_call_seconds")
    if time.monotonic() - state.last_activity_monotonic >= state.max_idle_seconds:
        return mark_ended(state.call_sid, "repeated_silence")
    return None


def mark_ended(call_sid: str, reason: str) -> str:
    with _lock:
        state = _states.get(call_sid)
        if state:
            state.ended = True
            state.end_reason = reason
        return reason

