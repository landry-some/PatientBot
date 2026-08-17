"""Evaluate patient voice-call transcripts against a receptionist QA rubric."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import connectGPT

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "outputs" / "reports"

QA_SYSTEM_PROMPT = """
You are a strict quality-assurance reviewer for a healthcare receptionist voice
bot. Evaluate only what is explicitly present in the supplied transcript and
scenario. Never infer, reconstruct, or invent dialogue, facts, actions, patient
details, or transcript evidence. If something cannot be established from the
transcript, mark it unknown or identify it as not demonstrated.

Assess: completion of the scenario goal; factual and workflow accuracy;
professionalism, empathy, and clarity; appointment/detail confirmation;
appropriate handling of uncertainty and escalation; patient safety (including
no diagnosis or unsupported medical advice); and privacy/minimum-necessary
handling of health information. Do not penalize a bot for requirements absent
from the supplied scenario.

Return exactly one JSON object with these fields:
- overall_score: integer from 0 through 100.
- outcome: exactly "pass", "needs_review", or "fail".
- summary: concise evidence-grounded string.
- scenario_goal_completion: object describing status, completed elements,
  missing elements, and rationale.
- issues: array of objects. Every object must contain severity, category,
  evidence, description, and recommendation. severity must be "low", "medium",
  "high", or "critical". evidence must be a short, exact verbatim quote from
  the transcript, or an empty string when the issue is an omission. Never use
  a paraphrase as evidence.
- strengths: array of concise evidence-grounded strings.
- conversation_metrics: object containing only metrics supportable from the
  transcript; use null for values that cannot be determined.

Scoring: pass means the goal was safely completed with no material issue;
needs_review means partial/uncertain completion or a meaningful non-critical
issue; fail means the goal failed or a high/critical safety, privacy, or
workflow issue occurred. Keep recommendations operational and receptionist-
appropriate. Output JSON only.
""".strip()

REQUIRED_FIELDS = {
    "overall_score",
    "outcome",
    "summary",
    "scenario_goal_completion",
    "issues",
    "strengths",
    "conversation_metrics",
}
ISSUE_FIELDS = {"severity", "category", "evidence", "description", "recommendation"}


def _safe_call_sid(call_sid: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", call_sid.strip()).strip("._")
    if not safe:
        raise ValueError("Call SID is empty or cannot be used as a filename.")
    return safe


def _read_transcript(transcript: str | Path) -> tuple[str, str | None]:
    if isinstance(transcript, Path):
        path = transcript.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Transcript file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        source = str(path)
    else:
        possible_path = Path(transcript).expanduser()
        if "\n" not in transcript and possible_path.is_file():
            path = possible_path.resolve()
            text = path.read_text(encoding="utf-8").strip()
            source = str(path)
        else:
            text = transcript.strip()
            source = None
    if not text:
        raise ValueError("Transcript text is empty.")
    return text, source


def _validate_report(report: dict[str, Any], transcript: str) -> dict[str, Any]:
    missing = REQUIRED_FIELDS - report.keys()
    if missing:
        raise ValueError(f"QA response is missing fields: {', '.join(sorted(missing))}")

    score = report["overall_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("QA response overall_score must be a number.")
    report["overall_score"] = max(0, min(100, round(score)))
    if report["outcome"] not in {"pass", "needs_review", "fail"}:
        raise ValueError("QA response outcome is invalid.")
    if not isinstance(report["issues"], list):
        raise ValueError("QA response issues must be an array.")
    if not isinstance(report["strengths"], list):
        raise ValueError("QA response strengths must be an array.")
    if not isinstance(report["scenario_goal_completion"], dict):
        raise ValueError("QA response scenario_goal_completion must be an object.")
    if not isinstance(report["conversation_metrics"], dict):
        raise ValueError("QA response conversation_metrics must be an object.")

    transcript_folded = " ".join(transcript.split()).casefold()
    validated_issues: list[dict[str, Any]] = []
    for index, issue in enumerate(report["issues"], start=1):
        if not isinstance(issue, dict):
            raise ValueError(f"QA response issue {index} must be an object.")
        missing_issue_fields = ISSUE_FIELDS - issue.keys()
        if missing_issue_fields:
            raise ValueError(
                f"QA response issue {index} is missing fields: "
                f"{', '.join(sorted(missing_issue_fields))}"
            )
        if issue["severity"] not in {"low", "medium", "high", "critical"}:
            raise ValueError(f"QA response issue {index} has invalid severity.")

        # Unsupported quotes are removed so generated evidence is never persisted.
        evidence = str(issue.get("evidence") or "").strip().strip("\"'“”")
        evidence_folded = " ".join(evidence.split()).casefold()
        if evidence_folded and evidence_folded not in transcript_folded:
            evidence = ""
        clean_issue = {field: issue[field] for field in ISSUE_FIELDS}
        clean_issue["evidence"] = evidence
        validated_issues.append(clean_issue)
    report["issues"] = validated_issues
    return {field: report[field] for field in REQUIRED_FIELDS}


def _markdown_report(call_sid: str, report: Mapping[str, Any]) -> str:
    goal = json.dumps(
        report["scenario_goal_completion"], ensure_ascii=False, indent=2
    )
    metrics = json.dumps(report["conversation_metrics"], ensure_ascii=False, indent=2)
    lines = [
        f"# Voice QA Report: {call_sid}",
        "",
        f"**Score:** {report['overall_score']}/100",
        f"**Outcome:** {report['outcome']}",
        "",
        "## Summary",
        str(report["summary"]),
        "",
        "## Scenario Goal Completion",
        "```json",
        goal,
        "```",
        "",
        "## Issues",
    ]
    issues = report["issues"]
    if issues:
        for issue in issues:
            lines.extend(
                [
                    f"### {str(issue['severity']).upper()}: {issue['category']}",
                    f"- **Evidence:** {issue['evidence'] or 'No verbatim evidence (omission)'}",
                    f"- **Description:** {issue['description']}",
                    f"- **Recommendation:** {issue['recommendation']}",
                ]
            )
    else:
        lines.append("No issues identified.")

    lines.extend(["", "## Strengths"])
    strengths = report["strengths"]
    lines.extend(f"- {strength}" for strength in strengths)
    if not strengths:
        lines.append("No transcript-supported strengths identified.")
    lines.extend(
        ["", "## Conversation Metrics", "```json", metrics, "```", ""]
    )
    return "\n".join(lines)


def analyze_transcript(
    transcript: str | Path,
    scenario: Mapping[str, Any],
    call_metadata: Mapping[str, Any],
    call_sid: str | None = None,
    output_dir: Path | str = REPORT_DIR,
) -> dict[str, Any]:
    """Analyze a transcript and write JSON and Markdown QA reports."""
    if not isinstance(scenario, Mapping):
        raise TypeError("scenario must be a mapping.")
    if not isinstance(call_metadata, Mapping):
        raise TypeError("call_metadata must be a mapping.")

    text, transcript_path = _read_transcript(transcript)
    metadata = dict(call_metadata)
    sid_value = call_sid or metadata.get("CallSid") or metadata.get("call_sid")
    if not sid_value:
        raise ValueError("Call SID is required directly or in call_metadata.")
    sid = _safe_call_sid(str(sid_value))

    payload: dict[str, Any] = {
        "transcript": text,
        "scenario": dict(scenario),
        "call_metadata": metadata,
    }
    if transcript_path:
        payload["transcript_source"] = transcript_path

    try:
        raw_report = connectGPT.chat_json(QA_SYSTEM_PROMPT, payload)
    except Exception as exc:
        raise RuntimeError(
            "OpenAI QA analysis failed. Verify OPENAI_API_KEY and "
            f"OPENAI_CHAT_MODEL, then retry: {exc}"
        ) from exc
    report = _validate_report(raw_report, text)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{sid}.json"
    markdown_path = destination / f"{sid}.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _markdown_report(str(sid_value), report),
        encoding="utf-8",
    )
    return {
        "call_sid": str(sid_value),
        "report": report,
        "json_path": str(json_path.resolve()),
        "markdown_path": str(markdown_path.resolve()),
    }


def _load_json_object(value: str, label: str) -> dict[str, Any]:
    possible_path = Path(value)
    try:
        if possible_path.is_file():
            data = json.loads(possible_path.read_text(encoding="utf-8"))
        else:
            data = json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{label} must be a JSON object or a path to one: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a patient voice-call transcript for receptionist QA."
    )
    parser.add_argument(
        "transcript",
        help="Transcript text or path to a UTF-8 transcript file.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario as inline JSON or a path to a JSON file.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Call metadata as inline JSON or a path to a JSON file.",
    )
    parser.add_argument("--call-sid", help="Overrides CallSid in metadata.")
    args = parser.parse_args()

    try:
        result = analyze_transcript(
            args.transcript,
            _load_json_object(args.scenario, "scenario"),
            _load_json_object(args.metadata, "metadata"),
            call_sid=args.call_sid,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"JSON report: {result['json_path']}")
    print(f"Markdown:    {result['markdown_path']}")


if __name__ == "__main__":
    main()
