"""Start a Twilio patient-simulation call."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv
from twilio.rest import Client

import patient_brain

load_dotenv()

ROOT = Path(__file__).resolve().parent


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        print(f"Missing {name} in .env")
        sys.exit(1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an automated patient scenario.")
    parser.add_argument(
        "-s",
        "--scenario",
        default="scheduling",
        help="Scenario id from scenarios/ (default: scheduling)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without placing a call",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Prepare exact JSON context and exit (no API call or phone call)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse an existing prepared context if the JSON is unchanged",
    )
    args = parser.parse_args()

    if args.list:
        for item in patient_brain.list_scenarios():
            print(f"{item['id']:12}  {item['name']}")
        return

    scenario = patient_brain.load_scenario(args.scenario)

    account_sid = require_env("TWILIO_ACCOUNT_SID")
    auth_token = require_env("TWILIO_AUTH_TOKEN")
    from_number = require_env("TWILIO_FROM_NUMBER")
    to_number = require_env("TWILIO_TO_NUMBER")
    public_base_url = require_env("PUBLIC_BASE_URL").rstrip("/")

    print(f"Scenario: {scenario['id']} - {scenario['name']}")
    print(f"Destination: {to_number}")

    if not args.skip_train:
        print("Preparing exact scenario JSON locally before dial...")
        try:
            primed = patient_brain.train_with_scenario(scenario)
        except Exception as exc:
            print(f"Scenario preparation failed: {exc}")
            sys.exit(1)
        print(f"Summary: {primed.get('summary')}")
        print(f"Patient: {primed.get('patient_name')}")
        print(f"DOB: {primed.get('date_of_birth')}")
        print(f"Opening: {primed.get('opening_line')}")
        print(f"Exact facts ({len(primed.get('known_facts') or [])}):")
        for fact in primed.get("known_facts") or []:
            print(f"  - {fact}")
        print(f"Primed file: outputs\\primed\\{scenario['id']}.json")
    else:
        primed = patient_brain.load_primed_context(scenario["id"])
        if not primed or not primed.get("trained"):
            print(
                "Refusing to reuse missing/stale prepared JSON. "
                "Run without --skip-train to regenerate it."
            )
            sys.exit(1)
        print("Using existing validated JSON context (--skip-train).")

    if args.train_only:
        print("Prepare-only mode complete. No OpenAI request or call was made.")
        return

    require_env("OPENAI_API_KEY")

    query = urlencode({"scenario": scenario["id"]})
    voice_url = f"{public_base_url}/voice?{query}"
    recording_callback = f"{public_base_url}/recording"
    status_callback = f"{public_base_url}/status"
    recording_channels = os.getenv("RECORDING_CHANNELS", "dual").strip().lower()
    if recording_channels not in {"mono", "dual"}:
        recording_channels = "mono"

    print(f"Voice URL: {voice_url}")
    print(f"Realtime model: {os.getenv('OPENAI_REALTIME_MODEL', 'gpt-realtime-2.1-mini')}")
    print(f"Recording: channels={recording_channels}")

    if args.dry_run:
        print("Dry run passed. No call was placed.")
        return

    client = Client(account_sid, auth_token)
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=voice_url,
        method="POST",
        record=True,
        recording_channels=recording_channels,
        recording_status_callback=recording_callback,
        recording_status_callback_method="POST",
        recording_status_callback_event=["completed"],
        status_callback=status_callback,
        status_callback_method="POST",
        status_callback_event=["completed"],
        time_limit=int(os.getenv("MAX_CALL_SECONDS", "240")),
    )

    print(f"Calling {to_number}")
    print(f"Recording callback: {recording_callback}")
    print(f"Status callback: {status_callback}")
    print(f"Call SID: {call.sid}")
    print(f"Status: {call.status}")
    print("Keep app.py (and ngrok) running until AFTER the call ends.")
    print(f"Recording will be saved to: recordings\\{call.sid}.mp3")


if __name__ == "__main__":
    main()
