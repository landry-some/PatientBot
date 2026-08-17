"""Download Twilio call recordings as MP3 (fallback if the webhook was missed)."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from twilio.rest import Client

from app import download_recording_mp3

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Twilio recordings as MP3.")
    parser.add_argument(
        "-c",
        "--call-sid",
        help="Download the recording for this call SID (default: most recent recording)",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=1,
        help="How many recent recordings to download when no call SID is given",
    )
    args = parser.parse_args()

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    if not account_sid or not auth_token:
        print("Missing TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN in .env")
        sys.exit(1)

    client = Client(account_sid, auth_token)

    if args.call_sid:
        recordings = client.recordings.list(call_sid=args.call_sid, limit=args.limit)
    else:
        recordings = client.recordings.list(limit=args.limit)

    if not recordings:
        print("No recordings found. The call may still be processing.")
        sys.exit(1)

    for rec in recordings:
        print(
            f"Recording {rec.sid} | call {rec.call_sid} | "
            f"{rec.duration}s | channels={rec.channels} | status={rec.status}"
        )
        if rec.status != "completed":
            print("  Skipped — not completed yet.")
            continue
        url = f"https://api.twilio.com{rec.uri.replace('.json', '')}"
        download_recording_mp3(rec.sid, url, rec.call_sid, rec.channels)


if __name__ == "__main__":
    main()
