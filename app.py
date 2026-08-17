"""FastAPI server: Twilio Media Streams ↔ OpenAI Realtime patient bot + recording QA."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import traceback
from pathlib import Path
from urllib.parse import urlparse

import requests
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response
from twilio.twiml.voice_response import Connect, Stream, VoiceResponse

import analyze
import call_controller
import patient_brain
import realtime_bridge
import session_store
import transcribe

load_dotenv()

ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

app = FastAPI(title="Patient Realtime QA Bot")
_BACKGROUND_TASKS: set[asyncio.Task[object]] = set()
_PROCESSING_CALLS: set[str] = set()
_PROCESSED_CALLS: set[str] = set()


def _twilio_auth() -> tuple[str, str]:
    return TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN


def _public_wss_base() -> str:
    raw = PUBLIC_BASE_URL
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = parsed.netloc or raw.replace("https://", "").replace("http://", "")
    return f"wss://{host}"


def download_recording_mp3(
    recording_sid: str,
    recording_url: str,
    call_sid: str,
    channels: int | str | None = None,
) -> Path:
    """Download recording; preserve dual channels by preferring WAV first."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    auth = _twilio_auth()
    base = recording_url.rstrip("/") if recording_url else (
        f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}"
        f"/Recordings/{recording_sid}"
    )

    mp3_path = RECORDINGS_DIR / f"{call_sid}.mp3"
    wav_path = RECORDINGS_DIR / f"{call_sid}.wav"

    is_dual = str(channels or "").lower() in {"2", "dual"}
    if is_dual:
        wav_resp = requests.get(f"{base}.wav", auth=auth, timeout=180)
        wav_resp.raise_for_status()
        wav_path.write_bytes(wav_resp.content)
        print(f"Saved dual-channel WAV: {wav_path} ({len(wav_resp.content)} bytes)")
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3")
            print(f"Converted stereo recording to MP3: {mp3_path}")
            return mp3_path
        except Exception as exc:
            print(f"MP3 conversion failed ({exc}). WAV preserved at {wav_path}")
            return wav_path

    mp3_resp = requests.get(f"{base}.mp3", auth=auth, timeout=180)
    mp3_resp.raise_for_status()
    if not mp3_resp.content or mp3_resp.content.lstrip().startswith(b"<?xml"):
        raise RuntimeError("Twilio returned no MP3 audio")
    mp3_path.write_bytes(mp3_resp.content)
    print(f"Saved recording MP3: {mp3_path} ({len(mp3_resp.content)} bytes)")
    return mp3_path


def finalize_recording_for_call(call_sid: str) -> Path | None:
    existing = RECORDINGS_DIR / f"{call_sid}.mp3"
    if existing.is_file() and existing.stat().st_size > 1000:
        return existing
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return None

    from twilio.rest import Client

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    for rec in client.recordings.list(call_sid=call_sid, limit=5):
        if rec.status != "completed":
            continue
        url = f"https://api.twilio.com{rec.uri.replace('.json', '')}"
        return download_recording_mp3(rec.sid, url, call_sid, rec.channels)
    return None


def _post_call_pipeline(call_sid: str, recording_path: Path) -> None:
    session = session_store.get(call_sid)
    scenario_id = (session or {}).get("scenario_id") or "scheduling"
    try:
        scenario = patient_brain.load_scenario(scenario_id)
    except (KeyError, ValueError):
        scenario = {"id": scenario_id, "name": scenario_id}

    metadata = {
        "call_sid": call_sid,
        "scenario_id": scenario.get("id"),
        "scenario_name": scenario.get("name"),
        "history": (session or {}).get("history", []),
        "end_reason": (session or {}).get("end_reason"),
        "controller_state": (session or {}).get("controller_state"),
        "recording_path": str(recording_path),
        "realtime_model": realtime_bridge.realtime_model(),
    }

    transcript_path = None
    report_path = None
    try:
        transcript_result = transcribe.transcribe_recording(
            recording_path,
            call_sid=call_sid,
            scenario=scenario,
            call_metadata=metadata,
        )
        transcript_path = transcript_result.get("text_path")
        transcript_text = transcript_result.get("transcript") or ""
        print(f"[{call_sid}] transcript saved: {transcript_path}")
    except Exception as exc:
        print(f"[{call_sid}] transcription skipped/failed: {exc}")
        traceback.print_exc()
        session_store.update(call_sid, recording_path=str(recording_path), status="recorded")
        return

    try:
        report_result = analyze.analyze_transcript(
            transcript_text or Path(str(transcript_path)),
            scenario=scenario,
            call_metadata=metadata,
            call_sid=call_sid,
        )
        report_path = report_result.get("json_path")
        report = report_result.get("report") or {}
        print(
            f"[{call_sid}] report saved: {report_path} "
            f"score={report.get('overall_score')} outcome={report.get('outcome')}"
        )
    except Exception as exc:
        print(f"[{call_sid}] analysis skipped/failed: {exc}")
        traceback.print_exc()

    session_store.finish(
        call_sid,
        status="analyzed" if report_path else "transcribed",
        recording_path=str(recording_path),
        transcript_path=str(transcript_path) if transcript_path else None,
        report_path=str(report_path) if report_path else None,
    )


def _spawn_background(coro: object) -> None:
    """Keep a strong reference while a webhook-triggered job runs."""
    task = asyncio.create_task(coro)  # type: ignore[arg-type]
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


async def _process_recording(
    call_sid: str,
    recording_sid: str,
    recording_url: str,
    channels: int | str | None,
) -> None:
    """Download and analyze once, outside the Twilio webhook response."""
    if call_sid in _PROCESSED_CALLS or call_sid in _PROCESSING_CALLS:
        return
    _PROCESSING_CALLS.add(call_sid)
    try:
        try:
            session_store.update(call_sid, status="recording_processing")
        except (KeyError, ValueError):
            pass
        path = await asyncio.to_thread(
            download_recording_mp3,
            recording_sid,
            recording_url,
            call_sid,
            channels,
        )
        await asyncio.to_thread(_post_call_pipeline, call_sid, path)
        _PROCESSED_CALLS.add(call_sid)
    except Exception as exc:
        print(f"[{call_sid}] background recording processing failed: {exc}")
        traceback.print_exc()
        try:
            session_store.update(call_sid, status="processing_failed")
        except (KeyError, ValueError):
            pass
    finally:
        _PROCESSING_CALLS.discard(call_sid)


async def _status_recording_fallback(call_sid: str) -> None:
    """Give Twilio time to finalize, then fetch if /recording was missed."""
    await asyncio.sleep(8)
    if call_sid in _PROCESSED_CALLS or call_sid in _PROCESSING_CALLS:
        return
    _PROCESSING_CALLS.add(call_sid)
    try:
        try:
            session_store.update(call_sid, status="recording_fallback")
        except (KeyError, ValueError):
            pass
        path = await asyncio.to_thread(finalize_recording_for_call, call_sid)
        if not path:
            print(f"[{call_sid}] recording not ready; manual fetch remains available")
            return
        await asyncio.to_thread(_post_call_pipeline, call_sid, path)
        _PROCESSED_CALLS.add(call_sid)
    except Exception as exc:
        print(f"[{call_sid}] status recording fallback failed: {exc}")
        traceback.print_exc()
    finally:
        _PROCESSING_CALLS.discard(call_sid)


@app.get("/health")
async def health() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.api_route("/voice", methods=["GET", "POST"])
async def voice(request: Request) -> Response:
    """Return TwiML that connects the call to the Realtime media stream."""
    form = await request.form()
    call_sid = str(form.get("CallSid") or "").strip()
    scenario_id = (
        str(request.query_params.get("scenario") or form.get("scenario") or "scheduling")
    ).strip()

    if not PUBLIC_BASE_URL:
        vr = VoiceResponse()
        vr.say("Server misconfigured. Public base URL is missing.")
        vr.hangup()
        return Response(content=str(vr), media_type="application/xml")

    try:
        scenario = patient_brain.load_scenario(scenario_id)
    except (KeyError, ValueError) as exc:
        print(f"/voice scenario error: {exc}")
        vr = VoiceResponse()
        vr.say("Scenario configuration error. Goodbye.")
        vr.hangup()
        return Response(content=str(vr), media_type="application/xml")

    if call_sid:
        session_store.create(call_sid, scenario["id"], scenario["name"])
        state = call_controller.create(
            call_sid,
            scenario,
            max_seconds=int(os.getenv("MAX_CALL_SECONDS", "240")),
            max_idle_seconds=int(os.getenv("MAX_SILENCE_SECONDS", "30")),
        )
        session_store.add_turn(
            call_sid,
            "system",
            f"Realtime session started with {realtime_bridge.realtime_model()}",
        )
        session_store.update(call_sid, controller_state=state.snapshot())

    wss_base = _public_wss_base()
    stream_url = f"{wss_base}/media-stream"

    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=stream_url)
    stream.parameter(name="scenario", value=scenario["id"])
    if call_sid:
        stream.parameter(name="call_sid", value=call_sid)
    connect.append(stream)
    response.append(connect)
    print(f"[{call_sid}] Connecting Media Stream -> {stream_url} scenario={scenario['id']}")
    return Response(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    """Bidirectional audio bridge between Twilio and OpenAI Realtime."""
    await websocket.accept()
    print("Twilio media stream client connected")

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY missing — closing media stream")
        await websocket.close()
        return

    model = realtime_bridge.realtime_model()
    openai_url = f"wss://api.openai.com/v1/realtime?model={model}"

    stream_sid: str | None = None
    call_sid: str | None = None
    scenario_id = "scheduling"
    latest_patient_transcript = ""
    pending_end_reason: str | None = None
    ended = False

    try:
        async with websockets.connect(
            openai_url,
            additional_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        ) as openai_ws:
            print(f"Connected to OpenAI Realtime model={model}")

            async def receive_from_twilio() -> None:
                nonlocal stream_sid, call_sid, scenario_id, pending_end_reason, ended
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        event = data.get("event")

                        if event == "start":
                            start = data.get("start") or {}
                            stream_sid = start.get("streamSid")
                            call_sid = start.get("callSid") or call_sid
                            custom = start.get("customParameters") or {}
                            scenario_id = custom.get("scenario") or scenario_id
                            call_sid = custom.get("call_sid") or call_sid
                            print(f"Stream started sid={stream_sid} call={call_sid} scenario={scenario_id}")

                            scenario = patient_brain.load_scenario(scenario_id)
                            primed = patient_brain.load_primed_context(scenario_id)
                            if not primed or not primed.get("trained"):
                                print(
                                    f"[{call_sid}] Missing/stale prepared JSON for scenario={scenario_id}. "
                                    "Run make_call.py without --skip-train first."
                                )
                                ended = True
                                break

                            if call_sid:
                                session_store.create(call_sid, scenario["id"], scenario["name"])
                                state = call_controller.create(
                                    call_sid,
                                    scenario,
                                    max_seconds=int(os.getenv("MAX_CALL_SECONDS", "240")),
                                    max_idle_seconds=int(
                                        os.getenv("MAX_SILENCE_SECONDS", "30")
                                    ),
                                )
                                session_store.update(
                                    call_sid,
                                    controller_state=state.snapshot(),
                                )

                            instructions = realtime_bridge.build_patient_instructions(scenario, primed)
                            await openai_ws.send(
                                json.dumps(realtime_bridge.session_update_event(instructions))
                            )
                            print(
                                f"[{call_sid}] Realtime ready; waiting for clinic greeting "
                                "before patient responds"
                            )

                        elif (
                            event == "media"
                            and not pending_end_reason
                            and openai_ws.state.name == "OPEN"
                        ):
                            payload = (data.get("media") or {}).get("payload")
                            if payload:
                                await openai_ws.send(
                                    json.dumps(
                                        {
                                            "type": "input_audio_buffer.append",
                                            "audio": payload,
                                        }
                                    )
                                )

                        elif event == "stop":
                            print(f"Twilio stream stopped: {stream_sid}")
                            ended = True
                            break

                        elif event == "mark" and pending_end_reason and call_sid:
                            mark_name = (data.get("mark") or {}).get("name")
                            if mark_name == "patient-final-audio":
                                reason = pending_end_reason
                                pending_end_reason = None
                                print(
                                    f"[{call_sid}] Final patient audio played; "
                                    f"ending call: {reason}"
                                )
                                await _complete_call(call_sid, reason)
                                ended = True
                                break
                except WebSocketDisconnect:
                    print("Twilio websocket disconnected")
                    ended = True
                finally:
                    if openai_ws.state.name == "OPEN":
                        await openai_ws.close()

            async def send_to_twilio() -> None:
                nonlocal latest_patient_transcript, pending_end_reason, ended
                try:
                    async for raw in openai_ws:
                        if ended:
                            break
                        event = json.loads(raw)
                        etype = event.get("type")

                        if etype == "session.updated":
                            print("OpenAI session updated")

                        elif etype == "input_audio_buffer.speech_started" and stream_sid:
                            # Remove already-buffered model audio when the clinic interrupts.
                            if call_sid:
                                call_controller.touch_activity(call_sid)
                            await websocket.send_json(
                                {"event": "clear", "streamSid": stream_sid}
                            )

                        elif etype == "response.output_audio.delta" and event.get("delta") and stream_sid:
                            audio_payload = base64.b64encode(
                                base64.b64decode(event["delta"])
                            ).decode("utf-8")
                            await websocket.send_json(
                                {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": audio_payload},
                                }
                            )

                        elif etype in {
                            "response.output_audio_transcript.delta",
                            "response.audio_transcript.delta",
                        }:
                            latest_patient_transcript += event.get("delta") or ""

                        elif etype in {
                            "response.output_audio_transcript.done",
                            "response.audio_transcript.done",
                        }:
                            text = (event.get("transcript") or latest_patient_transcript or "").strip()
                            latest_patient_transcript = ""
                            if text and call_sid:
                                session_store.add_turn(call_sid, "patient", text)
                                print(f"[{call_sid}] patient: {text!r}")
                                reason = call_controller.on_patient_turn(call_sid, text)
                                state = call_controller.get(call_sid)
                                if state:
                                    session_store.update(
                                        call_sid,
                                        controller_state=state.snapshot(),
                                    )
                                if reason:
                                    print(
                                        f"[{call_sid}] End rule matched: {reason}; "
                                        "waiting for final audio playback"
                                    )
                                    pending_end_reason = reason
                                    if stream_sid:
                                        await websocket.send_json(
                                            {
                                                "event": "mark",
                                                "streamSid": stream_sid,
                                                "mark": {"name": "patient-final-audio"},
                                            }
                                        )

                        elif etype == "conversation.item.input_audio_transcription.completed":
                            text = (event.get("transcript") or "").strip()
                            if text and call_sid:
                                session_store.add_turn(call_sid, "clinic", text)
                                print(f"[{call_sid}] clinic: {text!r}")
                                reason = call_controller.on_clinic_turn(call_sid, text)
                                state = call_controller.get(call_sid)
                                if state:
                                    session_store.update(
                                        call_sid,
                                        controller_state=state.snapshot(),
                                    )
                                    if state.goal_signal_seen:
                                        print(
                                            f"[{call_sid}] Completion signal detected; "
                                            "patient should close naturally"
                                        )
                                if reason:
                                    await _complete_call(call_sid, reason)
                                    ended = True
                                    break

                        elif etype == "error":
                            print(f"OpenAI Realtime error: {event}")
                except Exception as exc:
                    print(f"send_to_twilio error: {exc}")
                    traceback.print_exc()

            async def enforce_call_limits() -> None:
                nonlocal ended
                while not ended:
                    await asyncio.sleep(2)
                    if not call_sid:
                        continue
                    reason = call_controller.check_limits(call_sid)
                    if not reason:
                        continue
                    print(f"[{call_sid}] Watchdog end rule matched: {reason}")
                    await _complete_call(call_sid, reason)
                    ended = True
                    try:
                        await websocket.close()
                    except Exception:
                        pass
                    break

            await asyncio.gather(
                receive_from_twilio(),
                send_to_twilio(),
                enforce_call_limits(),
            )
    except Exception as exc:
        print(f"Media stream bridge failed: {exc}")
        traceback.print_exc()
    finally:
        print("Media stream bridge closed")


async def _complete_call(call_sid: str, reason: str) -> None:
    """Hang up via Twilio REST so recording finalizes."""
    call_controller.mark_ended(call_sid, reason)
    state = call_controller.get(call_sid)
    try:
        session_store.finish(
            call_sid,
            status="completed",
            end_reason=reason,
        )
        if state:
            session_store.update(call_sid, controller_state=state.snapshot())
    except Exception as exc:
        print(f"[{call_sid}] Failed to persist end state: {exc}")

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print(f"[{call_sid}] Cannot hang up through Twilio: credentials missing")
        return
    try:
        from twilio.rest import Client

        Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).calls(call_sid).update(status="completed")
        print(f"[{call_sid}] Call marked completed: {reason}")
    except Exception as exc:
        print(f"[{call_sid}] Failed to complete call: {exc}")


@app.api_route("/status", methods=["POST"])
async def status(request: Request) -> PlainTextResponse:
    form = await request.form()
    call_sid = str(form.get("CallSid") or "").strip()
    call_status = str(form.get("CallStatus") or "").lower()
    print(f"Call status [{call_sid}]: {call_status}")

    if call_sid and call_status in {"completed", "busy", "failed", "no-answer", "canceled"}:
        try:
            session_store.update(call_sid, status=f"call_{call_status}")
        except Exception:
            pass

    if call_sid and call_status == "completed":
        _spawn_background(_status_recording_fallback(call_sid))

    return PlainTextResponse("ok")


@app.api_route("/recording", methods=["POST"])
async def recording(request: Request) -> PlainTextResponse:
    form = await request.form()
    status_value = str(form.get("RecordingStatus") or "").lower()
    recording_sid = str(form.get("RecordingSid") or "")
    call_sid = str(form.get("CallSid") or "unknown")
    recording_url = str(form.get("RecordingUrl") or "")
    channels = str(form.get("RecordingChannels") or "")
    duration = str(form.get("RecordingDuration") or "")

    print(
        f"Recording callback: status={status_value} call={call_sid} "
        f"recording={recording_sid} channels={channels} duration={duration}s"
    )

    if status_value != "completed" or not recording_sid:
        return PlainTextResponse("ignored")

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return PlainTextResponse("missing credentials", status_code=500)

    _spawn_background(
        _process_recording(
            call_sid,
            recording_sid,
            recording_url,
            channels,
        )
    )
    # Return immediately so Twilio does not time out/retry during STT + QA.
    return PlainTextResponse("accepted", status_code=202)


if __name__ == "__main__":
    import uvicorn

    if not PUBLIC_BASE_URL:
        print("Warning: PUBLIC_BASE_URL is not set in .env")
    print(f"Public base URL: {PUBLIC_BASE_URL or '(not set)'}")
    print(f"Realtime model: {realtime_bridge.realtime_model()}")
    print("Voice webhook: /voice")
    print("Media stream: /media-stream")
    print("Recording webhook: /recording")
    print("Status webhook: /status")
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
