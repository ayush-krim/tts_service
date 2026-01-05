"""
Synthesize Router

TTS synthesis endpoints for batch and streaming audio generation.
"""

import base64
import io
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from server.models import (
    SynthesizeRequest,
    SynthesizeStreamRequest,
    AudioFormat,
    SampleRate,
    ErrorResponse,
)
from server.services.tts import TTSService

router = APIRouter()


def get_tts_service(request: Request) -> TTSService:
    """Dependency to get TTS service from app state."""
    return request.app.state.tts_service


def get_speaker_service(request: Request):
    """Dependency to get speaker service from app state."""
    return request.app.state.speaker_service


@router.post(
    "/synthesize",
    summary="Synthesize speech from text",
    description="Generate audio from input text with optional voice cloning",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "audio/x-pcm": {},
                "audio/basic": {},
            }
        },
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
async def synthesize(
    body: SynthesizeRequest,
    tts_service: TTSService = Depends(get_tts_service),
    speaker_service = Depends(get_speaker_service),
):
    """
    Synthesize speech from text.

    Returns audio in the requested format with performance metrics in headers.

    Headers:
    - X-Inference-Time-Ms: Time to generate audio
    - X-Audio-Duration-Ms: Duration of generated audio
    - X-RTF: Real-time factor (< 1.0 = faster than real-time)
    - X-Model-Version: Model version used
    """
    try:
        # Get speaker embedding
        speaker_embedding = await speaker_service.get_embedding(body.speaker_id)
        if speaker_embedding is None and body.speaker_id != "default":
            raise HTTPException(
                status_code=404,
                detail=f"Speaker not found: {body.speaker_id}"
            )

        # Synthesize
        audio, inference_time_ms = await tts_service.synthesize(
            text=body.text,
            speaker_embedding=speaker_embedding,
            sample_rate=body.sample_rate.value
        )

        # Calculate metrics
        audio_duration_ms = tts_service.get_audio_duration_ms(
            audio, body.sample_rate.value
        )
        rtf = tts_service.calculate_rtf(inference_time_ms, audio_duration_ms)

        # Convert to output format
        if body.format == AudioFormat.WAV:
            content = _to_wav(audio, body.sample_rate.value)
            media_type = "audio/wav"
        elif body.format == AudioFormat.PCM:
            from server.services.telephony import TelephonyConverter
            content = TelephonyConverter.float_to_pcm16(audio)
            media_type = "audio/x-pcm"
        elif body.format == AudioFormat.MULAW:
            from server.services.telephony import TelephonyConverter
            pcm = TelephonyConverter.float_to_pcm16(audio)
            content = TelephonyConverter.to_mulaw(pcm)
            media_type = "audio/basic"
        elif body.format == AudioFormat.ALAW:
            from server.services.telephony import TelephonyConverter
            pcm = TelephonyConverter.float_to_pcm16(audio)
            content = TelephonyConverter.to_alaw(pcm)
            media_type = "audio/basic"
        else:
            content = _to_wav(audio, body.sample_rate.value)
            media_type = "audio/wav"

        # Response with metrics headers
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "X-Inference-Time-Ms": f"{inference_time_ms:.2f}",
                "X-Audio-Duration-Ms": f"{audio_duration_ms:.2f}",
                "X-RTF": f"{rtf:.4f}",
                "X-Model-Version": tts_service.get_version(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/synthesize/stream",
    summary="Stream synthesized speech",
    description="Generate audio and stream it in chunks using Server-Sent Events",
    responses={
        200: {"content": {"text/event-stream": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
async def synthesize_stream(
    body: SynthesizeStreamRequest,
    tts_service: TTSService = Depends(get_tts_service),
    speaker_service = Depends(get_speaker_service),
):
    """
    Stream synthesized speech using Server-Sent Events.

    Events:
    - metadata: Stream configuration (sample_rate, format, etc.)
    - audio: Base64-encoded audio chunk
    - done: End of stream
    - error: Error message

    Each audio event contains a base64-encoded chunk of audio data.
    """
    try:
        # Get speaker embedding
        speaker_embedding = await speaker_service.get_embedding(body.speaker_id)

        async def generate():
            try:
                # Send metadata first
                yield {
                    "event": "metadata",
                    "data": {
                        "sample_rate": body.sample_rate.value,
                        "format": body.format.value,
                        "chunk_size_ms": body.chunk_size_ms,
                        "speaker_id": body.speaker_id,
                    }
                }

                # Stream audio chunks
                chunk_count = 0
                async for chunk in tts_service.synthesize_stream(
                    text=body.text,
                    speaker_embedding=speaker_embedding,
                    chunk_size_ms=body.chunk_size_ms,
                    sample_rate=body.sample_rate.value,
                    format=body.format.value
                ):
                    # Encode chunk as base64
                    chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                    yield {
                        "event": "audio",
                        "data": chunk_b64
                    }
                    chunk_count += 1

                # Send completion
                yield {
                    "event": "done",
                    "data": {"chunks": chunk_count}
                }

            except Exception as e:
                yield {
                    "event": "error",
                    "data": str(e)
                }

        return EventSourceResponse(generate())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _to_wav(audio, sample_rate: int) -> bytes:
    """Convert numpy array to WAV bytes."""
    import wave
    import numpy as np

    # Ensure audio is float32
    audio = np.asarray(audio, dtype=np.float32)

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write to WAV
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

    return buffer.getvalue()
