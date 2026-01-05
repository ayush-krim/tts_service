"""
Speakers Router

Speaker management endpoints for voice cloning.
"""

import io
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, Form

from server.models import (
    SpeakerCreate,
    SpeakerInfo,
    SpeakerListResponse,
    ErrorResponse,
)

router = APIRouter()


def get_speaker_service(request: Request):
    """Dependency to get speaker service from app state."""
    return request.app.state.speaker_service


@router.post(
    "/speakers",
    response_model=SpeakerCreate,
    summary="Create new speaker",
    description="Register a new speaker from reference audio for voice cloning",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
async def create_speaker(
    audio_file: UploadFile = File(..., description="Reference audio file (WAV, MP3, FLAC)"),
    name: Optional[str] = Form(None, description="Speaker name"),
    speaker_service = Depends(get_speaker_service),
):
    """
    Create a new speaker from reference audio.

    Upload a reference audio file (3-10 seconds recommended) to extract
    a speaker embedding for voice cloning.

    Supported formats: WAV, MP3, FLAC
    Recommended: 24kHz, mono, 3-10 seconds of clear speech
    """
    try:
        import soundfile as sf

        # Read audio file
        audio_bytes = await audio_file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        try:
            audio, sample_rate = sf.read(audio_buffer)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read audio file: {e}"
            )

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Ensure float32
        audio = audio.astype(np.float32)

        # Validate duration
        duration_sec = len(audio) / sample_rate
        if duration_sec < 1.0:
            raise HTTPException(
                status_code=400,
                detail="Audio too short. Minimum 1 second required."
            )
        if duration_sec > 60.0:
            raise HTTPException(
                status_code=400,
                detail="Audio too long. Maximum 60 seconds allowed."
            )

        # Create speaker
        speaker_id, _ = await speaker_service.create_speaker(
            audio=audio,
            sample_rate=sample_rate,
            name=name
        )

        return SpeakerCreate(
            speaker_id=speaker_id,
            name=name,
            created_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/speakers",
    response_model=SpeakerListResponse,
    summary="List speakers",
    description="List all registered speakers",
)
async def list_speakers(
    speaker_service = Depends(get_speaker_service),
):
    """
    List all registered speakers.

    Returns speaker IDs, names, and creation timestamps.
    """
    speakers = speaker_service.list_speakers()

    speaker_list = [
        SpeakerInfo(
            id=s["id"],
            name=s.get("name"),
            created_at=datetime.fromisoformat(s["created_at"]),
            embedding_dim=256
        )
        for s in speakers
    ]

    return SpeakerListResponse(
        speakers=speaker_list,
        total=len(speaker_list)
    )


@router.get(
    "/speakers/{speaker_id}",
    response_model=SpeakerInfo,
    summary="Get speaker info",
    description="Get information about a specific speaker",
    responses={
        404: {"model": ErrorResponse},
    }
)
async def get_speaker(
    speaker_id: str,
    speaker_service = Depends(get_speaker_service),
):
    """
    Get information about a specific speaker.

    Returns speaker metadata (not the embedding itself).
    """
    info = speaker_service.get_speaker_info(speaker_id)

    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Speaker not found: {speaker_id}"
        )

    return SpeakerInfo(
        id=info["id"],
        name=info.get("name"),
        created_at=datetime.fromisoformat(info["created_at"]),
        embedding_dim=256
    )


@router.delete(
    "/speakers/{speaker_id}",
    summary="Delete speaker",
    description="Delete a speaker and its embedding",
    responses={
        404: {"model": ErrorResponse},
    }
)
async def delete_speaker(
    speaker_id: str,
    speaker_service = Depends(get_speaker_service),
):
    """
    Delete a speaker.

    Removes the speaker from the registry and cache.
    The default speaker cannot be deleted.
    """
    if speaker_id == "default":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the default speaker"
        )

    deleted = await speaker_service.delete_speaker(speaker_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Speaker not found: {speaker_id}"
        )

    return {"status": "deleted", "speaker_id": speaker_id}
