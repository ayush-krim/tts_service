"""
Speaker Management Service

Handles speaker embeddings for voice cloning:
- Extract embeddings from reference audio
- Cache embeddings in Redis
- Manage speaker registry
"""

import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from server.config import settings


class SpeakerService:
    """
    Service for managing speaker embeddings.

    Speaker embeddings are 256-dimensional vectors extracted from
    reference audio using the VoiceEncoder (CAMPPlus) model.
    """

    EMBEDDING_DIM = 256
    DEFAULT_SPEAKER_ID = "default"

    def __init__(
        self,
        voice_encoder=None,
        redis_client=None,
        cache_ttl: int = 3600,
        device: str = "cuda"
    ):
        """
        Initialize speaker service.

        Args:
            voice_encoder: VoiceEncoder model for embedding extraction
            redis_client: Redis client for caching (optional)
            cache_ttl: Cache TTL in seconds
            device: Device for inference
        """
        self.voice_encoder = voice_encoder
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.device = device

        # In-memory cache for when Redis is not available
        self._memory_cache: Dict[str, Tuple[np.ndarray, float]] = {}

        # Speaker registry (in production, use database)
        self._registry: Dict[str, dict] = {
            self.DEFAULT_SPEAKER_ID: {
                "id": self.DEFAULT_SPEAKER_ID,
                "name": "Default Speaker",
                "created_at": datetime.now().isoformat(),
                "embedding_source": "pretrained"
            }
        }

        # Default embedding (will be set when model loads)
        self._default_embedding: Optional[np.ndarray] = None

    def set_voice_encoder(self, voice_encoder):
        """Set the voice encoder model."""
        self.voice_encoder = voice_encoder

    def set_default_embedding(self, embedding: np.ndarray):
        """Set the default speaker embedding."""
        self._default_embedding = embedding

    async def get_embedding(
        self,
        speaker_id: str
    ) -> Optional[np.ndarray]:
        """
        Get speaker embedding by ID.

        Checks cache first, then registry.

        Args:
            speaker_id: Speaker identifier

        Returns:
            256-dim numpy array or None if not found
        """
        # Handle default speaker
        if speaker_id == self.DEFAULT_SPEAKER_ID:
            return self._default_embedding

        # Check Redis cache first
        if self.redis:
            try:
                cached = await self.redis.get(f"speaker:{speaker_id}")
                if cached:
                    return pickle.loads(cached)
            except Exception:
                pass  # Fall through to memory cache

        # Check memory cache
        if speaker_id in self._memory_cache:
            embedding, timestamp = self._memory_cache[speaker_id]
            if time.time() - timestamp < self.cache_ttl:
                return embedding
            else:
                # Expired
                del self._memory_cache[speaker_id]

        return None

    async def create_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
        name: Optional[str] = None
    ) -> Tuple[str, np.ndarray]:
        """
        Create new speaker from reference audio.

        Args:
            audio: Reference audio waveform (float32)
            sample_rate: Audio sample rate
            name: Optional speaker name

        Returns:
            Tuple of (speaker_id, embedding)
        """
        if self.voice_encoder is None:
            raise RuntimeError("Voice encoder not initialized")

        # Extract embedding
        embedding = await self._extract_embedding(audio, sample_rate)

        # Generate speaker ID from audio hash
        audio_hash = hashlib.md5(audio.tobytes()).hexdigest()[:12]
        speaker_id = f"spk_{audio_hash}"

        # Store embedding
        await self._cache_embedding(speaker_id, embedding)

        # Register speaker
        self._registry[speaker_id] = {
            "id": speaker_id,
            "name": name or f"Speaker {speaker_id[:8]}",
            "created_at": datetime.now().isoformat(),
            "embedding_source": "reference_audio"
        }

        return speaker_id, embedding

    async def _extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000
    ) -> np.ndarray:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio waveform (float32, mono)
            sample_rate: Sample rate of input audio

        Returns:
            256-dim speaker embedding
        """
        import torch

        # Ensure correct sample rate (VoiceEncoder expects 24kHz)
        if sample_rate != 24000:
            from server.services.telephony import TelephonyConverter
            audio = TelephonyConverter.resample_high_quality(
                audio, sample_rate, 24000
            )

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dim

        audio_tensor = audio_tensor.to(self.device)

        # Extract embedding
        with torch.no_grad():
            embedding = self.voice_encoder(audio_tensor)

        return embedding.cpu().numpy().squeeze()

    async def _cache_embedding(
        self,
        speaker_id: str,
        embedding: np.ndarray
    ):
        """Cache speaker embedding."""
        # Redis cache
        if self.redis:
            try:
                await self.redis.setex(
                    f"speaker:{speaker_id}",
                    self.cache_ttl,
                    pickle.dumps(embedding)
                )
            except Exception:
                pass  # Fall through to memory cache

        # Memory cache
        self._memory_cache[speaker_id] = (embedding, time.time())

    async def delete_speaker(self, speaker_id: str) -> bool:
        """
        Delete speaker from registry and cache.

        Args:
            speaker_id: Speaker to delete

        Returns:
            True if deleted, False if not found
        """
        if speaker_id == self.DEFAULT_SPEAKER_ID:
            return False  # Cannot delete default speaker

        deleted = False

        # Remove from Redis
        if self.redis:
            try:
                await self.redis.delete(f"speaker:{speaker_id}")
                deleted = True
            except Exception:
                pass

        # Remove from memory cache
        if speaker_id in self._memory_cache:
            del self._memory_cache[speaker_id]
            deleted = True

        # Remove from registry
        if speaker_id in self._registry:
            del self._registry[speaker_id]
            deleted = True

        return deleted

    def list_speakers(self) -> List[dict]:
        """
        List all registered speakers.

        Returns:
            List of speaker info dicts
        """
        return list(self._registry.values())

    def get_speaker_info(self, speaker_id: str) -> Optional[dict]:
        """
        Get speaker metadata.

        Args:
            speaker_id: Speaker ID

        Returns:
            Speaker info dict or None
        """
        return self._registry.get(speaker_id)

    def speaker_exists(self, speaker_id: str) -> bool:
        """Check if speaker exists."""
        return speaker_id in self._registry

    async def preload_speakers(self, speaker_dir: Path):
        """
        Preload speakers from a directory.

        Expected structure:
            speaker_dir/
                speaker1/
                    reference.wav
                    metadata.json
                speaker2/
                    reference.wav
        """
        if not speaker_dir.exists():
            return

        import soundfile as sf

        for speaker_path in speaker_dir.iterdir():
            if not speaker_path.is_dir():
                continue

            # Find reference audio
            audio_file = None
            for ext in ['.wav', '.mp3', '.flac']:
                candidate = speaker_path / f"reference{ext}"
                if candidate.exists():
                    audio_file = candidate
                    break

            if not audio_file:
                continue

            # Load audio
            audio, sr = sf.read(str(audio_file))

            # Load metadata if exists
            metadata_file = speaker_path / "metadata.json"
            name = speaker_path.name
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    name = metadata.get("name", name)

            # Create speaker
            speaker_id, _ = await self.create_speaker(audio, sr, name)
            print(f"  Loaded speaker: {speaker_id} ({name})")
