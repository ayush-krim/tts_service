"""
TTS Service

Core text-to-speech service using Chatterbox Turbo model.
Supports both batch and streaming synthesis.
"""

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple

import numpy as np
import torch

from server.config import settings
from server.services.telephony import TelephonyConverter


class TTSService:
    """
    Text-to-Speech service using Chatterbox Turbo.

    Supports:
    - Batch synthesis (full audio at once)
    - Streaming synthesis (chunked output)
    - Multiple audio formats (WAV, PCM, mu-law, A-law)
    - Voice cloning via speaker embeddings
    """

    MODEL_SAMPLE_RATE = 24000  # Chatterbox native sample rate
    VERSION = "chatterbox-turbo-1.0"

    def __init__(
        self,
        model_path: str = None,
        engine_path: str = None,
        device: str = "cuda",
        use_tensorrt: bool = False,
        use_onnx: bool = False
    ):
        """
        Initialize TTS service.

        Args:
            model_path: Path to model weights
            engine_path: Path to TensorRT engines
            device: Device to run on ("cuda" or "cpu")
            use_tensorrt: Use TensorRT engines for inference
            use_onnx: Use ONNX Runtime for inference
        """
        self.model_path = model_path or settings.model_path
        self.engine_path = engine_path or settings.engine_path
        self.device = device
        self.use_tensorrt = use_tensorrt
        self.use_onnx = use_onnx

        self.model = None
        self.voice_encoder = None
        self._default_speaker_embedding = None
        self._loaded = False

    async def load(self):
        """Load the TTS model."""
        if self._loaded:
            return

        print(f"Loading Chatterbox Turbo model on {self.device}...")

        if self.use_tensorrt:
            await self._load_tensorrt()
        elif self.use_onnx:
            await self._load_onnx()
        else:
            await self._load_pytorch()

        self._loaded = True
        print("Model loaded successfully")

    async def _load_pytorch(self):
        """Load PyTorch model."""
        try:
            from chatterbox import ChatterboxTurboTTS

            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            self.model.eval()

            # Get voice encoder reference
            self.voice_encoder = self.model.voice_encoder

            # Generate default speaker embedding
            # Use a short silence as reference for neutral voice
            neutral_audio = np.zeros(self.MODEL_SAMPLE_RATE * 2, dtype=np.float32)
            self._default_speaker_embedding = self._extract_embedding_sync(neutral_audio)

        except ImportError:
            raise RuntimeError(
                "chatterbox-tts not installed. "
                "Install with: pip install chatterbox-tts"
            )

    async def _load_tensorrt(self):
        """Load TensorRT engines."""
        # TensorRT loading would go here
        # For now, fall back to PyTorch
        print("TensorRT loading not yet implemented, using PyTorch")
        await self._load_pytorch()

    async def _load_onnx(self):
        """Load ONNX Runtime models."""
        # ONNX loading would go here
        # For now, fall back to PyTorch
        print("ONNX loading not yet implemented, using PyTorch")
        await self._load_pytorch()

    def _extract_embedding_sync(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding synchronously."""
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)

        with torch.no_grad():
            embedding = self.voice_encoder(audio_tensor)

        return embedding.cpu().numpy().squeeze()

    @property
    def default_embedding(self) -> np.ndarray:
        """Get default speaker embedding."""
        return self._default_speaker_embedding

    async def synthesize(
        self,
        text: str,
        speaker_embedding: Optional[np.ndarray] = None,
        sample_rate: int = 24000
    ) -> Tuple[np.ndarray, float]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding for voice cloning
            sample_rate: Output sample rate

        Returns:
            Tuple of (audio_array, inference_time_ms)
        """
        if not self._loaded:
            await self.load()

        # Use default embedding if none provided
        if speaker_embedding is None:
            speaker_embedding = self._default_speaker_embedding

        # Run inference
        start_time = time.perf_counter()

        audio = await self._run_inference(text, speaker_embedding)

        if self.device == "cuda":
            torch.cuda.synchronize()

        inference_time = (time.perf_counter() - start_time) * 1000

        # Resample if needed
        if sample_rate != self.MODEL_SAMPLE_RATE:
            audio = TelephonyConverter.resample_high_quality(
                audio, self.MODEL_SAMPLE_RATE, sample_rate
            )

        return audio, inference_time

    async def _run_inference(
        self,
        text: str,
        speaker_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Run model inference.

        Returns audio at native 24kHz sample rate.
        """
        # Convert embedding to tensor
        emb_tensor = torch.from_numpy(speaker_embedding).float()
        if emb_tensor.dim() == 1:
            emb_tensor = emb_tensor.unsqueeze(0)
        emb_tensor = emb_tensor.to(self.device)

        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            self._inference_sync,
            text,
            emb_tensor
        )

        return audio

    def _inference_sync(
        self,
        text: str,
        speaker_embedding: torch.Tensor
    ) -> np.ndarray:
        """Synchronous inference."""
        with torch.no_grad():
            # Generate audio using Chatterbox
            audio = self.model.generate(
                text,
                speaker_embedding=speaker_embedding
            )

            # Handle different return types
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            if audio.ndim > 1:
                audio = audio.squeeze()

        return audio

    async def synthesize_stream(
        self,
        text: str,
        speaker_embedding: Optional[np.ndarray] = None,
        chunk_size_ms: int = 100,
        sample_rate: int = 8000,
        format: str = "pcm"
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio in chunks.

        This is a pseudo-streaming implementation that generates
        full audio then chunks it. True streaming requires model
        support for incremental generation.

        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding
            chunk_size_ms: Chunk duration in milliseconds
            sample_rate: Output sample rate
            format: Output format (pcm, mulaw, alaw)

        Yields:
            Audio chunks as bytes
        """
        # Generate full audio first
        audio, _ = await self.synthesize(text, speaker_embedding, sample_rate)

        # Calculate chunk size in samples
        chunk_samples = int(sample_rate * chunk_size_ms / 1000)

        # Convert to output format
        if format == "mulaw":
            converter = TelephonyConverter.to_mulaw
        elif format == "alaw":
            converter = TelephonyConverter.to_alaw
        else:
            converter = None

        # Stream chunks
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]

            # Convert to bytes
            pcm_bytes = TelephonyConverter.float_to_pcm16(chunk)

            if converter:
                chunk_bytes = converter(pcm_bytes)
            else:
                chunk_bytes = pcm_bytes

            yield chunk_bytes

            # Small delay to simulate real-time streaming
            await asyncio.sleep(0.001)

    def get_audio_duration_ms(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate audio duration in milliseconds."""
        return len(audio) / sample_rate * 1000

    def calculate_rtf(
        self,
        inference_time_ms: float,
        audio_duration_ms: float
    ) -> float:
        """
        Calculate Real-Time Factor.

        RTF < 1.0 means faster than real-time.
        """
        if audio_duration_ms == 0:
            return 0.0
        return inference_time_ms / audio_duration_ms

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_version(self) -> str:
        """Get model version."""
        return self.VERSION

    def get_memory_usage(self) -> dict:
        """Get GPU memory usage."""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1e6,
                "reserved_mb": torch.cuda.memory_reserved() / 1e6,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6
            }
        return {}
