"""
Telephony Audio Conversion Service

Converts 24kHz audio from Chatterbox to telephony formats:
- Resampling to 8kHz
- G.711 mu-law encoding
- G.711 A-law encoding
- RTP packet streaming
"""

import asyncio
import struct
import time
from typing import AsyncGenerator, Optional

import numpy as np


class TelephonyConverter:
    """Convert audio for telephony (VoIP) systems."""

    @staticmethod
    def resample(
        audio: np.ndarray,
        orig_sr: int = 24000,
        target_sr: int = 8000
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Uses linear interpolation for speed. For higher quality,
        use librosa.resample with 'kaiser_best'.
        """
        if orig_sr == target_sr:
            return audio

        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)

        # Linear interpolation (fast)
        indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)

        return resampled.astype(audio.dtype)

    @staticmethod
    def resample_high_quality(
        audio: np.ndarray,
        orig_sr: int = 24000,
        target_sr: int = 8000
    ) -> np.ndarray:
        """
        High-quality resampling using librosa.

        Slower but better audio quality.
        """
        try:
            import librosa
            return librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type='kaiser_best'
            )
        except ImportError:
            # Fallback to simple resampling
            return TelephonyConverter.resample(audio, orig_sr, target_sr)

    @staticmethod
    def float_to_pcm16(audio: np.ndarray) -> bytes:
        """
        Convert float32 audio [-1.0, 1.0] to 16-bit PCM bytes.
        """
        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)

        # Convert to int16
        pcm16 = (audio * 32767).astype(np.int16)

        return pcm16.tobytes()

    @staticmethod
    def pcm16_to_float(pcm_bytes: bytes) -> np.ndarray:
        """Convert 16-bit PCM bytes to float32 array."""
        pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return pcm16.astype(np.float32) / 32767.0

    @staticmethod
    def to_mulaw(pcm16_bytes: bytes) -> bytes:
        """
        Encode 16-bit PCM to G.711 mu-law.

        G.711 mu-law is standard for North American telephony.
        """
        try:
            import audioop
            return audioop.lin2ulaw(pcm16_bytes, 2)
        except ImportError:
            # Manual mu-law encoding
            return TelephonyConverter._mulaw_encode(pcm16_bytes)

    @staticmethod
    def _mulaw_encode(pcm16_bytes: bytes) -> bytes:
        """Manual mu-law encoding when audioop not available."""
        MU = 255

        pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        # Normalize to [-1, 1]
        x = pcm16.astype(np.float32) / 32768.0

        # Mu-law compression
        sign = np.sign(x)
        x_abs = np.abs(x)
        compressed = sign * np.log1p(MU * x_abs) / np.log1p(MU)

        # Quantize to 8-bit
        mulaw = ((compressed + 1) / 2 * 255).astype(np.uint8)

        return mulaw.tobytes()

    @staticmethod
    def to_alaw(pcm16_bytes: bytes) -> bytes:
        """
        Encode 16-bit PCM to G.711 A-law.

        G.711 A-law is standard for European telephony.
        """
        try:
            import audioop
            return audioop.lin2alaw(pcm16_bytes, 2)
        except ImportError:
            # Manual A-law encoding (simplified)
            return TelephonyConverter._alaw_encode(pcm16_bytes)

    @staticmethod
    def _alaw_encode(pcm16_bytes: bytes) -> bytes:
        """Manual A-law encoding when audioop not available."""
        A = 87.6

        pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        x = pcm16.astype(np.float32) / 32768.0

        sign = np.sign(x)
        x_abs = np.abs(x)

        # A-law compression
        mask = x_abs < (1 / A)
        compressed = np.where(
            mask,
            A * x_abs / (1 + np.log(A)),
            (1 + np.log(A * x_abs)) / (1 + np.log(A))
        )
        compressed = sign * compressed

        # Quantize to 8-bit
        alaw = ((compressed + 1) / 2 * 255).astype(np.uint8)

        return alaw.tobytes()

    @staticmethod
    def from_mulaw(mulaw_bytes: bytes) -> bytes:
        """Decode G.711 mu-law to 16-bit PCM."""
        try:
            import audioop
            return audioop.ulaw2lin(mulaw_bytes, 2)
        except ImportError:
            raise NotImplementedError("Manual mu-law decoding not implemented")

    @staticmethod
    def from_alaw(alaw_bytes: bytes) -> bytes:
        """Decode G.711 A-law to 16-bit PCM."""
        try:
            import audioop
            return audioop.alaw2lin(alaw_bytes, 2)
        except ImportError:
            raise NotImplementedError("Manual A-law decoding not implemented")

    @classmethod
    def convert_for_telephony(
        cls,
        audio_24k: np.ndarray,
        encoding: str = "mulaw",
        target_sr: int = 8000,
        high_quality: bool = False
    ) -> bytes:
        """
        Full conversion pipeline for telephony.

        Args:
            audio_24k: Float32 audio at 24kHz
            encoding: "mulaw", "alaw", or "pcm"
            target_sr: Target sample rate (default 8000)
            high_quality: Use high-quality resampling

        Returns:
            Encoded audio bytes
        """
        # Resample to 8kHz
        if high_quality:
            audio_8k = cls.resample_high_quality(audio_24k, 24000, target_sr)
        else:
            audio_8k = cls.resample(audio_24k, 24000, target_sr)

        # Convert to PCM16
        pcm16 = cls.float_to_pcm16(audio_8k)

        # Encode
        if encoding == "mulaw":
            return cls.to_mulaw(pcm16)
        elif encoding == "alaw":
            return cls.to_alaw(pcm16)
        else:
            return pcm16


class RTPStreamer:
    """
    Stream audio via RTP for VoIP integration.

    RTP (Real-time Transport Protocol) is used for streaming
    audio in VoIP systems like Twilio, Vonage, etc.
    """

    # RTP payload types
    PAYLOAD_TYPE_PCMU = 0   # G.711 mu-law
    PAYLOAD_TYPE_PCMA = 8   # G.711 A-law
    PAYLOAD_TYPE_L16 = 11   # 16-bit linear PCM

    # Standard packet size for telephony (20ms @ 8kHz)
    PACKET_SIZE = 160
    PACKET_DURATION_MS = 20

    def __init__(self, ssrc: Optional[int] = None):
        """
        Initialize RTP streamer.

        Args:
            ssrc: Synchronization source identifier (random if not provided)
        """
        self.ssrc = ssrc or int(time.time()) & 0xFFFFFFFF
        self.sequence = 0
        self.timestamp = 0

    def build_rtp_packet(
        self,
        payload: bytes,
        payload_type: int = PAYLOAD_TYPE_PCMU,
        marker: bool = False
    ) -> bytes:
        """
        Build RTP packet with header.

        RTP Header (12 bytes):
        - Version (2 bits): 2
        - Padding (1 bit): 0
        - Extension (1 bit): 0
        - CSRC count (4 bits): 0
        - Marker (1 bit): For start of talk burst
        - Payload type (7 bits): Audio codec
        - Sequence number (16 bits): Packet counter
        - Timestamp (32 bits): Sampling instant
        - SSRC (32 bits): Source identifier
        """
        version = 2
        padding = 0
        extension = 0
        cc = 0  # CSRC count

        first_byte = (version << 6) | (padding << 5) | (extension << 4) | cc
        second_byte = (int(marker) << 7) | payload_type

        header = struct.pack(
            '!BBHII',
            first_byte,
            second_byte,
            self.sequence & 0xFFFF,
            self.timestamp & 0xFFFFFFFF,
            self.ssrc
        )

        # Increment for next packet
        self.sequence += 1
        self.timestamp += self.PACKET_SIZE

        return header + payload

    def reset(self):
        """Reset sequence and timestamp counters."""
        self.sequence = 0
        self.timestamp = 0

    async def stream_rtp(
        self,
        audio_bytes: bytes,
        dest_host: str,
        dest_port: int,
        payload_type: int = PAYLOAD_TYPE_PCMU
    ):
        """
        Stream audio as RTP packets over UDP.

        Args:
            audio_bytes: Encoded audio (mu-law, A-law, or PCM)
            dest_host: Destination IP address
            dest_port: Destination UDP port
            payload_type: RTP payload type
        """
        loop = asyncio.get_event_loop()

        # Create UDP socket
        transport, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol,
            remote_addr=(dest_host, dest_port)
        )

        try:
            for i in range(0, len(audio_bytes), self.PACKET_SIZE):
                chunk = audio_bytes[i:i + self.PACKET_SIZE]

                # Pad last packet if needed (silence byte for mu-law)
                if len(chunk) < self.PACKET_SIZE:
                    padding_byte = b'\x7F' if payload_type == self.PAYLOAD_TYPE_PCMU else b'\xD5'
                    chunk = chunk + padding_byte * (self.PACKET_SIZE - len(chunk))

                # Mark first packet of stream
                marker = (i == 0)
                packet = self.build_rtp_packet(chunk, payload_type, marker)

                transport.sendto(packet)

                # Maintain real-time pacing (20ms per packet)
                await asyncio.sleep(self.PACKET_DURATION_MS / 1000)

        finally:
            transport.close()

    def packetize(
        self,
        audio_bytes: bytes,
        payload_type: int = PAYLOAD_TYPE_PCMU
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate RTP packets without sending.

        Useful for WebSocket streaming or custom transport.

        Yields:
            RTP packets as bytes
        """
        for i in range(0, len(audio_bytes), self.PACKET_SIZE):
            chunk = audio_bytes[i:i + self.PACKET_SIZE]

            # Pad last packet
            if len(chunk) < self.PACKET_SIZE:
                padding_byte = b'\x7F' if payload_type == self.PAYLOAD_TYPE_PCMU else b'\xD5'
                chunk = chunk + padding_byte * (self.PACKET_SIZE - len(chunk))

            marker = (i == 0)
            yield self.build_rtp_packet(chunk, payload_type, marker)
