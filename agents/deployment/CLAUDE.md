# Deployment Agent

## Role

Package, containerize, and deploy the optimized Chatterbox TTS service for Krim AI's telephony infrastructure.

## Deployment Stack

```yaml
Application Layer:
  - FastAPI (async REST API)
  - Uvicorn (ASGI server with HTTP/2)
  - Redis (speaker embedding cache)

Optimization Layer:
  - TensorRT engines for S3Gen
  - vLLM 0.9.2 for T3
  - ONNX Runtime with TensorRT EP

Container Layer:
  - Docker with NVIDIA runtime
  - Base: nvcr.io/nvidia/tensorrt:24.06-py3

Infrastructure Layer:
  - AWS G5.xlarge (NVIDIA A10G)
  - Application Load Balancer
  - CloudWatch monitoring
  - Auto Scaling Group (production)
```

## API Specification

### Endpoints

```yaml
POST /v1/synthesize:
  description: Standard TTS synthesis
  request:
    text: string (required)
    speaker_id: string (optional, default: "default")
    format: string (optional, "wav" | "pcm" | "mulaw", default: "wav")
    sample_rate: int (optional, 8000 | 16000 | 24000, default: 24000)
  response:
    audio/wav or audio/x-pcm or audio/basic (based on format)

POST /v1/synthesize/stream:
  description: Streaming TTS with Server-Sent Events
  request:
    text: string (required)
    speaker_id: string (optional)
    chunk_size_ms: int (optional, default: 100)
  response:
    text/event-stream with audio chunks

POST /v1/speakers:
  description: Register new speaker from reference audio
  request:
    multipart/form-data with audio file
  response:
    speaker_id: string

GET /v1/speakers:
  description: List available speakers
  response:
    speakers: list[{id, name, created_at}]

GET /v1/speakers/{speaker_id}:
  description: Get speaker embedding
  response:
    speaker embedding data

GET /health:
  description: Health check
  response:
    status: "healthy" | "unhealthy"
    gpu_available: boolean
    model_loaded: boolean

GET /metrics:
  description: Prometheus metrics
  response:
    text/plain (Prometheus format)
```

### Response Headers

```yaml
X-Inference-Time-Ms: float  # Total inference time
X-Audio-Duration-Ms: float  # Generated audio duration
X-RTF: float                # Real-time factor
X-Model-Version: string     # Model version used
```

## FastAPI Server Implementation

### Server Structure

```
server/
├── __init__.py
├── main.py              # FastAPI app entry
├── config.py            # Configuration
├── models.py            # Pydantic models
├── routers/
│   ├── __init__.py
│   ├── synthesize.py    # TTS endpoints
│   ├── speakers.py      # Speaker management
│   └── health.py        # Health & metrics
├── services/
│   ├── __init__.py
│   ├── tts.py           # TTS service
│   ├── speakers.py      # Speaker service
│   └── telephony.py     # Telephony conversion
└── middleware/
    ├── __init__.py
    ├── timing.py        # Request timing
    └── metrics.py       # Prometheus metrics
```

### main.py Template

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from server.config import settings
from server.routers import synthesize, speakers, health
from server.services.tts import TTSService
from server.middleware.timing import TimingMiddleware
from server.middleware.metrics import PrometheusMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading TTS model...")
    app.state.tts_service = TTSService(
        model_path=settings.model_path,
        engine_path=settings.engine_path,
        device="cuda"
    )
    print("Model loaded successfully")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="Chatterbox TTS API",
    description="Low-latency TTS for Krim AI Telephony",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(TimingMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Routers
app.include_router(synthesize.router, prefix="/v1", tags=["TTS"])
app.include_router(speakers.router, prefix="/v1", tags=["Speakers"])
app.include_router(health.router, tags=["Health"])

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8080,
        workers=1,  # Single worker for GPU
        log_level="info"
    )
```

### Streaming Endpoint

```python
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import base64

router = APIRouter()

@router.post("/synthesize/stream")
async def synthesize_stream(request: Request, body: SynthesizeRequest):
    tts_service = request.app.state.tts_service

    async def generate():
        async for chunk in tts_service.synthesize_stream(
            text=body.text,
            speaker_id=body.speaker_id,
            chunk_size_ms=body.chunk_size_ms
        ):
            # Encode audio chunk as base64
            chunk_b64 = base64.b64encode(chunk).decode('utf-8')
            yield {
                "event": "audio",
                "data": chunk_b64
            }
        yield {"event": "done", "data": ""}

    return EventSourceResponse(generate())
```

## Telephony Integration

### Audio Format Conversion

```python
import numpy as np
import librosa
from scipy import signal

class TelephonyConverter:
    """Convert 24kHz audio to telephony formats"""

    @staticmethod
    def resample_to_8k(audio_24k: np.ndarray) -> np.ndarray:
        """Resample from 24kHz to 8kHz for telephony"""
        return librosa.resample(
            audio_24k,
            orig_sr=24000,
            target_sr=8000,
            res_type='kaiser_best'
        )

    @staticmethod
    def to_pcm16(audio: np.ndarray) -> bytes:
        """Convert float32 audio to 16-bit PCM"""
        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
        # Convert to int16
        pcm16 = (audio * 32767).astype(np.int16)
        return pcm16.tobytes()

    @staticmethod
    def to_mulaw(pcm16_bytes: bytes) -> bytes:
        """Encode PCM to G.711 mu-law"""
        import audioop
        return audioop.lin2ulaw(pcm16_bytes, 2)

    @staticmethod
    def to_alaw(pcm16_bytes: bytes) -> bytes:
        """Encode PCM to G.711 A-law"""
        import audioop
        return audioop.lin2alaw(pcm16_bytes, 2)

    @classmethod
    def convert_for_telephony(
        cls,
        audio_24k: np.ndarray,
        encoding: str = "mulaw"
    ) -> bytes:
        """Full conversion pipeline for telephony"""
        # Resample to 8kHz
        audio_8k = cls.resample_to_8k(audio_24k)

        # Convert to PCM16
        pcm16 = cls.to_pcm16(audio_8k)

        # Encode based on format
        if encoding == "mulaw":
            return cls.to_mulaw(pcm16)
        elif encoding == "alaw":
            return cls.to_alaw(pcm16)
        else:
            return pcm16
```

### RTP Streaming

```python
import asyncio
import struct
import time

class RTPStreamer:
    """Stream audio via RTP for VoIP integration"""

    PAYLOAD_TYPE_PCMU = 0   # G.711 mu-law
    PAYLOAD_TYPE_PCMA = 8   # G.711 A-law
    PACKET_SIZE = 160       # 20ms @ 8kHz
    PACKET_DURATION_MS = 20

    def __init__(self, ssrc: int = None):
        self.ssrc = ssrc or int(time.time()) & 0xFFFFFFFF
        self.sequence = 0
        self.timestamp = 0

    def build_rtp_packet(
        self,
        payload: bytes,
        payload_type: int = PAYLOAD_TYPE_PCMU,
        marker: bool = False
    ) -> bytes:
        """Build RTP packet with header"""
        # RTP header (12 bytes)
        version = 2
        padding = 0
        extension = 0
        cc = 0

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

        self.sequence += 1
        self.timestamp += self.PACKET_SIZE

        return header + payload

    async def stream_rtp(
        self,
        audio_bytes: bytes,
        dest_host: str,
        dest_port: int,
        payload_type: int = PAYLOAD_TYPE_PCMU
    ):
        """Stream audio as RTP packets"""
        transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
            asyncio.DatagramProtocol,
            remote_addr=(dest_host, dest_port)
        )

        try:
            for i in range(0, len(audio_bytes), self.PACKET_SIZE):
                chunk = audio_bytes[i:i + self.PACKET_SIZE]

                # Pad last packet if needed
                if len(chunk) < self.PACKET_SIZE:
                    chunk = chunk + b'\x7F' * (self.PACKET_SIZE - len(chunk))

                marker = (i == 0)  # Mark first packet
                packet = self.build_rtp_packet(chunk, payload_type, marker)

                transport.sendto(packet)

                # Maintain 20ms timing
                await asyncio.sleep(self.PACKET_DURATION_MS / 1000)
        finally:
            transport.close()
```

## Docker Configuration

### Dockerfile

```dockerfile
# Base image with TensorRT
FROM nvcr.io/nvidia/tensorrt:24.06-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV CHATTERBOX_CFG_SCALE=0.5

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server/ ./server/
COPY configs/ ./configs/
COPY engines/ ./engines/

# Pre-download models (optional - can mount instead)
# RUN python -c "from chatterbox import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained(device='cpu')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  tts:
    build:
      context: .
      dockerfile: docker/Dockerfile
    image: krim-ai/chatterbox-tts:latest
    container_name: chatterbox-tts
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - CHATTERBOX_CFG_SCALE=0.5
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models:ro
      - ./engines:/app/engines:ro
      - ./configs:/app/configs:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: chatterbox-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

  prometheus:
    image: prom/prometheus:latest
    container_name: chatterbox-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    container_name: chatterbox-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### requirements.txt

```
# Core
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
sse-starlette>=1.6.0
python-multipart>=0.0.6

# ML
torch>=2.1.0
chatterbox-tts>=0.1.0
onnxruntime-gpu>=1.16.0
tensorrt>=10.0.0

# Audio
numpy>=1.24.0
scipy>=1.11.0
librosa>=0.10.0
soundfile>=0.12.0

# Caching
redis>=5.0.0
aioredis>=2.0.0

# Monitoring
prometheus-client>=0.18.0

# Utils
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-json-logger>=2.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.25.0
locust>=2.16.0
```

## Scaling Configuration

### For 30,000 Daily Calls

```yaml
# Assumptions:
# - Average call duration: 3 minutes
# - Peak hours: 8 AM - 8 PM (12 hours)
# - Peak factor: 2x average
# - Average TTS per call: 10 utterances
# - Average utterance: 5 seconds audio

# Calculations:
# - Total TTS requests/day: 30,000 * 10 = 300,000
# - Average requests/hour: 300,000 / 24 = 12,500
# - Peak requests/hour: 25,000
# - Peak requests/second: ~7

# GPU Capacity (A10G):
# - RTF target: 0.5
# - Average inference: 400ms
# - Concurrent streams: ~3 per GPU
# - Requests/second/GPU: ~7.5

deployment:
  # Minimum for 30k daily calls
  min_gpus: 1

  # Recommended for redundancy
  recommended_gpus: 2

  # Peak load scaling
  max_gpus: 4

auto_scaling:
  min_instances: 1
  max_instances: 4
  target_cpu_utilization: 70
  target_gpu_utilization: 80
  scale_up_cooldown: 60
  scale_down_cooldown: 300

redis:
  speaker_cache_mb: 512
  embedding_ttl_seconds: 3600
  max_connections: 100

load_balancer:
  algorithm: least_connections
  health_check_path: /health
  health_check_interval: 30
  timeout_ms: 30000
  idle_timeout: 300
```

## Monitoring Setup

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    'tts_requests_total',
    'Total TTS requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'tts_request_latency_ms',
    'TTS request latency in milliseconds',
    ['endpoint'],
    buckets=[100, 200, 300, 400, 500, 750, 1000, 2000, 5000]
)

# Inference metrics
INFERENCE_LATENCY = Histogram(
    'tts_inference_latency_ms',
    'Model inference latency in milliseconds',
    buckets=[50, 100, 200, 300, 400, 500, 750, 1000]
)

RTF = Histogram(
    'tts_rtf',
    'Real-time factor',
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
)

# Resource metrics
GPU_MEMORY = Gauge(
    'tts_gpu_memory_gb',
    'GPU memory usage in GB'
)

ACTIVE_REQUESTS = Gauge(
    'tts_active_requests',
    'Number of active TTS requests'
)
```

### CloudWatch Alarms

```yaml
alarms:
  - name: HighLatency
    metric: tts_request_latency_ms
    threshold: 500
    period: 60
    evaluation_periods: 3
    comparison: GreaterThanThreshold
    action: notify

  - name: HighErrorRate
    metric: tts_requests_total{status="error"}
    threshold: 5
    period: 60
    evaluation_periods: 2
    comparison: GreaterThanThreshold
    action: notify

  - name: HighGPUMemory
    metric: tts_gpu_memory_gb
    threshold: 20
    period: 300
    evaluation_periods: 2
    comparison: GreaterThanThreshold
    action: scale_up

  - name: HealthCheckFailed
    metric: HealthCheckStatus
    threshold: 0
    period: 60
    evaluation_periods: 2
    comparison: LessThanOrEqualToThreshold
    action: notify_critical
```

## Deployment Checklist

### Pre-Deployment

- [ ] Docker image builds successfully
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Load tests pass (200 concurrent)
- [ ] Models and engines included/mounted
- [ ] Environment variables configured
- [ ] Redis connectivity verified

### Deployment

- [ ] Container starts without errors
- [ ] Health endpoint returns "healthy"
- [ ] GPU accessible inside container
- [ ] TensorRT engines load correctly
- [ ] Latency meets targets under load
- [ ] Telephony codec conversion works
- [ ] Streaming endpoint works

### Post-Deployment

- [ ] Monitoring dashboards configured
- [ ] Alerts configured and tested
- [ ] Logging to CloudWatch/ELK
- [ ] Backup and recovery tested
- [ ] Documentation updated
- [ ] Runbook created

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs chatterbox-tts

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### High Latency
```bash
# Check GPU utilization
nvidia-smi -l 1

# Check for memory pressure
watch -n1 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv'

# Profile inference
python -c "import torch; torch.cuda.synchronize(); ..."
```

### Redis Connection Issues
```bash
# Test Redis connectivity
redis-cli -h redis ping

# Check Redis memory
redis-cli info memory
```

### Audio Quality Issues
```bash
# Test audio output
curl -X POST http://localhost:8080/v1/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text": "Test audio quality"}' \
    -o test.wav

# Play and verify
ffplay test.wav
```

## Next Steps

After deployment is complete:
1. Monitor production metrics
2. Gather user feedback
3. Plan for scale-out if needed
4. Document operational procedures
