# WhisperLiveKit + whisper-cpp Integration Guide

This guide explains how to use WhisperLiveKit with a local whisper-cpp server running on Intel Arc GPU with OpenVINO support.

## Overview

WhisperLiveKit now includes a dedicated `whisper-cpp` backend that connects directly to a whisper-cpp server's `/inference` endpoint. This enables:

- **Privacy**: All transcription happens locally
- **Cost savings**: No API fees
- **Hardware acceleration**: Use Intel Arc GPU with OpenVINO for fast inference

## Prerequisites

1. **whisper-cpp server** compiled with OpenVINO support
2. **OpenVINO toolkit** installed and configured
3. **WhisperLiveKit** installed from this modified source

## Setup whisper-cpp Server

### 1. Set up OpenVINO environment

```batch
C:\data\OpenVINO\w_openvino_toolkit_windows_2024.6.0.17404.4c0f47d2335_x86_64\w_openvino_toolkit_windows_2024.6.0.17404.4c0f47d2335_x86_64\setupvars.bat
```

### 2. Start the whisper-cpp server

```batch
cd C:\path\to\whisper-cpp
whisper-server.exe -m ggml-large-v3-turbo.bin --port 8080
```

The server will start and listen on `http://localhost:8080`.

## Using WhisperLiveKit with whisper-cpp

### Method 1: Command Line Arguments

```bash
# Start WhisperLiveKit with local whisper-cpp server
wlk --backend whisper-cpp \
    --backend-policy localagreement \
    --whisper-cpp-url http://localhost:8080 \
    --lan en
```

### Method 2: Environment Variables

```bash
# Set environment variable
set WHISPER_CPP_URL=http://localhost:8080

# Start WhisperLiveKit
wlk --backend whisper-cpp --backend-policy localagreement --lan en
```

### Method 3: Python API

```python
from whisperlivekit import TranscriptionEngine

# Initialize with local whisper-cpp server
engine = TranscriptionEngine(
    backend="whisper-cpp",
    backend_policy="localagreement",
    lan="en",
    whisper_cpp_url="http://localhost:8080"
)
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--backend` | Must be set to `whisper-cpp` | auto |
| `--backend-policy` | Must be set to `localagreement` | simulstreaming |
| `--whisper-cpp-url` | URL of the whisper-cpp server | http://localhost:8080 |
| `--lan` | Language code (e.g., 'en', 'auto') | auto |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `WHISPER_CPP_URL` | Alternative to `--whisper-cpp-url` |

## API Endpoint Details

The whisper-cpp backend uses the `/inference` endpoint with multipart form data:

- **Endpoint**: `POST http://localhost:8080/inference`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: WAV audio file (16kHz, mono, PCM_16)
  - `language`: Language code (optional)
  - `response_format`: `verbose_json` for detailed output
  - `temperature`: Sampling temperature (default: 0.0)
  - `prompt`: Initial prompt for context (optional)

## Troubleshooting

### Connection refused
- Ensure whisper-cpp server is running on the specified port
- Check that OpenVINO environment is set up before starting the server
- Verify the URL doesn't have a trailing slash

### Slow transcription
- Verify Intel Arc GPU is being used (check OpenVINO logs)
- Consider using a smaller model (e.g., `ggml-base.bin`)

### No transcription output
- Check whisper-cpp server logs for errors
- Ensure audio is being captured correctly
- Try with `--lan en` instead of `--lan auto`

### Response format errors
- The backend handles multiple response formats from whisper-cpp
- If you see parsing errors, check the whisper-cpp version

## Architecture

```
┌─────────────────────┐     WebSocket      ┌──────────────────────┐
│   Browser/Client    │ ◄──────────────────► │   WhisperLiveKit     │
│   (Audio Stream)    │                      │   (FastAPI Server)   │
└─────────────────────┘                      └──────────┬───────────┘
                                                        │
                                                        │ HTTP POST
                                                        │ /inference
                                                        ▼
                                             ┌──────────────────────┐
                                             │   whisper-cpp        │
                                             │   (OpenVINO/Arc GPU) │
                                             │   localhost:8080     │
                                             └──────────────────────┘
```

## Performance Notes

- The whisper-cpp server with OpenVINO on Intel Arc GPU provides significantly faster inference than CPU-only processing
- For real-time streaming, use smaller models (tiny, base, small) for lower latency
- The `large-v3-turbo` model offers a good balance of accuracy and speed
- The backend automatically handles WAV conversion and response parsing

## Differences from OpenAI API Backend

| Feature | OpenAI API | whisper-cpp |
|---------|------------|-------------|
| Endpoint | `/v1/audio/transcriptions` | `/inference` |
| Authentication | API key required | No authentication |
| Response format | OpenAI format | whisper-cpp format |
| Word timestamps | Native support | Estimated from segments |
| Cost | Per-minute pricing | Free (local) |
