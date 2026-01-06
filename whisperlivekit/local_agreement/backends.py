import io
import logging
import math
import sys
from typing import List

import numpy as np
import soundfile as sf

from whisperlivekit.model_paths import detect_model_format, resolve_model_path
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper.transcribe import transcribe as whisper_transcribe

logger = logging.getLogger(__name__)
class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, model_size=None, cache_dir=None, model_dir=None, lora_path=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.lora_path = lora_path
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(model_size, cache_dir, model_dir)

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(self.start + offset, self.end + offset, self.text)

    def __repr__(self):
        return f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"

    def load_model(self, model_size, cache_dir, model_dir):
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class WhisperASR(ASRBase):
    """Uses WhisperLiveKit's built-in Whisper implementation."""
    sep = " "

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from whisperlivekit.whisper import load_model as load_whisper_model

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)            
            if resolved_path.is_dir():
                model_info = detect_model_format(resolved_path)
                if not model_info.has_pytorch:
                    raise FileNotFoundError(
                        f"No supported PyTorch checkpoint found under {resolved_path}"
                    )            
            logger.debug(f"Loading Whisper model from custom path {resolved_path}")
            return load_whisper_model(str(resolved_path), lora_path=self.lora_path)

        if model_size is None:
            raise ValueError("Either model_size or model_dir must be set for WhisperASR")

        return load_whisper_model(model_size, download_root=cache_dir, lora_path=self.lora_path)

    def transcribe(self, audio, init_prompt=""):
        options = dict(self.transcribe_kargs)
        options.pop("vad", None)
        options.pop("vad_filter", None)
        language = self.original_language if self.original_language else None

        result = whisper_transcribe(
            self.model,
            audio,
            language=language,
            initial_prompt=init_prompt,
            condition_on_previous_text=True,
            word_timestamps=True,
            **options,
        )
        return result

    def ts_words(self, r) -> List[ASRToken]:
        """
        Converts the Whisper result to a list of ASRToken objects.
        """
        tokens = []
        for segment in r["segments"]:
            for word in segment["words"]:
                token = ASRToken(
                    word["start"],
                    word["end"],
                    word["word"],
                    probability=word.get("probability"),
                )
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [segment["end"] for segment in res["segments"]]

    def use_vad(self):
        logger.warning("VAD is not currently supported for WhisperASR backend and will be ignored.")

class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading faster-whisper model from {resolved_path}. "
                         f"model_size and cache_dir parameters are not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("Either model_size or model_dir must be set")
        device = "auto" # Allow CTranslate2 to decide available device
        compute_type = "auto" # Allow CTranslate2 to decide faster compute type
                              

        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper optimized for Apple Silicon.
    """
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder, transcribe

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading MLX Whisper model from {resolved_path}. model_size parameter is not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = self.translate_model_name(model_size)
            logger.debug(f"Loading whisper model {model_size}. You use mlx whisper, so {model_size_or_path} will be used.")
        else:
            raise ValueError("Either model_size or model_dir must be set")

        self.model_size_or_path = model_size_or_path
        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }
        mlx_model_path = model_mapping.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
        )
        return segments.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                probability=word["probability"]
                token = ASRToken(word["start"], word["end"], word["word"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for transcription."""
    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.direct_english_translation = False

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments) -> List[ASRToken]:
        """
        Converts OpenAI API response words into ASRToken objects while
        optionally skipping words that fall into no-speech segments.
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment.no_speech_prob > 0.8:
                    no_speech_segments.append((segment.start, segment.end))
        tokens = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            tokens.append(ASRToken(start, end, word.word))
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if not self.direct_english_translation and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt
        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True


class WhisperCppASR(ASRBase):
    """Uses whisper-cpp server for transcription.
    
    This backend connects to a whisper-cpp server running locally,
    typically compiled with OpenVINO support for Intel Arc GPU acceleration.
    
    The server should be started with:
        whisper-server.exe -m ggml-large-v3-turbo.bin --port 8080
    
    Args:
        lan: Language code (e.g., 'en', 'auto')
        endpoint_url: URL of whisper-cpp server (default: http://localhost:8080)
        temperature: Sampling temperature (default: 0.0)
    """
    sep = " "  # whisper-cpp returns clean words without leading spaces, so we need a space separator
    
    def __init__(self, lan=None, endpoint_url=None, temperature=0.0, logfile=sys.stderr):
        import os
        self.logfile = logfile
        self.original_language = None if lan == "auto" else lan
        self.endpoint_url = (endpoint_url or os.environ.get("WHISPER_CPP_URL", "http://localhost:8080")).rstrip('/')
        self.inference_endpoint = f"{self.endpoint_url}/inference"
        self.temperature = temperature
        self.response_format = "verbose_json"
        self.use_vad_opt = False
        self.direct_english_translation = False
        self.transcribed_seconds = 0
        self.load_model()
    
    def load_model(self, *args, **kwargs):
        """Validate connection to whisper-cpp server."""
        import requests
        try:
            logger.info(f"Connecting to whisper-cpp server at {self.endpoint_url}...")
            # Try health endpoint first, fall back to just checking if server responds
            try:
                response = requests.get(f"{self.endpoint_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"Successfully connected to whisper-cpp server")
                    return None
            except:
                # Some whisper-cpp builds don't have /health, try /inference with OPTIONS
                pass
            
            # If health check fails, we'll just proceed and let transcribe fail if server is down
            logger.warning(f"Could not verify whisper-cpp server health, will attempt transcription anyway")
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to whisper-cpp at {self.endpoint_url}")
            logger.error("Make sure whisper-cpp server is running:")
            logger.error("  whisper-server.exe -m ggml-large-v3-turbo.bin --port 8080")
            raise
        except Exception as e:
            logger.warning(f"Connection test warning: {e}")
        
        return None
    
    def transcribe(self, audio_data, init_prompt="", *args, **kwargs):
        """Transcribe audio using whisper-cpp server.
        
        Args:
            audio_data: Audio array (16kHz, mono, float32)
            init_prompt: Optional transcription context/prompt
        
        Returns:
            List of segment dicts with: start, end, text, no_speech_prob
        """
        import requests
        import json
        
        # Ensure audio is float32
        if not isinstance(audio_data, np.ndarray):
            raise TypeError(f"audio_data must be numpy array, got {type(audio_data)}")
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Convert to WAV in memory
        wav_buffer = io.BytesIO()
        wav_buffer.name = "audio.wav"
        
        try:
            sf.write(wav_buffer, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')
        except Exception as e:
            raise RuntimeError(f"Failed to encode audio: {e}")
        
        wav_buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        
        # Prepare multipart form data for whisper-cpp /inference endpoint
        files = {
            'file': ('audio.wav', wav_buffer, 'audio/wav')
        }
        
        data = {
            'response_format': self.response_format,
            'temperature': str(self.temperature),
        }
        
        # Add language if specified
        if self.original_language:
            data['language'] = self.original_language
        
        # Add prompt if provided
        if init_prompt:
            data['prompt'] = init_prompt
        
        # Make request to whisper-cpp
        try:
            logger.debug(f"Sending request to {self.inference_endpoint}")
            response = requests.post(
                self.inference_endpoint,
                files=files,
                data=data,
                timeout=300  # 5 minute timeout for long audio
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError("Request timeout - whisper-cpp took too long")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to whisper-cpp at {self.endpoint_url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"whisper-cpp error: {e.response.status_code} - {e.response.text}")
        
        try:
            result = response.json()
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            return [{
                'start': 0.0,
                'end': float(len(audio_data) / 16000),
                'text': response.text,
                'no_speech_prob': 0.0
            }]
        
        logger.debug(f"whisper-cpp processed accumulated {self.transcribed_seconds} seconds")
        return self._parse_response(result, len(audio_data) / 16000)
    
    def _parse_response(self, response, audio_duration):
        """Parse whisper-cpp response into standard segment format."""
        segments = []
        
        # Handle different response formats from whisper-cpp
        if isinstance(response, dict):
            if 'segments' in response:
                segments = response['segments']
            elif 'text' in response:
                # Simple text response without segments
                return [{
                    'start': 0.0,
                    'end': audio_duration,
                    'text': response['text'],
                    'no_speech_prob': 0.0
                }]
        elif isinstance(response, list):
            segments = response
        
        # Normalize segment format
        normalized = []
        for seg in segments:
            normalized.append({
                'start': seg.get('t0', seg.get('start', 0.0)) / 1000.0 if seg.get('t0') else seg.get('start', 0.0),
                'end': seg.get('t1', seg.get('end', audio_duration)) / 1000.0 if seg.get('t1') else seg.get('end', audio_duration),
                'text': seg.get('text', ''),
                'no_speech_prob': seg.get('no_speech_prob', 0.0)
            })
        
        return normalized if normalized else [{'start': 0.0, 'end': audio_duration, 'text': '', 'no_speech_prob': 1.0}]
    
    def ts_words(self, segments) -> List[ASRToken]:
        """Convert segments to word-level ASRToken objects.
        
        Since whisper-cpp may not provide word-level timestamps,
        we estimate word timings by splitting segment text.
        """
        tokens = []
        
        for segment in segments:
            if segment.get('no_speech_prob', 0.0) > 0.9:
                continue
            
            text = segment.get('text', '').strip()
            if not text:
                continue
            
            # Check if segment has word-level timestamps
            if 'words' in segment:
                for word in segment['words']:
                    start = word.get('t0', word.get('start', 0.0))
                    end = word.get('t1', word.get('end', start + 0.1))
                    # Convert from ms to seconds if needed
                    if start > 1000:
                        start /= 1000.0
                        end /= 1000.0
                    tokens.append(ASRToken(start, end, word.get('word', word.get('text', ''))))
            else:
                # Estimate word timings from segment
                words = [w.strip() for w in text.split() if w.strip()]
                if not words:
                    continue
                
                seg_start = segment.get('start', 0.0)
                seg_end = segment.get('end', seg_start + 1.0)
                seg_duration = seg_end - seg_start
                
                if seg_duration <= 0:
                    seg_duration = 1.0
                
                duration_per_word = seg_duration / len(words)
                current_time = seg_start
                
                for word in words:
                    word_end = current_time + duration_per_word
                    tokens.append(ASRToken(current_time, word_end, word))
                    current_time = word_end
        
        return tokens
    
    def segments_end_ts(self, segments) -> List[float]:
        """Extract end timestamps from segments."""
        return [s.get('end', 0.0) for s in segments]
    
    def use_vad(self):
        """Enable VAD filtering."""
        self.use_vad_opt = True
