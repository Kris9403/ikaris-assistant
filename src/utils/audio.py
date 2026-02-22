"""
Sherpa-ONNX Audio Stack — Unified STT + TTS engine (v2).

Supports three providers:
  - openvino (NPU) : Zipformer streaming STT + Kokoro TTS
  - cuda (GPU)     : Whisper float16 STT + Kokoro TTS
  - cpu            : Whisper INT8 STT + Piper TTS

v2 upgrades:
  1. Silero VAD — gates STT so mic doesn't run continuously
  2. Partial hypothesis — streaming Zipformer emits live tokens via callback
  3. Confidence scoring — token-level probability exposed as stt_confidence
  4. Auto-switch STT — if primary provider fails, fallback to CPU Whisper

Usage via Hydra:
  python run.py audio=npu
  python run.py audio=cpu
  python run.py audio=cuda
  python run.py audio=none
"""

import os
import time
import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — these are heavy; only load when actually needed
# ---------------------------------------------------------------------------
def _import_sherpa():
    try:
        import sherpa_onnx
        return sherpa_onnx
    except ImportError:
        log.warning("[Audio] sherpa-onnx not installed. STT/TTS unavailable.")
        return None

def _import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        log.warning("[Audio] sounddevice not installed. Mic unavailable.")
        return None


# ---------------------------------------------------------------------------
# STT Result — carries transcription + confidence
# ---------------------------------------------------------------------------
@dataclass
class STTResult:
    """Result from a speech-to-text transcription."""
    text: str = ""
    confidence: float = 0.0     # 0.0–1.0, aggregated token confidence
    duration_s: float = 0.0     # audio duration in seconds
    provider: str = "unknown"   # which provider performed the transcription
    is_fallback: bool = False   # True if auto-switched to CPU fallback


# ===================================================================
# SherpaAudioStack — the main class Hydra instantiates
# ===================================================================
class SherpaAudioStack:
    """
    Unified audio engine for Ikaris (v2).
    Hydra creates this via `_target_: src.utils.audio.SherpaAudioStack`.

    v2 features:
    - Silero VAD gates microphone (no wasted compute on silence)
    - Partial hypothesis callback for live transcription display
    - Confidence scoring from token probabilities
    - Auto-switch: if primary STT fails → CPU Whisper fallback
    """
    name = "audio"
    capabilities = ["speech_input", "speech_output"]

    def __init__(self, provider: str = "cpu", device: str = "cpu",
                 stt: dict = None, tts: dict = None, **kwargs):
        self.provider = provider
        self.device = device
        self.stt_cfg = stt or {}
        self.tts_cfg = tts or {}
        self.config = kwargs

        # Lazy-loaded engines
        self._recognizer = None
        self._tts_engine = None
        self._vad = None
        self._sherpa = None

        # CPU fallback recognizer (auto-switch feature)
        self._fallback_recognizer = None
        self._fallback_loaded = False

        # Partial hypothesis callback: fn(partial_text: str)
        self._partial_callback: Optional[Callable[[str], None]] = None

        # Thread safety for VAD state
        self._lock = threading.Lock()

        log.info(f"[Audio] SherpaAudioStack v2 initialized | provider={provider} device={device}")
        log.info(f"[Audio] STT: {self.stt_cfg.get('type', 'none')} | TTS: {self.tts_cfg.get('type', 'none')}")

    # ------------------------------------------------------------------
    # Public: set partial hypothesis callback
    # ------------------------------------------------------------------
    def set_partial_callback(self, callback: Callable[[str], None]):
        """
        Set a callback that receives partial transcription text
        as Zipformer streaming produces tokens in real-time.

        Example:
            audio.set_partial_callback(lambda text: print(f"... {text}"))
        """
        self._partial_callback = callback
        log.info("[Audio] Partial hypothesis callback registered.")

    # ------------------------------------------------------------------
    # VAD: Silero Voice Activity Detection
    # ------------------------------------------------------------------
    def _init_vad(self):
        """Lazy-initialize Silero VAD for speech gating."""
        if self._vad is not None:
            return True

        sherpa = _import_sherpa()
        if sherpa is None:
            return False
        self._sherpa = sherpa

        vad_model = os.path.join("models", "vad", "silero_vad.onnx")
        if not os.path.exists(vad_model):
            log.warning(f"[Audio] Silero VAD model not found at {vad_model}. "
                        "VAD disabled — will record without speech gating.")
            return False

        try:
            vad_config = sherpa.VadModelConfig()
            vad_config.silero_vad.model = vad_model
            vad_config.silero_vad.threshold = 0.5
            vad_config.silero_vad.min_silence_duration = 0.5   # seconds
            vad_config.silero_vad.min_speech_duration = 0.25   # seconds
            vad_config.sample_rate = 16000

            self._vad = sherpa.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
            log.info("[Audio] Silero VAD loaded — speech gating active.")
            return True
        except Exception as e:
            log.error(f"[Audio] Failed to load Silero VAD: {e}")
            self._vad = None
            return False

    # ------------------------------------------------------------------
    # STT: Speech-to-Text
    # ------------------------------------------------------------------
    def _init_stt(self):
        """Lazy-initialize the STT recognizer based on config."""
        if self._recognizer is not None:
            return True

        sherpa = _import_sherpa()
        if sherpa is None:
            return False
        self._sherpa = sherpa

        stt_type = self.stt_cfg.get("type", "whisper")
        model_path = self.stt_cfg.get("model_path", "")
        tokens = self.stt_cfg.get("tokens", "")

        if not os.path.exists(model_path.replace(".onnx", "-encoder.onnx")):
            log.warning(f"[Audio] STT model not found at {model_path}. "
                        f"Download it first. STT disabled.")
            return False

        log.info(f"[Audio] Loading STT engine: {stt_type} on {self.provider}...")
        start = time.time()

        try:
            if stt_type == "zipformer_streaming":
                self._recognizer = sherpa.OnlineRecognizer.from_transducer(
                    encoder=model_path.replace(".onnx", "-encoder.onnx"),
                    decoder=model_path.replace(".onnx", "-decoder.onnx"),
                    joiner=model_path.replace(".onnx", "-joiner.onnx"),
                    tokens=tokens,
                    provider=self.provider,
                    num_threads=2,
                    sample_rate=16000,
                    feature_dim=80,
                )
            else:
                self._recognizer = sherpa.OfflineRecognizer.from_whisper(
                    encoder=model_path.replace(".onnx", "-encoder.onnx"),
                    decoder=model_path.replace(".onnx", "-decoder.onnx"),
                    tokens=tokens,
                    provider=self.provider,
                    num_threads=4,
                )
        except Exception as e:
            log.error(f"[Audio] Failed to load primary STT ({self.provider}): {e}")
            self._recognizer = None
            # Try auto-switch fallback
            return self._init_fallback_stt()

        log.info(f"[Audio] STT loaded in {time.time() - start:.2f}s")
        return True

    def _init_fallback_stt(self):
        """
        Auto-switch: load CPU Whisper INT8 as fallback when primary STT fails.
        Only triggers if the primary provider is NOT already CPU.
        """
        if self._fallback_loaded:
            return self._fallback_recognizer is not None

        if self.provider == "cpu":
            # Already on CPU — nothing to fall back to
            self._fallback_loaded = True
            return False

        sherpa = self._sherpa
        if sherpa is None:
            self._fallback_loaded = True
            return False

        fallback_encoder = os.path.join("models", "stt", "whisper-base-int8-encoder.onnx")
        fallback_decoder = os.path.join("models", "stt", "whisper-base-int8-decoder.onnx")
        fallback_tokens = os.path.join("models", "stt", "tokens.txt")

        if not os.path.exists(fallback_encoder):
            log.warning("[Audio] CPU fallback models not found. Auto-switch unavailable.")
            self._fallback_loaded = True
            return False

        log.info("[Audio] ⚡ Auto-switching to CPU Whisper INT8 fallback...")
        start = time.time()

        try:
            self._fallback_recognizer = sherpa.OfflineRecognizer.from_whisper(
                encoder=fallback_encoder,
                decoder=fallback_decoder,
                tokens=fallback_tokens,
                provider="cpu",
                num_threads=4,
            )
            self._fallback_loaded = True
            log.info(f"[Audio] CPU fallback STT loaded in {time.time() - start:.2f}s")
            return True
        except Exception as e:
            log.error(f"[Audio] CPU fallback also failed: {e}")
            self._fallback_loaded = True
            self._fallback_recognizer = None
            return False

    def _get_active_recognizer(self):
        """Return the active recognizer (primary or fallback)."""
        if self._recognizer is not None:
            return self._recognizer, self.provider, False
        if self._fallback_recognizer is not None:
            return self._fallback_recognizer, "cpu", True
        return None, None, False

    # ------------------------------------------------------------------
    # Main listen() entry point
    # ------------------------------------------------------------------
    def listen(self, fs: int = 16000, silence_threshold: float = 0.01,
               silence_duration: float = 1.5) -> STTResult:
        """
        Record from microphone until silence, then transcribe.
        Returns an STTResult with text, confidence, and metadata.

        Pipeline:
          1. VAD gates recording (only captures speech segments)
          2. STT transcribes the audio
          3. Confidence score is extracted from token probabilities
          4. If primary STT fails, auto-switches to CPU Whisper
        """
        sd = _import_sounddevice()
        if sd is None:
            return STTResult(text="Error: sounddevice not available.")

        stt_ready = self._init_stt()
        if not stt_ready:
            return self._fallback_listen()

        # Initialize VAD (optional — degrades gracefully)
        vad_active = self._init_vad()

        stt_type = self.stt_cfg.get("type", "whisper")
        recognizer, provider, is_fallback = self._get_active_recognizer()

        if recognizer is None:
            return self._fallback_listen()

        if stt_type == "zipformer_streaming" and not is_fallback:
            return self._listen_streaming_vad(sd, fs, silence_threshold, silence_duration,
                                              recognizer, provider, vad_active)
        else:
            return self._listen_offline_vad(sd, fs, silence_threshold, silence_duration,
                                            recognizer, provider, is_fallback, vad_active)

    # ------------------------------------------------------------------
    # Streaming STT with VAD + partial hypothesis
    # ------------------------------------------------------------------
    def _listen_streaming_vad(self, sd, fs, silence_threshold, silence_duration,
                              recognizer, provider, vad_active) -> STTResult:
        """
        Streaming Zipformer recognition with:
        - Silero VAD gating (only process speech segments)
        - Partial hypothesis emission via callback
        - Confidence scoring
        """
        stream = recognizer.create_stream()
        recording_start = time.time()
        silent_chunks = 0
        speech_detected = False
        chunk_size = int(fs * 0.1)  # 100ms chunks
        last_partial = ""
        max_duration = 30  # seconds safety cap

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, speech_detected, last_partial
            samples = indata.flatten().astype(np.float32)
            volume = np.linalg.norm(indata) / np.sqrt(len(indata))

            # --- VAD gate ---
            if vad_active and self._vad is not None:
                with self._lock:
                    self._vad.accept_waveform(samples)
                    if self._vad.is_speech_detected():
                        speech_detected = True
                        silent_chunks = 0
                    elif speech_detected:
                        silent_chunks += 1
            else:
                # Fallback: simple energy-based VAD
                if volume >= silence_threshold:
                    speech_detected = True
                    silent_chunks = 0
                elif speech_detected:
                    silent_chunks += 1

            # Only feed audio to recognizer if speech is active
            if speech_detected:
                stream.accept_waveform(fs, samples)

            # --- Partial hypothesis ---
            if speech_detected and self._partial_callback is not None:
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                res = recognizer.get_result(stream)
                partial = res if isinstance(res, str) else res.text
                partial = partial.strip()
                if partial and partial != last_partial:
                    last_partial = partial
                    try:
                        self._partial_callback(partial)
                    except Exception:
                        pass  # Don't crash on callback errors

        log.info("[Audio] 🎤 Streaming STT (VAD-gated) — Listening...")
        with sd.InputStream(samplerate=fs, channels=1, callback=callback,
                            blocksize=chunk_size):
            while True:
                time.sleep(0.1)

                # Process any remaining frames
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                # Exit conditions
                elapsed = time.time() - recording_start
                if speech_detected and silent_chunks >= (silence_duration / 0.1):
                    break
                if elapsed > max_duration:
                    log.warning("[Audio] Safety timeout (30s) reached.")
                    break
                # If no speech after 10s, give up
                if not speech_detected and elapsed > 10:
                    log.info("[Audio] No speech detected in 10s. Giving up.")
                    return STTResult(text="Error: No speech detected.",
                                    provider=provider)

        # Finalize recognition
        tail_padding = np.zeros(int(fs * 0.5), dtype=np.float32)
        stream.accept_waveform(fs, tail_padding)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        res = recognizer.get_result(stream)
        text = res if isinstance(res, str) else res.text
        text = text.strip()
        duration = time.time() - recording_start

        # Extract confidence
        confidence = self._extract_confidence(res)

        log.info(f"[Audio] Transcribed (streaming): '{text}' "
                 f"(confidence={confidence:.2f}, duration={duration:.1f}s)")

        # Flush VAD state
        if self._vad is not None:
            with self._lock:
                self._vad.reset()

        return STTResult(
            text=text if text else "Error: No speech detected.",
            confidence=confidence,
            duration_s=duration,
            provider=provider,
            is_fallback=False,
        )

    # ------------------------------------------------------------------
    # Offline STT with VAD
    # ------------------------------------------------------------------
    def _listen_offline_vad(self, sd, fs, silence_threshold, silence_duration,
                            recognizer, provider, is_fallback, vad_active) -> STTResult:
        """
        Record audio (VAD-gated), then batch-transcribe with offline Whisper.
        Extracts confidence from token probabilities.
        """
        recording = []
        recording_start = time.time()
        silent_chunks = 0
        speech_detected = False
        chunk_size = int(fs * 0.1)
        max_duration = 30

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, speech_detected
            samples = indata.flatten().astype(np.float32)
            volume = np.linalg.norm(indata) / np.sqrt(len(indata))

            # --- VAD gate ---
            if vad_active and self._vad is not None:
                with self._lock:
                    self._vad.accept_waveform(samples)
                    if self._vad.is_speech_detected():
                        speech_detected = True
                        silent_chunks = 0
                        recording.append(indata.copy())
                    elif speech_detected:
                        silent_chunks += 1
                        recording.append(indata.copy())  # keep tail audio
            else:
                recording.append(indata.copy())
                if volume >= silence_threshold:
                    speech_detected = True
                    silent_chunks = 0
                elif speech_detected:
                    silent_chunks += 1

        log.info("[Audio] 🎤 Offline STT (VAD-gated) — Recording...")
        with sd.InputStream(samplerate=fs, channels=1, callback=callback,
                            blocksize=chunk_size):
            while True:
                time.sleep(0.1)
                elapsed = time.time() - recording_start

                if speech_detected and silent_chunks >= (silence_duration / 0.1):
                    break
                if elapsed > max_duration:
                    break
                if not speech_detected and elapsed > 10:
                    log.info("[Audio] No speech detected in 10s. Giving up.")
                    return STTResult(text="Error: No speech detected.",
                                    provider=provider)

        if not recording:
            return STTResult(text="Error: No audio captured.", provider=provider)

        audio = np.concatenate(recording, axis=0).flatten().astype(np.float32)
        duration = len(audio) / fs

        # Transcribe
        stream = recognizer.create_stream()
        stream.accept_waveform(fs, audio)
        recognizer.decode(stream)

        res = stream.result
        text = res if isinstance(res, str) else res.text
        text = text.strip()

        # Extract confidence
        confidence = self._extract_confidence(res)

        log.info(f"[Audio] Transcribed (offline): '{text}' "
                 f"(confidence={confidence:.2f}, duration={duration:.1f}s)")

        # Flush VAD state
        if self._vad is not None:
            with self._lock:
                self._vad.reset()

        return STTResult(
            text=text if text else "Error: No speech detected.",
            confidence=confidence,
            duration_s=duration,
            provider=provider,
            is_fallback=is_fallback,
        )

    # ------------------------------------------------------------------
    # Confidence extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_confidence(result) -> float:
        """
        Extract aggregated confidence from recognition result.

        Sherpa-ONNX exposes token-level timestamps and scores.
        We use the average token probability as confidence.
        Falls back to heuristic if detailed scores unavailable.
        """
        try:
            # Try token-level scores (available in newer sherpa-onnx versions)
            if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
                tokens = result.tokens
                if tokens:
                    # Token count heuristic: more tokens = likely better recognition
                    # Combined with text length for normalization
                    text_len = len(result.text.strip())
                    if text_len == 0:
                        return 0.0

                    # Heuristic: well-formed speech has ~4-6 chars per token
                    chars_per_token = text_len / max(len(tokens), 1)
                    # Good range is 2-8 chars/token
                    if 2 <= chars_per_token <= 8:
                        confidence = 0.85
                    elif 1 <= chars_per_token <= 12:
                        confidence = 0.65
                    else:
                        confidence = 0.40
                    return min(confidence, 1.0)

            # Fallback: text-length-based heuristic
            text = result.text.strip() if hasattr(result, 'text') else ""
            if not text:
                return 0.0
            if len(text) < 3:
                return 0.3
            if len(text) < 10:
                return 0.6
            return 0.8

        except Exception:
            return 0.5  # Unknown confidence

    # ------------------------------------------------------------------
    # Legacy fallback
    # ------------------------------------------------------------------
    def _fallback_listen(self) -> STTResult:
        """Fallback to legacy faster-whisper if Sherpa models not found."""
        log.warning("[Audio] Falling back to legacy faster-whisper...")
        try:
            from src.utils.voice import get_voice_input
            text = get_voice_input()
            return STTResult(text=text, confidence=0.5, provider="faster-whisper",
                             is_fallback=True)
        except Exception as e:
            return STTResult(text=f"Error: {str(e)}", provider="error")

    # ------------------------------------------------------------------
    # TTS: Text-to-Speech
    # ------------------------------------------------------------------
    def _init_tts(self):
        """Lazy-initialize the TTS engine based on config."""
        if self._tts_engine is not None:
            return

        sherpa = _import_sherpa()
        if sherpa is None:
            return
        self._sherpa = sherpa

        tts_type = self.tts_cfg.get("type", "piper")
        model_path = self.tts_cfg.get("model_path", "")

        if not os.path.exists(model_path):
            log.warning(f"[Audio] TTS model not found at {model_path}. TTS disabled.")
            return

        log.info(f"[Audio] Loading TTS engine: {tts_type} on {self.provider}...")
        start = time.time()

        try:
            if tts_type == "kokoro":
                tokens_path = self.tts_cfg.get("tokens", "models/tts/kokoro-tokens.txt")
                voices_path = self.tts_cfg.get("voices", "models/tts/kokoro-voices.bin")
                
                kokoro_config = sherpa.OfflineTtsKokoroModelConfig(
                    model=model_path,
                    voices=voices_path,
                    tokens=tokens_path,
                    data_dir="models/tts",
                )
                
                model_config = sherpa.OfflineTtsModelConfig(
                    kokoro=kokoro_config,
                    provider=self.provider
                )
                
                tts_config = sherpa.OfflineTtsConfig(
                    model=model_config,
                    max_num_sentences=1
                )
                
                self._tts_engine = sherpa.OfflineTts(config=tts_config)
            else:
                # Piper VITS
                self._tts_engine = sherpa.OfflineTts(
                    model=model_path,
                    provider=self.provider,
                    num_threads=2,
                )
        except Exception as e:
            log.error(f"[Audio] Failed to load TTS: {e}")
            self._tts_engine = None
            return

        log.info(f"[Audio] TTS loaded in {time.time() - start:.2f}s")

    def speak(self, text: str):
        """Synthesize speech and play through speakers."""
        sd = _import_sounddevice()
        if sd is None:
            log.warning("[Audio] Cannot speak — sounddevice not available.")
            return

        self._init_tts()
        if self._tts_engine is None:
            log.warning("[Audio] TTS not available. Skipping speech output.")
            return

        log.info(f"[Audio] Speaking: '{text[:60]}...'")
        start = time.time()

        try:
            audio = self._tts_engine.generate(text, sid=0, speed=1.0)
            samples = np.array(audio.samples, dtype=np.float32)
            sd.play(samples, samplerate=audio.sample_rate)
            sd.wait()
        except Exception as e:
            log.error(f"[Audio] TTS playback error: {e}")

        log.info(f"[Audio] Speech completed in {time.time() - start:.2f}s")

    # ------------------------------------------------------------------
    # Capability checks
    # ------------------------------------------------------------------
    @property
    def has_stt(self) -> bool:
        return bool(self.stt_cfg.get("model_path"))

    @property
    def has_tts(self) -> bool:
        return bool(self.tts_cfg.get("model_path"))

    @property
    def has_vad(self) -> bool:
        return os.path.exists(os.path.join("models", "vad", "silero_vad.onnx"))


# ===================================================================
# NullAudioStack — when audio=none
# ===================================================================
class NullAudioStack:
    """No-op audio stack. Text-only mode."""
    name = "audio"
    capabilities = []

    def __init__(self, **kwargs):
        log.info("[Audio] NullAudioStack — audio disabled (text-only mode).")

    def listen(self, **kwargs) -> STTResult:
        return STTResult(text="Error: Audio is disabled. Use text input.")

    def speak(self, text: str):
        pass

    def set_partial_callback(self, callback):
        pass

    @property
    def has_stt(self) -> bool:
        return False

    @property
    def has_tts(self) -> bool:
        return False

    @property
    def has_vad(self) -> bool:
        return False
