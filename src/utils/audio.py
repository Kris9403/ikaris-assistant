"""
Sherpa-ONNX Audio Stack — Unified STT + TTS engine.

Supports three providers:
  - openvino (NPU) : Zipformer streaming STT + Kokoro TTS
  - cuda (GPU)     : Whisper float16 STT + Kokoro TTS
  - cpu            : Whisper INT8 STT + Piper TTS

Usage via Hydra:
  python run.py audio=npu
  python run.py audio=cpu
  python run.py audio=cuda
  python run.py audio=none
"""

import os
import time
import logging
import numpy as np

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


# ===================================================================
# SherpaAudioStack — the main class Hydra instantiates
# ===================================================================
class SherpaAudioStack:
    """
    Unified audio engine for Ikaris.
    Hydra creates this via `_target_: src.utils.audio.SherpaAudioStack`.
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
        self._sherpa = None

        log.info(f"[Audio] SherpaAudioStack initialized | provider={provider} device={device}")
        log.info(f"[Audio] STT: {self.stt_cfg.get('type', 'none')} | TTS: {self.tts_cfg.get('type', 'none')}")

    # ------------------------------------------------------------------
    # STT: Speech-to-Text
    # ------------------------------------------------------------------
    def _init_stt(self):
        """Lazy-initialize the STT recognizer based on config."""
        if self._recognizer is not None:
            return

        sherpa = _import_sherpa()
        if sherpa is None:
            return
        self._sherpa = sherpa

        stt_type = self.stt_cfg.get("type", "whisper")
        model_path = self.stt_cfg.get("model_path", "")
        tokens = self.stt_cfg.get("tokens", "")

        if not os.path.exists(model_path):
            log.warning(f"[Audio] STT model not found at {model_path}. "
                        f"Download it first. STT disabled.")
            return

        log.info(f"[Audio] Loading STT engine: {stt_type} on {self.provider}...")
        start = time.time()

        try:
            if stt_type == "zipformer_streaming":
                # --- Streaming Zipformer (Transducer) ---
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
                # --- Offline Whisper ---
                self._recognizer = sherpa.OfflineRecognizer.from_whisper(
                    encoder=model_path.replace(".onnx", "-encoder.onnx"),
                    decoder=model_path.replace(".onnx", "-decoder.onnx"),
                    tokens=tokens,
                    provider=self.provider,
                    num_threads=4,
                )
        except Exception as e:
            log.error(f"[Audio] Failed to load STT: {e}")
            self._recognizer = None
            return

        log.info(f"[Audio] STT loaded in {time.time() - start:.2f}s")

    def listen(self, fs: int = 16000, silence_threshold: float = 0.01,
               silence_duration: float = 1.5) -> str:
        """
        Record from microphone until silence, then transcribe.
        Returns the transcribed text.
        """
        sd = _import_sounddevice()
        if sd is None:
            return "Error: sounddevice not available."

        self._init_stt()
        if self._recognizer is None:
            return self._fallback_listen()

        stt_type = self.stt_cfg.get("type", "whisper")

        if stt_type == "zipformer_streaming":
            return self._listen_streaming(sd, fs, silence_threshold, silence_duration)
        else:
            return self._listen_offline(sd, fs, silence_threshold, silence_duration)

    def _listen_streaming(self, sd, fs, silence_threshold, silence_duration) -> str:
        """Streaming recognition with partial results."""
        stream = self._recognizer.create_stream()
        recording = []
        silent_chunks = 0
        chunk_size = int(fs * 0.1)

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks
            volume = np.linalg.norm(indata) / np.sqrt(len(indata))
            recording.append(indata.copy())

            # Feed audio to streaming recognizer
            samples = indata.flatten().astype(np.float32)
            stream.accept_waveform(fs, samples)

            if volume < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

        log.info("[Audio] 🎤 Streaming STT — Listening...")
        with sd.InputStream(samplerate=fs, channels=1, callback=callback,
                            blocksize=chunk_size):
            while silent_chunks < (silence_duration / 0.1):
                time.sleep(0.1)
                # Process available frames
                while self._recognizer.is_ready(stream):
                    self._recognizer.decode_stream(stream)
                if len(recording) > (fs * 30 / chunk_size):
                    break  # 30s safety timeout

        # Finalize
        tail_padding = np.zeros(int(fs * 0.5), dtype=np.float32)
        stream.accept_waveform(fs, tail_padding)
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        text = self._recognizer.get_result(stream).text.strip()
        log.info(f"[Audio] Transcribed (streaming): '{text}'")
        return text if text else "Error: No speech detected."

    def _listen_offline(self, sd, fs, silence_threshold, silence_duration) -> str:
        """Record full audio, then batch-transcribe with offline Whisper."""
        recording = []
        silent_chunks = 0
        chunk_size = int(fs * 0.1)

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks
            volume = np.linalg.norm(indata) / np.sqrt(len(indata))
            recording.append(indata.copy())
            if volume < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

        log.info("[Audio] 🎤 Offline STT — Recording...")
        with sd.InputStream(samplerate=fs, channels=1, callback=callback,
                            blocksize=chunk_size):
            while silent_chunks < (silence_duration / 0.1):
                time.sleep(0.1)
                if len(recording) > (fs * 30 / chunk_size):
                    break

        if not recording:
            return "Error: No audio captured."

        audio = np.concatenate(recording, axis=0).flatten().astype(np.float32)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(fs, audio)
        self._recognizer.decode(stream)

        text = stream.result.text.strip()
        log.info(f"[Audio] Transcribed (offline): '{text}'")
        return text if text else "Error: No speech detected."

    def _fallback_listen(self) -> str:
        """Fallback to legacy faster-whisper if Sherpa models not found."""
        log.warning("[Audio] Falling back to legacy faster-whisper...")
        try:
            from src.utils.voice import get_voice_input
            return get_voice_input()
        except Exception as e:
            return f"Error: {str(e)}"

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
                self._tts_engine = sherpa.OfflineTts(
                    model=model_path,
                    provider=self.provider,
                    num_threads=2,
                )
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
    # Capability check
    # ------------------------------------------------------------------
    @property
    def has_stt(self) -> bool:
        return bool(self.stt_cfg.get("model_path"))

    @property
    def has_tts(self) -> bool:
        return bool(self.tts_cfg.get("model_path"))


# ===================================================================
# NullAudioStack — when audio=none
# ===================================================================
class NullAudioStack:
    """No-op audio stack. Text-only mode."""
    name = "audio"
    capabilities = []

    def __init__(self, **kwargs):
        log.info("[Audio] NullAudioStack — audio disabled (text-only mode).")

    def listen(self, **kwargs) -> str:
        return "Error: Audio is disabled. Use text input."

    def speak(self, text: str):
        pass

    @property
    def has_stt(self) -> bool:
        return False

    @property
    def has_tts(self) -> bool:
        return False
