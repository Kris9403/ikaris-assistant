import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import os
import time

# Set device to 'cuda' for RTX 5070 Ti, fallback to 'cpu'
device = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
# Using 'small' model for better accuracy, runs well on RTX 5070 Ti
model = WhisperModel("small", device=device, compute_type="float16" if device == "cuda" else "int8")

def record_until_silence(fs=16000, silence_threshold=0.01, silence_duration=1.5):
    """Records audio from the microphone until silence is detected."""
    print("--- Ikaris is listening... ---")
    
    recording = []
    silent_chunks = 0
    chunk_size = int(fs * 0.1) # 100ms chunks
    
    def callback(indata, frames, time, status):
        nonlocal recording, silent_chunks
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
        recording.append(indata.copy())
        
        if volume_norm < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0

    with sd.InputStream(samplerate=fs, channels=1, callback=callback, blocksize=chunk_size):
        while silent_chunks < (silence_duration / 0.1):
            time.sleep(0.1)
            # Add a safety timeout (e.g. 30 seconds)
            if len(recording) > (fs * 30 / chunk_size):
                break

    print("--- Thinking... ---")
    return np.concatenate(recording, axis=0), fs

def transcribe_audio(recording, fs):
    """Transcribes audio using Faster-Whisper."""
    temp_filename = "temp_voice.wav"
    # Ensure clipping is avoided and normalized
    audio_data = (recording * 32767).astype(np.int16)
    write(temp_filename, fs, audio_data)
    
    # Leverages built-in Silero VAD for high-accuracy noise filtering
    segments, info = model.transcribe(
        temp_filename, 
        beam_size=5, 
        vad_filter=True,
        vad_parameters=dict(min_speech_duration_ms=250)
    )
    
    text = ""
    for segment in segments:
        text += segment.text
        
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return text.strip()

def get_voice_input():
    """Wrapper to record with VAD and transcribe."""
    try:
        recording, fs = record_until_silence()
        text = transcribe_audio(recording, fs)
        return text
    except Exception as e:
        return f"Error with voice input: {str(e)}"
