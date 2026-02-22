import sys
import sherpa_onnx
import time

provider = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
print(f"Testing TTS with provider: {provider}")

model_path = "models/tts/kokoro.onnx"
# Kokoro TTS uses an offline TTS class in sherpa
try:
    tokens_path = "models/tts/kokoro-tokens.txt"
    voices_path = "models/tts/kokoro-voices.bin"
    kokoro_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=model_path,
        voices=voices_path,
        tokens=tokens_path,
        data_dir="models/tts",
    )
    model_config = sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro_config, provider=provider)
    tts_config = sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1)
    tts = sherpa_onnx.OfflineTts(config=tts_config)

    print(f"[{provider}] TTS engine loaded successfully.")
    
    text = "Hello Krishna. Ikaris is online and running on the " + provider + " stack."
    print(f"[{provider}] Synthesizing text: '{text}'")
    
    start = time.time()
    audio = tts.generate(text, sid=0, speed=1.0)
    print(f"[{provider}] Generation took {time.time()-start:.2f}s")
    
    import wave
    import numpy as np
    
    samples = np.array(audio.samples, dtype=np.float32)
    # Convert float32 to int16 for wave output
    samples_int16 = (samples * 32767).astype(np.int16)
    
    filename = f"test_{provider}.wav"
    with wave.open(filename, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(audio.sample_rate)
        f.writeframes(samples_int16.tobytes())
        
    print(f"[{provider}] Saved audio to {filename}")

except Exception as e:
    print(f"[{provider}] TTS Error: {e}")
