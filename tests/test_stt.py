import sys
import sounddevice as sd
import sherpa_onnx
import time

provider = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
print(f"Testing STT with provider: {provider}")

encoder = "models/stt/zipformer-encoder.onnx"
decoder = "models/stt/zipformer-decoder.onnx"
joiner = "models/stt/zipformer-joiner.onnx"
tokens = "models/stt/tokens.txt"

try:
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        provider=provider,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
    )
    print(f"[{provider}] STT recognizer loaded successfully.")
    
    # Optional: test functionality with real audio
    print(f"[{provider}] Recording 5 seconds of audio...")
    audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, audio.flatten())
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    
    res = recognizer.get_result(stream)
    text = res if isinstance(res, str) else res.text
    print(f"[{provider}] You said: '{text}'")

except Exception as e:
    print(f"[{provider}] STT Error: {e}")
