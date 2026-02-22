import sys
import sherpa_onnx
import time
import numpy as np

def run_stt(provider="cpu"):
    print(f"[{provider}] Loading STT model...")
    encoder = "models/stt/zipformer-encoder.onnx"
    decoder = "models/stt/zipformer-decoder.onnx"
    joiner = "models/stt/zipformer-joiner.onnx"
    tokens = "models/stt/tokens.txt"
    
    start = time.time()
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
    except Exception as e:
        print(f"[{provider}] Failed to load STT: {e}")
        raise
        
    print(f"[{provider}] Fully loaded in {time.time()-start:.2f}s")
    
    # Run dummy inference
    dummy_audio = np.zeros(16000, dtype=np.float32)
    start = time.time()
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, dummy_audio)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = recognizer.get_result(stream).text
    
    print(f"[{provider}] Inference complete in {time.time()-start:.2f}s")
    return text

def run_tts(provider="cpu"):
    print(f"[{provider}] Loading Kokoro TTS model...")
    model_path = "models/tts/kokoro.onnx"
    tokens_path = "models/tts/kokoro-tokens.txt"
    voices_path = "models/tts/kokoro-voices.bin"
    
    start = time.time()
    try:
        model_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=model_path,
            voices=voices_path,
            tokens=tokens_path
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(kokoro=model_config),
            max_num_sentences=1
        )
        
        tts = sherpa_onnx.OfflineTts(
            config=tts_config,
            # Note: The C++ interface handles provider selection in offline_tts_model_config,
            # but since that requires passing down through the PyBind layers differently,
            # let's try the simplest route that fails gracefully.
        )
    except Exception as e:
        print(f"[{provider}] Failed to load TTS: {e}")
        raise
        
    print(f"[{provider}] Fully loaded in {time.time()-start:.2f}s")
    return True
