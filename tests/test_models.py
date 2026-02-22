import pytest
import numpy as np
import time

try:
    from .devices import DEVICES
except ImportError:
    # Fallback if run directly
    DEVICES = ["cpu", "cuda", "openvino"]

# Ensure we're running from the project root where models/ exists
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.mark.parametrize("device", DEVICES)
def test_stt_sherpa_zipformer(device):
    """Smoke test for Zipformer streaming STT loading and minimal decoding."""
    import sherpa_onnx
    print(f"\n--- Testing STT Provider: {device} ---")
    
    encoder = "models/stt/zipformer-encoder.onnx"
    decoder = "models/stt/zipformer-decoder.onnx"
    joiner = "models/stt/zipformer-joiner.onnx"
    tokens = "models/stt/tokens.txt"
    
    if not all(os.path.exists(p) for p in [encoder, decoder, joiner, tokens]):
        pytest.skip(f"Test skipped: Required model files missing in models/stt/")

    start_load = time.time()
    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            provider=device,
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
        )
    except Exception as e:
        pytest.fail(f"Failed to load STT with provider '{device}': {str(e)}")
        
    print(f"[{device}] STT loaded in {time.time()-start_load:.3f}s")
    
    # Run dummy 1-second audio frame (zeros)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    start_infer = time.time()
    
    try:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, dummy_audio)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        
        # Flush stream
        tail = np.zeros(8000, dtype=np.float32)
        stream.accept_waveform(16000, tail)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        text = recognizer.get_result(stream)
    except Exception as e:
        pytest.fail(f"Failed to infer STT with provider '{device}': {str(e)}")
        
    print(f"[{device}] Inference completed in {time.time()-start_infer:.3f}s. Result text: '{text}'")
    assert isinstance(text, str), "Inference didn't return a string"

@pytest.mark.parametrize("device", DEVICES)
def test_tts_sherpa_kokoro(device):
    """Smoke test for Kokoro offline TTS loading and minimal generation."""
    import sherpa_onnx
    print(f"\n--- Testing TTS Provider: {device} ---")
    
    model_path = "models/tts/kokoro.onnx"
    tokens_path = "models/tts/kokoro-tokens.txt"
    voices_path = "models/tts/kokoro-voices.bin"
    
    if not all(os.path.exists(p) for p in [model_path, tokens_path, voices_path]):
        pytest.skip(f"Test skipped: Required TTS model files missing in models/tts/")

    start_load = time.time()
    try:
        # Construct the complex offline config payload required for modern Sherpa
        # Using positional arguments to match the C++ PyBind signature explicitly
        # (model: str, voices: str, tokens: str, lexicon: str = '', data_dir: str, dict_dir: str = '', length_scale: float = 1.0, lang: str = '')
        kokoro_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=model_path,
            voices=voices_path,
            tokens=tokens_path,
            data_dir="models/tts",
        )
        
        # Use provider='cpu' to wrap the base OfflineTtsModelConfig parameter. 
        # The provider actually is set in OfflineTtsModelConfig or directly if available 
        # For this test, to avoid API breaking changes, we test the primary initialization structure.
        model_config = sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro_config, provider=device)
        
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            max_num_sentences=1
        )
        
        tts = sherpa_onnx.OfflineTts(config=tts_config)
    except Exception as e:
        pytest.fail(f"Failed to load TTS with provider '{device}': {str(e)}")
        
    print(f"[{device}] TTS loaded in {time.time()-start_load:.3f}s")
    
    # Run a quick generation
    start_infer = time.time()
    try:
        audio = tts.generate("Test.", sid=0, speed=1.0)
    except Exception as e:
        pytest.fail(f"Failed to synthesize TTS with provider '{device}': {str(e)}")
        
    print(f"[{device}] Inference completed in {time.time()-start_infer:.3f}s")
    assert audio is not None, "TTS returned None audio payload"
    assert len(audio.samples) > 0, "TTS returned empty sample array"
