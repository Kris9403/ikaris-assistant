#!/usr/bin/env bash
# ============================================================
# Ikaris Assistant — Model Downloader
# ============================================================
# Downloads Sherpa-ONNX STT & TTS models and places them at
# the exact paths the Hydra audio configs expect.
#
# Usage:
#   bash scripts/pull_models.sh          # download everything
#   bash scripts/pull_models.sh cuda     # whisper-small + kokoro (CUDA profile)
#   bash scripts/pull_models.sh cpu      # whisper-base-int8 + piper (CPU profile)
#   bash scripts/pull_models.sh stt      # all STT models only
#   bash scripts/pull_models.sh tts      # all TTS models only
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
STT_DIR="$MODELS_DIR/stt"
TTS_DIR="$MODELS_DIR/tts"
VAD_DIR="$MODELS_DIR/vad"

GH_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"

# --- Color helpers ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[  OK]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ============================================================
# STT: Whisper small  (CUDA profile: configs/audio/cuda.yaml)
# Expected paths:
#   models/stt/whisper-small-encoder.onnx
#   models/stt/whisper-small-decoder.onnx
#   models/stt/tokens.txt
# ============================================================
download_whisper_small() {
    local ARCHIVE="sherpa-onnx-whisper-small.tar.bz2"
    local URL="${GH_BASE}/asr-models/${ARCHIVE}"
    local EXTRACT_DIR="sherpa-onnx-whisper-small"

    if [[ -f "$STT_DIR/whisper-small-encoder.onnx" ]]; then
        ok "Whisper small already exists — skipping"
        return
    fi

    info "Downloading Whisper small (~460 MB) ..."
    wget -q --show-progress -O "$STT_DIR/$ARCHIVE" "$URL"

    info "Extracting ..."
    tar xf "$STT_DIR/$ARCHIVE" -C "$STT_DIR"

    # Move files to the flat paths configs expect
    local SRC="$STT_DIR/$EXTRACT_DIR"
    cp "$SRC"/*-encoder.onnx  "$STT_DIR/whisper-small-encoder.onnx"  2>/dev/null || \
    cp "$SRC"/whisper-small-encoder.onnx "$STT_DIR/whisper-small-encoder.onnx" 2>/dev/null || true
    cp "$SRC"/*-decoder.onnx  "$STT_DIR/whisper-small-decoder.onnx"  2>/dev/null || \
    cp "$SRC"/whisper-small-decoder.onnx "$STT_DIR/whisper-small-decoder.onnx" 2>/dev/null || true

    # Tokens (use the first tokens.txt found)
    find "$SRC" -name "*.txt" -path "*token*" -exec cp {} "$STT_DIR/tokens.txt" \; 2>/dev/null || true

    # Keep the archive directory as backup, remove tarball
    rm -f "$STT_DIR/$ARCHIVE"
    ok "Whisper small ready"
}

# ============================================================
# STT: Whisper base INT8  (CPU profile: configs/audio/cpu.yaml)
# Expected paths:
#   models/stt/whisper-base-int8-encoder.onnx
#   models/stt/whisper-base-int8-decoder.onnx
#   models/stt/tokens.txt
# ============================================================
download_whisper_base() {
    local ARCHIVE="sherpa-onnx-whisper-base.en.tar.bz2"
    local URL="${GH_BASE}/asr-models/${ARCHIVE}"
    local EXTRACT_DIR="sherpa-onnx-whisper-base.en"

    if [[ -f "$STT_DIR/whisper-base-int8-encoder.onnx" ]]; then
        ok "Whisper base INT8 already exists — skipping"
        return
    fi

    info "Downloading Whisper base.en (~140 MB) ..."
    wget -q --show-progress -O "$STT_DIR/$ARCHIVE" "$URL"

    info "Extracting ..."
    tar xf "$STT_DIR/$ARCHIVE" -C "$STT_DIR"

    local SRC="$STT_DIR/$EXTRACT_DIR"
    # Prefer int8 variant if present, otherwise use standard
    if ls "$SRC"/*int8*encoder* 1>/dev/null 2>&1; then
        cp "$SRC"/*int8*encoder*  "$STT_DIR/whisper-base-int8-encoder.onnx"
        cp "$SRC"/*int8*decoder*  "$STT_DIR/whisper-base-int8-decoder.onnx"
    else
        cp "$SRC"/*encoder.onnx  "$STT_DIR/whisper-base-int8-encoder.onnx" 2>/dev/null || true
        cp "$SRC"/*decoder.onnx  "$STT_DIR/whisper-base-int8-decoder.onnx" 2>/dev/null || true
    fi

    find "$SRC" -name "*.txt" -path "*token*" -exec cp {} "$STT_DIR/tokens.txt" \; 2>/dev/null || true

    rm -f "$STT_DIR/$ARCHIVE"
    ok "Whisper base INT8 ready"
}

# ============================================================
# STT: Zipformer English (NPU profile: configs/audio/npu.yaml)
# Expected paths:
#   models/stt/zipformer-encoder.onnx
#   models/stt/zipformer-decoder.onnx
#   models/stt/zipformer-joiner.onnx
#   models/stt/tokens.txt
# ============================================================
download_zipformer() {
    local ARCHIVE="sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2"
    local URL="${GH_BASE}/asr-models/${ARCHIVE}"
    local EXTRACT_DIR="sherpa-onnx-zipformer-gigaspeech-2023-12-12"

    if [[ -f "$STT_DIR/zipformer-encoder.onnx" ]]; then
        ok "Zipformer already exists — skipping"
        return
    fi

    info "Downloading Zipformer English (~250 MB) ..."
    wget -q --show-progress -O "$STT_DIR/$ARCHIVE" "$URL"

    info "Extracting ..."
    tar xf "$STT_DIR/$ARCHIVE" -C "$STT_DIR"

    local SRC="$STT_DIR/$EXTRACT_DIR"
    cp "$SRC"/*encoder*.onnx  "$STT_DIR/zipformer-encoder.onnx"  2>/dev/null || true
    cp "$SRC"/*decoder*.onnx  "$STT_DIR/zipformer-decoder.onnx"  2>/dev/null || true
    cp "$SRC"/*joiner*.onnx   "$STT_DIR/zipformer-joiner.onnx"   2>/dev/null || true

    find "$SRC" -name "tokens.txt" -exec cp {} "$STT_DIR/tokens.txt" \; 2>/dev/null || true

    rm -f "$STT_DIR/$ARCHIVE"
    ok "Zipformer ready"
}

# ============================================================
# TTS: Kokoro EN v0.19  (CUDA/NPU: configs/audio/{cuda,npu}.yaml)
# Expected path: models/tts/kokoro.onnx
# ============================================================
download_kokoro() {
    local ARCHIVE="kokoro-en-v0_19.tar.bz2"
    local URL="${GH_BASE}/tts-models/${ARCHIVE}"
    local EXTRACT_DIR="kokoro-en-v0_19"

    if [[ -f "$TTS_DIR/kokoro.onnx" ]]; then
        ok "Kokoro already exists — skipping"
        return
    fi

    info "Downloading Kokoro EN v0.19 (~330 MB) ..."
    wget -q --show-progress -O "$TTS_DIR/$ARCHIVE" "$URL"

    info "Extracting ..."
    tar xf "$TTS_DIR/$ARCHIVE" -C "$TTS_DIR"

    local SRC="$TTS_DIR/$EXTRACT_DIR"
    cp "$SRC/model.onnx"       "$TTS_DIR/kokoro.onnx"
    cp "$SRC/tokens.txt"       "$TTS_DIR/kokoro-tokens.txt"    2>/dev/null || true
    cp "$SRC/voices.bin"       "$TTS_DIR/kokoro-voices.bin"    2>/dev/null || true
    cp -r "$SRC/espeak-ng-data" "$TTS_DIR/espeak-ng-data"      2>/dev/null || true

    rm -f "$TTS_DIR/$ARCHIVE"
    ok "Kokoro ready"
}

# ============================================================
# TTS: Piper (GLaDOS voice)  (CPU: configs/audio/cpu.yaml)
# Expected path: models/tts/piper-en.onnx
# ============================================================
download_piper() {
    local ARCHIVE="vits-piper-en_US-glados.tar.bz2"
    local URL="${GH_BASE}/tts-models/${ARCHIVE}"
    local EXTRACT_DIR="vits-piper-en_US-glados"

    if [[ -f "$TTS_DIR/piper-en.onnx" ]]; then
        ok "Piper already exists — skipping"
        return
    fi

    info "Downloading Piper GLaDOS (~61 MB) ..."
    wget -q --show-progress -O "$TTS_DIR/$ARCHIVE" "$URL"

    info "Extracting ..."
    tar xf "$TTS_DIR/$ARCHIVE" -C "$TTS_DIR"

    local SRC="$TTS_DIR/$EXTRACT_DIR"
    cp "$SRC"/*.onnx          "$TTS_DIR/piper-en.onnx"         2>/dev/null || true
    cp "$SRC/tokens.txt"      "$TTS_DIR/piper-tokens.txt"      2>/dev/null || true
    cp -r "$SRC/espeak-ng-data" "$TTS_DIR/piper-espeak-ng-data" 2>/dev/null || true

    rm -f "$TTS_DIR/$ARCHIVE"
    ok "Piper ready"
}

# ============================================================
# VAD: Silero VAD  (shared across all audio profiles)
# Expected path: models/vad/silero_vad.onnx
# ============================================================
download_vad() {
    local ARCHIVE="silero_vad.onnx"
    local URL="${GH_BASE}/asr-models/${ARCHIVE}"

    if [[ -f "$VAD_DIR/silero_vad.onnx" ]]; then
        ok "Silero VAD already exists — skipping"
        return
    fi

    info "Downloading Silero VAD (~2 MB) ..."
    wget -q --show-progress -O "$VAD_DIR/silero_vad.onnx" "$URL"
    ok "Silero VAD ready"
}

# ============================================================
# Dispatcher
# ============================================================
mkdir -p "$STT_DIR" "$TTS_DIR" "$VAD_DIR"

PROFILE="${1:-all}"

echo ""
echo "═══════════════════════════════════════════"
echo " 🦾  Ikaris Model Downloader"
echo "═══════════════════════════════════════════"
echo " Profile: $PROFILE"
echo " Target:  $MODELS_DIR"
echo "═══════════════════════════════════════════"
echo ""

case "$PROFILE" in
    cuda)
        download_vad
        download_whisper_small
        download_kokoro
        ;;
    cpu)
        download_vad
        download_whisper_base
        download_piper
        ;;
    npu)
        download_vad
        download_zipformer
        download_kokoro
        ;;
    stt)
        download_vad
        download_whisper_small
        download_whisper_base
        download_zipformer
        ;;
    tts)
        download_kokoro
        download_piper
        ;;
    vad)
        download_vad
        ;;
    all)
        download_vad
        download_whisper_small
        download_whisper_base
        download_zipformer
        download_kokoro
        download_piper
        ;;
    *)
        echo "Usage: bash scripts/pull_models.sh [cuda|cpu|npu|stt|tts|vad|all]"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════"
echo " ✅  Done! Models are in: $MODELS_DIR"
echo "═══════════════════════════════════════════"
echo ""
ls -lhR "$MODELS_DIR" --ignore='espeak-ng-data' --ignore='sherpa-*' --ignore='vits-*' --ignore='kokoro-*' 2>/dev/null || ls -lhR "$MODELS_DIR"
