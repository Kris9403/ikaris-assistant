#!/usr/bin/env bash

# This script runs the parameterized audio smoke tests
# ensuring STT and TTS function across the host's supported hardware.

echo "Running STT and TTS smoke tests via PyTest..."
pytest -v -s tests/test_models.py
