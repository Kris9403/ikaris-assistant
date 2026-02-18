"""
Ikaris Assistant — GUI Entry Point
Launches the PyQt5 researcher interface.
"""
import os
import sys

# --- 1. Linux GUI Fix (Prevents "Wayland" warnings & crashes) ---
os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- 2. Hugging Face Optimization (The "Reuse Forever" Fix) ---
# tells HF to NEVER check the internet for models (forces local cache)
os.environ["HF_HUB_OFFLINE"] = "1" 
# Silences the "Unauthenticated" warning
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
# Silences the "Loading weights" progress bars and info logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from PyQt5.QtWidgets import QApplication
from src.ui.main_window import IkarisMainWindow
from src.ui.styles import DARK_THEME

def main():
    # Fix for high-DPI displays on Linux
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    app.setApplicationName("Ikaris Assistant")
    app.setStyleSheet(DARK_THEME)

    window = IkarisMainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
