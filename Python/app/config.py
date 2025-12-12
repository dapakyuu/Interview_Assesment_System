# app/config.py

import os
import torch
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# DIRECTORY CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
TRANSCRIPTION_DIR = BASE_DIR / "transcriptions"
RESULTS_DIR = BASE_DIR / "results"
AUDIO_DIR = BASE_DIR / "audio"
FFMPEG_BIN = BASE_DIR / "bin"  # ✅ Sesuaikan dengan struktur Anda

# Create directories if not exist
for directory in [UPLOAD_DIR, TRANSCRIPTION_DIR, RESULTS_DIR,AUDIO_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================
# WHISPER MODEL CONFIGURATION
# ============================================================
# Detect device automatically
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Model settings
WHISPER_MODEL_SIZE = "large-v3"  # Most accurate model

# ============================================================
# DEEPL TRANSLATOR CONFIGURATION
# ============================================================
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")

# ============================================================
# ANTHROPIC API CONFIGURATION (if needed)
# ============================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ============================================================
# PRINT CONFIGURATION
# ============================================================
print(f"\n🔧 Config loaded:")
print(f"   Device: {DEVICE.upper()}")
print(f"   Compute Type: {COMPUTE_TYPE}")
print(f"   Whisper Model: {WHISPER_MODEL_SIZE}")
print(f"   DeepL API: {'✅ Configured' if DEEPL_API_KEY else '❌ Not configured'}\n")