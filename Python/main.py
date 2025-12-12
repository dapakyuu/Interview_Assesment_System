import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import logging
# logging.getLogger("mediapipe").setLevel(logging.ERROR)
# logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ============================================================
# KONFIGURASI FFMPEG PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_BIN_PATH = os.path.join(BASE_DIR, "bin")

if os.path.exists(FFMPEG_BIN_PATH):
    # Tambahkan PATH
    os.environ["PATH"] = FFMPEG_BIN_PATH + os.pathsep + os.environ.get("PATH", "")
    
    # Set environment variable Pydub agar tidak error
    os.environ["FFMPEG_BINARY"] = "ffmpeg"
    os.environ["FFPROBE_BINARY"] = "ffprobe"

    print("✅ FFmpeg loaded from:", FFMPEG_BIN_PATH)
    print("   ├─ ffmpeg:", os.path.exists(os.path.join(FFMPEG_BIN_PATH, "ffmpeg.exe")))
    print("   ├─ ffprobe:", os.path.exists(os.path.join(FFMPEG_BIN_PATH, "ffprobe.exe")))
    print("   └─ ffplay:", os.path.exists(os.path.join(FFMPEG_BIN_PATH, "ffplay.exe")))

else:
    print("⚠️ FFmpeg not found at:", FFMPEG_BIN_PATH)

# ============================================================
# RUN APP
# ============================================================
from app.server import create_app
import uvicorn

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
