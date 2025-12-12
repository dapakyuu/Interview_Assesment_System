import os
import threading
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ✅ Import dari config
try:
    from .config import UPLOAD_DIR, RESULTS_DIR, TRANSCRIPTION_DIR, AUDIO_DIR
    # Convert Path to string untuk kompatibilitas
    UPLOAD_DIR = str(UPLOAD_DIR)
    RESULTS_DIR = str(RESULTS_DIR)
    TRANSCRIPTION_DIR = str(TRANSCRIPTION_DIR)
    AUDIO_DIR = str(AUDIO_DIR)
except ImportError:
    # Fallback jika config tidak ada
    UPLOAD_DIR = "uploads"
    RESULTS_DIR = "results"
    TRANSCRIPTION_DIR = "transcriptions"
    AUDIO_DIR = "audio"
    
    # Buat direktori
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

# ================================
# Global Configuration
# ================================

processing_status = {}
processing_lock = threading.Lock()

executor = ThreadPoolExecutor(max_workers=2)

# ================================
# Helper Functions
# ================================

def get_local_file_path(url: str):
    """Extract local file path from uploaded file URL."""
    try:
        parsed = urlparse(url)

        # Pastikan path mengandung /uploads/
        if "/uploads/" in parsed.path:
            filename = parsed.path.split("/uploads/")[-1]
            local_path = os.path.join(UPLOAD_DIR, filename)

            # Cek apakah file benar-benar ada
            if os.path.exists(local_path):
                return local_path

    except Exception as e:
        print(f"Error parsing URL: {e}")

    return None

# Export config constants for backward compatibility
__all__ = [
    'processing_status', 
    'processing_lock', 
    'UPLOAD_DIR', 
    'RESULTS_DIR',
    'AUDIO_DIR',
    'TRANSCRIPTION_DIR',
    'executor',
    'get_local_file_path'
]