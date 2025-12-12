from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import router
from .config import UPLOAD_DIR, TRANSCRIPTION_DIR, RESULTS_DIR, AUDIO_DIR
from .services import get_whisper_model, get_deepl_translator, get_voice_encoder

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # ============================================================
    # STARTUP: Initialize models and services
    # ============================================================
    print("\n" + "="*60)
    print("🚀 Starting AI Interview Assessment System")
    print("="*60)
    
    # Load Whisper model
    try:
        whisper_model = get_whisper_model()
        if whisper_model is None:
            raise Exception("Whisper model is None")
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        raise
    
    # Initialize DeepL translator
    get_deepl_translator()
    get_voice_encoder()
    
    print("="*60)
    print("✅ System ready to accept requests!")
    print("="*60 + "\n")
    
    yield  # Application runs here
    
    # ============================================================
    # SHUTDOWN: Cleanup
    # ============================================================
    print("\n" + "="*60)
    print("👋 Shutting down AI Interview Assessment System")
    print("="*60 + "\n")

def create_app():
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="AI Interview Assessment System",
        description="Transcribe and analyze interview videos with AI",
        version="1.0.0",
        lifespan=lifespan
    )

    # ============================================================
    # CORS Middleware
    # ============================================================
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
        expose_headers=["*"],
    )

    # ============================================================
    # Static Files - Convert Path to string
    # ============================================================
    app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
    app.mount("/transcriptions", StaticFiles(directory=str(TRANSCRIPTION_DIR)), name="transcriptions")
    app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
    app.mount("/results", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

    # ============================================================
    # Routes
    # ============================================================
    app.include_router(router)

    return app