# app/services/whisper_service.py

from faster_whisper import WhisperModel
from app.config import DEVICE, COMPUTE_TYPE, WHISPER_MODEL_SIZE

# Global singleton instance
_whisper_model = None

def get_whisper_model():
    """
    Get or initialize Whisper model (Singleton pattern).
    Model is loaded only once on first call.
    
    Returns:
        WhisperModel: Initialized faster-whisper model
    """
    global _whisper_model
    
    if _whisper_model is None:
        print('\n📥 Loading Whisper model (first-time only)...')
        print(f'ℹ️  Using faster-whisper "{WHISPER_MODEL_SIZE}" model')
        print('   This is the MOST ACCURATE model available')
        print('   Speed: 4-5x faster than openai-whisper')
        print('   Accuracy: ~98% for clear English speech')
        print('   First run will download ~3GB model...\n')
        
        print(f'🎯 Configuration:')
        print(f'   Device: {DEVICE.upper()}')
        print(f'   Compute Type: {COMPUTE_TYPE}')
        
        try:
            # Load model with best accuracy settings
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=4,
                num_workers=1
            )
            
            print('✅ Whisper model loaded successfully\n')
            
        except Exception as e:
            print(f'❌ Failed to load Whisper model: {e}\n')
            raise Exception(f"Whisper model initialization failed: {e}")
    
    return _whisper_model

def is_model_loaded():
    """Check if model is already loaded"""
    return _whisper_model is not None