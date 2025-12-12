# ============================================================================  
# 🔊 INITIALIZE VOICE ENCODER FOR SPEAKER DIARIZATION  
# ============================================================================  
_voice_encoder = None

def get_voice_encoder():
    """Get or initialize VoiceEncoder (Singleton pattern)."""
    global _voice_encoder

    if _voice_encoder is None:
        print('\n📥 Loading Voice Encoder for Speaker Diarization...')
        try:
            import torch
            from resemblyzer import VoiceEncoder

            print('   Configuring for CPU mode (avoiding cuDNN errors)...')
            if torch.cuda.is_available():
                print('   ℹ️  GPU available but using CPU to avoid cuDNN conflicts')

            # Force device to CPU
            device = torch.device('cpu')
            _voice_encoder = VoiceEncoder(device='cpu')

            print('✅ Voice Encoder loaded successfully (~50MB)')
            print('   Device: CPU (cuDNN conflict avoided)')
            print('   Model: Resemblyzer GE2E (Google Embeddings)')
            print('   Purpose: Detect multiple speakers in audio')
            print('   Note: CPU mode is slower but more stable\n')

        except Exception as e:
            print(f'⚠️  Voice Encoder failed to load: {e}')
            print('   Speaker diarization will return default values\n')
            _voice_encoder = None

    return _voice_encoder

def is_voice_encoder_loaded():
    """Check if voice encoder is already loaded"""
    return _voice_encoder is not None