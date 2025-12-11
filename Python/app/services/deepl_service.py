# app/services/deepl_service.py

import deepl
from app.config import DEEPL_API_KEY

# Global singleton instance
_translator = None
_is_available = False

def get_deepl_translator():
    """
    Get or initialize DeepL translator (Singleton pattern).
    Translator is initialized only once on first call.
    
    Returns:
        deepl.Translator | None: Initialized DeepL translator or None if not available
    """
    global _translator, _is_available
    
    if _translator is None and not _is_available:
        if DEEPL_API_KEY and DEEPL_API_KEY != "YOUR_DEEPL_API_KEY_HERE":
            try:
                print('\n🌐 Initializing DeepL translator...')
                _translator = deepl.Translator(DEEPL_API_KEY)
                _is_available = True
                print('✅ DeepL translator initialized successfully\n')
                
            except Exception as e:
                print(f'⚠️  DeepL initialization failed: {e}')
                print('   Translation to Indonesian will be skipped\n')
                _is_available = False
        else:
            print('\n⚠️  DeepL API key not configured')
            print('   Translation to Indonesian will be skipped\n')
            _is_available = False
    
    return _translator

def is_translator_available():
    """Check if DeepL translator is available"""
    return _is_available

def translate_text(text, target_lang="ID", source_lang=None):
    """
    Translate text using DeepL.
    
    Args:
        text (str): Text to translate
        target_lang (str): Target language code (default: "ID" for Indonesian)
        source_lang (str): Source language code (optional, auto-detect if None)
    
    Returns:
        str | None: Translated text or None if translation fails
    """
    translator = get_deepl_translator()
    
    if translator is None:
        print('⚠️  DeepL translator not available, skipping translation')
        return None
    
    try:
        result = translator.translate_text(
            text,
            target_lang=target_lang,
            source_lang=source_lang
        )
        return result.text
    
    except Exception as e:
        print(f'❌ Translation failed: {e}')
        return None