# app/utils/translation.py

import re

# ✅ Import service untuk get translator
from ..services import get_deepl_translator

def translate_to_indonesian(text):
    """Translate English → Indonesian using DeepL"""
    
    # ✅ Dapatkan translator dari service
    translator = get_deepl_translator()
    
    if not translator:
        print('   ⚠️ Translation skipped (DeepL not available)')
        return {"translated_text": text}  # Return original text jika translator tidak ada

    try:
        max_chunk_size = 5000

        # Text kecil, langsung translate
        if len(text) <= max_chunk_size:
            result = translator.translate_text(text, source_lang="EN", target_lang="ID")
            translated_text = result.text

        else:
            # Bagi per kalimat
            sentences = text.split('. ')
            chunks = []
            current = ""

            for s in sentences:
                if len(current) + len(s) + 2 <= max_chunk_size:
                    current += s + ". "
                else:
                    chunks.append(current)
                    current = s + ". "

            if current:
                chunks.append(current)

            translated_chunks = [
                translator.translate_text(chunk, source_lang="EN", target_lang="ID").text
                for chunk in chunks
            ]

            translated_text = " ".join(translated_chunks)

        print(f"   ✅ Translation EN→ID completed ({len(text)} → {len(translated_text)} chars)")
        return {"translated_text": translated_text}

    except Exception as e:
        print(f"   ❌ Translation failed: {str(e)}")
        # Return original text jika gagal
        return {"translated_text": text}


def translate_to_english(text):
    """Translate Indonesian → English (US) using DeepL"""
    
    # ✅ Dapatkan translator dari service
    translator = get_deepl_translator()
    
    if not translator:
        print('   ⚠️ Translation skipped (DeepL not available)')
        return {"translated_text": text}  # Return original text jika translator tidak ada

    try:
        max_chunk_size = 5000

        if len(text) <= max_chunk_size:
            result = translator.translate_text(text, source_lang="ID", target_lang="EN-US")
            translated_text = result.text

        else:
            sentences = text.split('. ')
            chunks = []
            current = ""

            for s in sentences:
                if len(current) + len(s) + 2 <= max_chunk_size:
                    current += s + ". "
                else:
                    chunks.append(current)
                    current = s + ". "

            if current:
                chunks.append(current)

            translated_text_list = [
                translator.translate_text(chunk, source_lang="ID", target_lang="EN-US").text
                for chunk in chunks
            ]

            translated_text = " ".join(translated_text_list)

        print(f"   ✅ Translation ID→EN completed ({len(text)} → {len(translated_text)} chars)")
        return {"translated_text": translated_text}

    except Exception as e:
        print(f"   ❌ Translation failed: {str(e)}")
        # Return original text jika gagal
        return {"translated_text": text}