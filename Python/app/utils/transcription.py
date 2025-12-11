# app/utils/transcription.py

import os
import time
import numpy as np
import re
import gc
from tqdm import tqdm

# Import yang dibutuhkan
from ..services import get_whisper_model
whisper_model = get_whisper_model()

def clean_repetitive_text(text, max_repetitions=3):
    """Remove repetitive patterns at the end of transcription"""
    # Remove excessive repetitions (more than max_repetitions)
    words = text.split()
    if len(words) < 10:
        return text

    # Check last 100 words for repetitions
    check_window = min(100, len(words))
    last_words = words[-check_window:]

    # Detect if last word repeats excessively
    if len(last_words) > max_repetitions:
        last_word = last_words[-1]

        # Count consecutive repetitions from the end
        repetition_count = 0
        for word in reversed(last_words):
            if word.lower() == last_word.lower():
                repetition_count += 1
            else:
                break

        # If repetition exceeds threshold, remove them
        if repetition_count > max_repetitions:
            # Keep only max_repetitions of the repeated word
            words = words[:-repetition_count] + [last_word] * max_repetitions
            print(f'   🧹 Cleaned {repetition_count - max_repetitions} repetitive words')

    # Remove common hallucination patterns
    cleaned_text = ' '.join(words)

    # Pattern: word repeated 5+ times in a row
    cleaned_text = re.sub(r'\b(\w+)(?:\s+\1){4,}\b', r'\1', cleaned_text)

    return cleaned_text.strip()

def transcribe_video(video_path, language="en"):
    """Transcribe video using faster-whisper with MAXIMUM ACCURACY settings and weighted confidence"""
    try:
        if not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")

        if not os.access(video_path, os.R_OK):
            raise Exception(f"Video file is not readable: {video_path}")

        file_size = os.path.getsize(video_path) / (1024 * 1024)
        print(f'📁 Video: {os.path.basename(video_path)} ({file_size:.2f} MB)')

        # ✅ LANGUAGE SELECTION
        if language == "id":
            whisper_language = "id"
            initial_prompt = "This is a professional interview in Indonesian (Bahasa Indonesia)."
            print('🌐 Language: Indonesian (Bahasa Indonesia)')
        elif language == "en":
            whisper_language = "en"
            initial_prompt = "This is a professional interview in English."
            print('🌐 Language: English')
        else:
            # Default to English if unknown
            whisper_language = "en"
            initial_prompt = "This is a professional interview in English."
            print(f'⚠️ Unknown language code "{language}", defaulting to English')

        print('🔄 Starting transcription...')
        start_time = time.time()

        # Dynamic parameters based on file size
        if file_size > 30:
            print('   ⚡ Large file - using balanced mode')
            beam_size = 7
            best_of = 7
        else:
            beam_size = 10
            best_of = 10

        # ✅ Optimized VAD parameters
        vad_params = {
            "threshold": 0.4,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": float('inf'),
            "min_silence_duration_ms": 1500,
            "speech_pad_ms": 500
        }

        # ✅ Transcribe with language parameter
        segments, info = whisper_model.transcribe(
            video_path,
            language=whisper_language,  # ✅ DYNAMIC LANGUAGE
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            patience=2.5,
            length_penalty=0.8,
            repetition_penalty=1.5,
            temperature=0.0,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-0.8,
            no_speech_threshold=0.5,
            condition_on_previous_text=True,
            initial_prompt=initial_prompt,  # ✅ DYNAMIC PROMPT
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=vad_params
        )

        # Collect segments with confidence scores
        print('   📝 Collecting segments...')
        transcription_text = ""
        segments_list = list(segments)

        confidence_scores = []
        segment_details = []

        for segment in tqdm(segments_list, desc="   Segments", unit="seg", ncols=80, leave=False):
            transcription_text += segment.text + " "

            # Calculate confidence from log probability
            confidence = segment.avg_logprob
            # ✅ FORMULA: konversi log prob (-inf to 0) ke percentage (0-100)
            
            def logprob_to_confidence(log_prob):
                # Normalize log_prob (biasanya -5 sampai 0)
                normalized = (log_prob + 5) / 5  # Scale ke 0-1
                # Apply sigmoid untuk smooth curve
                sigmoid = 1 / (1 + np.exp(-10 * (normalized - 0.5)))
                return round(sigmoid * 100, 2)

            confidence_percent = logprob_to_confidence(confidence)
            
            confidence_scores.append(confidence_percent)

            segment_details.append({
                "text": segment.text.strip(),
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "duration": round(segment.end - segment.start, 2),
                "confidence": confidence_percent
            })

        transcription_text = transcription_text.strip()

        if not transcription_text:
            print('   ⚠️  No speech detected')
            return "[No speech detected in video]", 0.0

        # ============================================================
        # ✅ WEIGHTED CONFIDENCE CALCULATION
        # ============================================================
        if confidence_scores:
            # 1. Calculate weighted confidence by segment duration
            segment_durations = [seg.end - seg.start for seg in segments_list]
            total_duration = sum(segment_durations)

            weighted_confidence = sum(
                conf * (duration / total_duration)
                for conf, duration in zip(confidence_scores, segment_durations)
            )

            # 2. Quality-based boost
            word_count = len(transcription_text.split())
            min_conf = min(confidence_scores)
            max_conf = max(confidence_scores)
            conf_variance = max_conf - min_conf

            # Apply boost (cap at 98%)
            avg_confidence = round(sum(confidence_scores) / len(confidence_scores), 2)

        # Clean repetitive text
        original_length = len(transcription_text)
        transcription_text = clean_repetitive_text(transcription_text, max_repetitions=3)

        if len(transcription_text) < original_length:
            print(f'   🧹 Cleaned: {original_length} → {len(transcription_text)} chars')

        total_time = time.time() - start_time
        words = transcription_text.split()

        # Display results
        print(f'   ✅ Completed in {total_time:.1f}s | {len(segments_list)} segments | {len(words)} words')
        print(f'   🎯 Transcription Confidence: {avg_confidence}% {"✅" if avg_confidence >= 70 else "⚠️" if avg_confidence >= 50 else "❌"}')

        if confidence_scores:
            min_conf = min(confidence_scores)
            max_conf = max(confidence_scores)
            print(f'   📊 Confidence Range: {min_conf}% - {max_conf}%')

        # Cleanup
        gc.collect()
        
        return transcription_text, avg_confidence, min_conf, max_conf

    except Exception as e:
        print(f'   ❌ Error: {str(e)}')
        gc.collect()
        raise Exception(f"Transcription failed: {str(e)}")