import os
import gc
import uuid
import json
import traceback
import time
import numpy as np

from datetime import datetime, timezone

# IMPORT DARI FOLDER utils/
from .transcription import transcribe_video
from .translation import translate_to_indonesian, translate_to_english
from .cheating_detection import aggregate_cheating_results,comprehensive_cheating_detection
from .non_verbal import analyze_interview_video_with_confidence, summarize_non_verbal_batch
from .llm_evaluator import evaluate_with_llm, summarize_llm_analysis_batch

# ✅ IMPORT DARI STATE
from ..state import (
    get_local_file_path,
    processing_status,
    processing_lock,
    RESULTS_DIR,
    TRANSCRIPTION_DIR
)

# ✅ Import service untuk verifikasi model
from ..services import is_model_loaded

def process_transcriptions_sync(session_id: str, candidate_name: str, uploaded_videos: list, base_url: str, language: str = "en"):
    """Background transcription processing WITH COMPREHENSIVE LOGGING"""

    # ⭐ SETUP LOGGING TO FILE
    log_file = f'session_{session_id}.log'
    log_handle = open(log_file, 'w', encoding='utf-8', buffering=1)

    def log_print(msg):
        """Print to both console and log file"""
        print(msg, flush=True)
        log_handle.write(msg + '\n')
        log_handle.flush()

    try:
        log_print(f'\n{"="*70}')
        log_print(f'🎙️  SESSION: {session_id}')
        log_print(f'👤 CANDIDATE: {candidate_name}')
        log_print(f'🌐 LANGUAGE: {"English" if language == "en" else "Indonesian" if language == "id" else language}')
        log_print(f'📹 VIDEOS: {len(uploaded_videos)}')
        log_print(f'📝 LOG FILE: {log_file}')
        log_print(f'{"="*70}\n')

        transcriptions = []
        assessment_results = []

        with processing_lock:
            processing_status[session_id] = {'status': 'processing', 'progress': '0/0'}

        # Process each video
        for idx, interview in enumerate(uploaded_videos, 1):
            log_print(f'\n{"─"*70}')
            log_print(f'Processing video {idx}/{len(uploaded_videos)}')
            log_print(f'{"─"*70}')

            if not interview.get('isVideoExist') or not interview.get('recordedVideoUrl'):
                log_print(f'⚠️ Video {idx} - No video exists or no URL')
                transcriptions.append({
                    'positionId': interview['positionId'],
                    'error': interview.get('error', 'Video upload failed')
                })
                continue

            position_id = interview['positionId']
            video_url = interview['recordedVideoUrl']
            question = interview.get('question', '')

            try:
                log_print(f'\n┌─ Video {position_id}/{len(uploaded_videos)} ─{"─"*50}┐')
                if question:
                    log_print(f'│ ❓ Question: {question[:60]}{"..." if len(question) > 60 else ""}')

                local_file = get_local_file_path(video_url)
                if not local_file:
                    raise Exception(f"Local file not found")

                log_print(f'│ 📁 Local file: {local_file}')
                log_print(f'│ 📏 File exists: {os.path.exists(local_file)}')

                file_size_mb = os.path.getsize(local_file) / (1024 * 1024)
                log_print(f'│ 📊 File size: {file_size_mb:.1f} MB')

                with processing_lock:
                    processing_status[session_id] = {
                        'status': 'processing',
                        'progress': f'{position_id}/{len(uploaded_videos)}',
                        'current_video': position_id,
                        'message': f'Processing video {position_id}/{len(uploaded_videos)}...'
                    }

                video_start = time.time()

                # Step 1: Transcribe
                log_print(f'│ 1️⃣  TRANSCRIPTION ({file_size_mb:.1f} MB)')
                try:
                    transcription_text, avg_confidence, min_conf, max_conf = transcribe_video(local_file, language=language)
                    transcribe_time = time.time() - video_start
                    log_print(f'│    ✅ Transcription completed')
                    log_print(f'│    🎯 Transcription Confidence: {avg_confidence}%')
                    log_print(f'│    📝 Text length: {len(transcription_text)} chars')
                except Exception as e:
                    log_print(f'│    ❌ Transcription ERROR: {str(e)}')
                    raise

                # Step 2: Translate (conditional based on language)
                log_print(f'│ 2️⃣  TRANSLATION')
                try:
                    translate_start = time.time()

                    if language == "en":
                        # English → Indonesian
                        translation_result = translate_to_indonesian(transcription_text)
                        transcription_en = transcription_text  # Original is English
                        transcription_id = translation_result['translated_text']  # Translated to Indonesian
                        log_print(f'│    🌐 Direction: English → Indonesian')
                    elif language == "id":
                        # Indonesian → English
                        translation_result = translate_to_english(transcription_text)
                        transcription_id = transcription_text  # Original is Indonesian
                        transcription_en = translation_result['translated_text']  # Translated to English
                        log_print(f'│    🌐 Direction: Indonesian → English')
                    else:
                        # Default: assume English
                        translation_result = translate_to_indonesian(transcription_text)
                        transcription_en = transcription_text
                        transcription_id = translation_result['translated_text']
                        log_print(f'│    ⚠️  Unknown language, defaulting to English → Indonesian')

                    translate_time = time.time() - translate_start
                    log_print(f'│    ✅ Translation completed in {translate_time:.1f}s')
                    log_print(f'│    📝 EN length: {len(transcription_en)} chars')
                    log_print(f'│    📝 ID length: {len(transcription_id)} chars')
                except Exception as e:
                    log_print(f'│    ❌ Translation ERROR: {str(e)}')
                    raise

                # Step 3: Cheating Detection
                log_print(f'│ 2️⃣½ CHEATING DETECTION')
                print('\n🔍 Running Cheating Detection...')
                try:
                    cheating_start = time.time()
                    cheating_result = comprehensive_cheating_detection(local_file)
                    cheating_time = time.time() - cheating_start
                    log_print(f'│    ✅ Cheating detection completed in {cheating_time:.1f}s')
                except Exception as e:
                    log_print(f'│    ❌ Cheating detection ERROR: {str(e)}')
                    raise

                # Step 4: Non-Verbal Analysis
                log_print(f'│ 2️⃣¾ NON-VERBAL ANALYSIS')
                try:
                    non_verbal_start = time.time()
                    non_verbal_result = analyze_interview_video_with_confidence(
                        video_path=local_file,
                        audio_path=None
                    )
                    non_verbal_time = time.time() - non_verbal_start
                    log_print(f'│    ✅ Non-verbal analysis completed in {non_verbal_time:.1f}s')
                    log_print(f'│    📊 Non-Verbal Confidence: {non_verbal_result["confidence_score"]}%')
                except Exception as e:
                    log_print(f'│    ❌ Non-verbal analysis ERROR: {str(e)}')
                    raise

                # Step 5: LLM Evaluation (always use English text for better accuracy)
                log_print(f'│ 3️⃣  AI ASSESSMENT')
                try:
                    llm_start = time.time()
                    llm_evaluation = evaluate_with_llm(transcription_en, question, position_id)  # ✅ Use English version
                    llm_time = time.time() - llm_start
                    log_print(f'│    ✅ LLM evaluation completed in {llm_time:.1f}s')
                    log_print(f'│    📊 Total Score: {llm_evaluation["total"]}/100')
                except Exception as e:
                    log_print(f'│    ❌ LLM evaluation ERROR: {str(e)}')
                    raise

                # Step 6: Save transcription file
                log_print(f'│ 4️⃣  SAVING FILES')
                trans_fname = f"transcription_pos{position_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}.txt"
                trans_path = os.path.join(TRANSCRIPTION_DIR, trans_fname)

                with open(trans_path, 'w', encoding='utf-8') as f:
                    f.write(f"Candidate: {candidate_name}\n")
                    f.write(f"Position ID: {position_id}\n")
                    f.write(f"Question: {question}\n")
                    f.write(f"Video URL: {video_url}\n")
                    f.write(f"Language: {language}\n")  # ✅ Added language info
                    f.write(f"Transcribed at: {datetime.now(timezone.utc).isoformat()}\n")
                    f.write(f"\n{'='*50}\n")

                    if language == "id":
                        f.write(f"INDONESIAN TRANSCRIPTION (Original):\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(transcription_id)
                        f.write(f"\n\n{'='*50}\n")
                        f.write(f"ENGLISH TRANSLATION:\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(transcription_en)
                    else:
                        f.write(f"ENGLISH TRANSCRIPTION (Original):\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(transcription_en)
                        f.write(f"\n\n{'='*50}\n")
                        f.write(f"INDONESIAN TRANSLATION:\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(transcription_id)

                    f.write(f"\n\nTranscription Confidence: {avg_confidence}%\n")

                log_print(f'│    ✅ Transcription file saved: {trans_fname}')

                transcription_url = f"{base_url}/transcriptions/{trans_fname}"

                # Build assessment
                words = transcription_en.split()

                assessment = {
                    "penilaian": {
                        "confidence_score": llm_evaluation['scores']['confidence_score'],
                        "kualitas_jawaban": llm_evaluation['scores']['kualitas_jawaban'],
                        "relevansi": llm_evaluation['scores']['relevansi'],
                        "koherensi": llm_evaluation['scores']['koherensi'],
                        "analisis_llm": llm_evaluation['analysis'],
                        "total": llm_evaluation['total'],
                        # 🆕 NEW: Add logprobs data
                        "logprobs_confidence": llm_evaluation.get('logprobs_confidence'),
                        "logprobs_probability": llm_evaluation.get('logprobs_probability'),
                        "logprobs_available": llm_evaluation.get('logprobs_available', False)
                    },
                    "non_verbal_analysis": non_verbal_result['analysis'],
                    "non_verbal_confidence_score": non_verbal_result['confidence_score'],
                    "transkripsi_en": transcription_en,
                    "transkripsi_id": transcription_id,
                    "transkripsi_confidence": avg_confidence,
                    "transkripsi_min_confidence": min_conf,
                    "transkripsi_max_confidence": max_conf,
                    "cheating_detection": cheating_result,
                    "metadata": {
                        "word_count": len(words),
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        # 🆕 NEW: Logprobs metadata
                        "logprobs_enabled": True,
                        "source_language": "English" if language == "en" else "Indonesian" if language == "id" else "Unknown"
                    }
                }

                assessment_results.append({
                    "id": position_id,
                    "question": question,
                    "result": assessment
                })
                log_print(f'│    ✅ Assessment added to results (total: {len(assessment_results)})')

                transcriptions.append({
                    'positionId': position_id,
                    'question': question,
                    'videoUrl': video_url,
                    'transcription': transcription_en,
                    'transcription_id': transcription_id,
                    'transcriptionUrl': transcription_url,
                    'transcriptionFile': trans_fname,
                    'assessment': assessment
                })

                # Delete video
                if os.path.exists(local_file):
                    os.remove(local_file)
                    log_print(f'│ 🗑️  Video deleted ({file_size_mb:.1f} MB freed)')

                total_time = time.time() - video_start
                log_print(f'│ ⏱️  Total: {total_time:.1f}s')
                log_print(f'└─{"─"*68}┘')

                gc.collect()

            except Exception as e:
                log_print(f'│ ❌ ERROR processing video {position_id}: {str(e)}')
                log_print(f'│ 📋 Traceback:')
                for line in traceback.format_exc().split('\n'):
                    log_print(f'│    {line}')
                log_print(f'└─{"─"*68}┘')

                transcriptions.append({
                    'positionId': position_id,
                    'question': question,
                    'videoUrl': video_url,
                    'error': str(e)
                })

        # ============================================================================
        # AGGREGATE ANALYSIS
        # ============================================================================
        log_print(f'\n{"="*70}')
        log_print(f'📊 STARTING AGGREGATE ANALYSIS')
        log_print(f'{"="*70}')
        log_print(f'Assessment Results Count: {len(assessment_results)}')

        if len(assessment_results) == 0:
            log_print(f'⚠️ WARNING: No assessment results! Cannot create aggregate analysis.')
            log_print(f'   Total transcriptions: {len(transcriptions)}')
            log_print(f'   Transcriptions with errors: {sum(1 for t in transcriptions if "error" in t)}')

        # 1. Aggregate Cheating
        try:
            log_print(f'\n👀 Calculating aggregate non-verbal...')
            aggregate_cheating = aggregate_cheating_results(assessment_results)
            log_print(f'✅ Aggregate cheating completed')
        except Exception as e:
            log_print(f'❌ ERROR in aggregate_non_verbal: {str(e)}')
            log_print(f'   Traceback: {traceback.format_exc()}')
            aggregate_cheating = {"error": str(e)}


        # 2. Aggregate Non-Verbal
        try:
            log_print(f'\n👀 Calculating aggregate non-verbal...')
            aggregate_non_verbal = summarize_non_verbal_batch(assessment_results)
            log_print(f'✅ Aggregate non-verbal completed')
        except Exception as e:
            log_print(f'❌ ERROR in aggregate_non_verbal: {str(e)}')
            log_print(f'   Traceback: {traceback.format_exc()}')
            aggregate_non_verbal = {"error": str(e)}

        # 3. LLM Summary
        try:
            log_print(f'\n🤖 Generating LLM summary...')
            hasil_llm = summarize_llm_analysis_batch(assessment_results)
            log_print(f'✅ LLM summary completed')
        except Exception as e:
            log_print(f'❌ ERROR in LLM summary: {str(e)}')
            log_print(f'   Traceback: {traceback.format_exc()}')
            hasil_llm = {
                "kesimpulan_llm": f"Error: {str(e)}",
                "rata_rata_confidence_score": 0,
                "error": str(e)
            }

        log_print(f'\n{"="*70}')
        log_print(f'✅ ALL AGGREGATE ANALYSIS COMPLETED')
        log_print(f'{"="*70}')

        # ============================================================================
        # SAVE JSON
        # ============================================================================
        if assessment_results:
            try:
                log_print(f'\n💾 SAVING JSON RESULTS...')

                results_json = {
                   "success": True,
                    "name": candidate_name,
                    "session": session_id,
                    "llm_results": hasil_llm,
                    "aggregate_cheating_detection": aggregate_cheating,
                    "aggregate_non_verbal_analysis": aggregate_non_verbal,
                    "content": assessment_results,
                    "metadata": {
                        "total_videos": len(uploaded_videos),
                        "successful_videos": len(assessment_results),
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "model": "faster-whisper large-v3",
                        "llm_model": "meta-llama/Llama-3.1-8B-Instruct"
                    }
                }

                results_filename = f"{session_id}.json"
                results_path = os.path.join(RESULTS_DIR, results_filename)

                log_print(f'📂 Results path: {results_path}')
                log_print(f'📊 JSON size: {len(str(results_json))} chars')

                # Ensure directory exists
                os.makedirs(RESULTS_DIR, exist_ok=True)
                log_print(f'✅ Results directory ensured: {RESULTS_DIR}')

                # Write JSON
                try:
                    with open(results_path, 'w', encoding='utf-8') as f:
                        json.dump(results_json, f, ensure_ascii=False, indent=2)

                    file_size = os.path.getsize(results_path)
                    print(f'✅ JSON saved successfully')
                    print(f'   File: {results_filename}')
                    print(f'   Size: {file_size:,} bytes ({file_size/1024:.1f} KB)')

                except Exception as save_error:
                    print(f'❌ ERROR saving JSON: {save_error}')
                    print(f'   Attempting alternative save method...')

                    # Fallback: Manually convert NumPy types
                    def convert_to_native(obj):
                        if isinstance(obj, dict):
                            return {k: convert_to_native(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_native(item) for item in obj]
                        elif isinstance(obj, (np.integer, np.int32, np.int64)):
                            return int(obj)
                        elif isinstance(obj, (np.floating, np.float32, np.float64)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        return obj

                    try:
                        cleaned_json = convert_to_native(results_json)
                        with open(results_path, 'w', encoding='utf-8') as f:
                            json.dump(cleaned_json, f, ensure_ascii=False, indent=2)
                        print(f'✅ JSON saved successfully (fallback method)')
                    except Exception as fallback_error:
                        print(f'❌ CRITICAL: Both save methods failed: {fallback_error}')
                        raise

                log_print(f'✅ JSON written to file')

                # Verify
                if os.path.exists(results_path):
                    file_size = os.path.getsize(results_path)
                    log_print(f'✅✅✅ JSON FILE SAVED SUCCESSFULLY! ✅✅✅')
                    log_print(f'   Path: {results_path}')
                    log_print(f'   Size: {file_size} bytes')
                else:
                    log_print(f'❌❌❌ WARNING: JSON FILE NOT CREATED! ❌❌❌')

                results_url = f"{base_url}/results/{results_filename}"
                log_print(f'🌐 Results URL: {results_url}')

            except Exception as e:
                log_print(f'❌ CRITICAL ERROR saving JSON: {str(e)}')
                log_print(f'   Traceback: {traceback.format_exc()}')
        else:
            log_print(f'\n⚠️⚠️⚠️ WARNING: assessment_results is EMPTY! ⚠️⚠️⚠️')
            log_print(f'   JSON will NOT be saved.')

        successful_count = sum(1 for t in transcriptions if 'transcription' in t)

        with processing_lock:
            processing_status[session_id] = {
                'status': 'completed',
                'result': {
                    'success': True,
                    'transcriptions': transcriptions,
                    'processed_videos': len(transcriptions),
                    'successful_videos': successful_count,
                    'failed_videos': len(transcriptions) - successful_count,
                    'results_url': f"{base_url}/results/{session_id}.json" if assessment_results else None
                }
            }

        log_print(f'\n{"="*70}')
        log_print(f'✅ SESSION COMPLETED')
        log_print(f'   Success: {successful_count}/{len(transcriptions)} videos')
        log_print(f'   Log file: {log_file}')
        log_print(f'{"="*70}\n')

    except Exception as e:
        log_print(f'\n❌ SESSION ERROR:\n{traceback.format_exc()}')

        with processing_lock:
            processing_status[session_id] = {
                'status': 'error',
                'error': str(e),
                'error_detail': traceback.format_exc()
            }

    finally:
        log_handle.close()