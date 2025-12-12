# app/utils/cheating_detection.py

import cv2
import torch
import numpy as np
import torchaudio
from pydub import AudioSegment
import mediapipe as mp
import traceback
from silero_vad import load_silero_vad
import librosa
import json
import os
import cv2
from sklearn.cluster import AgglomerativeClustering
from typing import List
from resemblyzer import preprocess_wav
from moviepy.editor import VideoFileClip

from ..services import get_voice_encoder
VoiceEncoder = get_voice_encoder()

from ..state import (
    AUDIO_DIR
)

# Monkey patch JSON encoder to handle NumPy types automatically
_original_default = json.JSONEncoder.default

def _numpy_default(self, obj):
    """Custom JSON encoder that converts NumPy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return _original_default(self, obj)

# Apply the patch
json.JSONEncoder.default = _numpy_default

# ============================================================================
# 🎯 CHEATING DETECTION CONFIGURATION (dari eye_detection.ipynb)
# ============================================================================

# Threshold Parameters
EYE_RATIO_RIGHT_LIMIT = 0.6
EYE_RATIO_LEFT_LIMIT = 1.6
HEAD_TURN_LEFT_LIMIT = 0.35
HEAD_TURN_RIGHT_LIMIT = 0.65
SCORE_HIGH_RISK = 20.0
SCORE_MEDIUM_RISK = 5.0

# Landmark Indices
LEFT_EYE = [33, 133, 468]
RIGHT_EYE = [362, 263, 473]
NOSE_TIP = 1
FACE_LEFT_EDGE = 234
FACE_RIGHT_EDGE = 454

print('\n🎯 Cheating Detection Configuration:')
print(f'   Eye Ratio Range: {EYE_RATIO_RIGHT_LIMIT} - {EYE_RATIO_LEFT_LIMIT}')
print(f'   Head Turn Range: {HEAD_TURN_LEFT_LIMIT} - {HEAD_TURN_RIGHT_LIMIT}')
print(f'   Risk Thresholds: >5% Medium, >20% High\n')

# ============================================================================
# 🔍 CHEATING DETECTION FUNCTIONS
# ============================================================================

def get_gaze_ratio(eye_points, landmarks):
    """Menghitung rasio posisi iris untuk eye tracking"""
    left_corner = np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y])
    right_corner = np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y])
    iris_center = np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y])

    dist_to_left = np.linalg.norm(iris_center - left_corner)
    dist_to_right = np.linalg.norm(iris_center - right_corner)

    if dist_to_right == 0:
        return 5.0

    ratio = dist_to_left / dist_to_right
    return ratio

def get_head_turn_ratio(landmarks):
    """Menghitung posisi relatif hidung untuk head pose detection"""
    nose = landmarks[NOSE_TIP].x
    left_edge = landmarks[FACE_LEFT_EDGE].x
    right_edge = landmarks[FACE_RIGHT_EDGE].x

    face_width = right_edge - left_edge
    nose_dist = nose - left_edge

    if face_width == 0:
        return 0.5

    relative_pos = nose_dist / face_width
    return relative_pos

def analyze_video_cheating_detection(video_path: str, show_progress=True):
    """
    Visual Analysis: Eye gaze, head pose, multiple face detection

    Returns:
        dict: {
            status, total_frames, suspicious_frames, cheating_score,
            verdict, confidence, details, plot_data
        }
    """
    if not os.path.exists(video_path):
        return {"status": "error", "message": f"File not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": "Cannot open video"}

    total_frames = 0
    suspicious_frames = 0
    eye_fail_count = 0
    head_fail_count = 0
    no_face_count = 0
    multiple_face_count = 0

    # Data for plotting
    gaze_ratios = []
    head_ratios = []
    frame_numbers = []
    confidence_scores = []
    face_counts = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as mesh, \
    mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.6
    ) as face_detector:

        while True:
            success, frame = cap.read()
            if not success:
                break

            total_frames += 1

            if show_progress and total_frames % 30 == 0:
                progress = (total_frames / total_video_frames) * 100
                print(f"   Processing cheating detection: {progress:.1f}% ({total_frames}/{total_video_frames} frames)", end='\r')

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # FACE DETECTION
            detection_results = face_detector.process(img_rgb)
            face_confidence = 0.0
            num_faces = 0

            if detection_results.detections:
                num_faces = len(detection_results.detections)

                if num_faces > 1:
                    multiple_face_count += 1

                face_confidence = detection_results.detections[0].score[0] * 100

            face_counts.append(num_faces)

            # FACE MESH (Eye Tracking)
            mesh_results = mesh.process(img_rgb)
            is_frame_suspicious = False

            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark

                # Eye Gaze Check
                left_ratio = get_gaze_ratio(LEFT_EYE, landmarks)
                right_ratio = get_gaze_ratio(RIGHT_EYE, landmarks)
                avg_gaze_ratio = (left_ratio + right_ratio) / 2

                if avg_gaze_ratio < EYE_RATIO_RIGHT_LIMIT or avg_gaze_ratio > EYE_RATIO_LEFT_LIMIT:
                    is_frame_suspicious = True
                    eye_fail_count += 1

                # Head Pose Check
                head_ratio = get_head_turn_ratio(landmarks)

                if not is_frame_suspicious:
                    if head_ratio < HEAD_TURN_LEFT_LIMIT or head_ratio > HEAD_TURN_RIGHT_LIMIT:
                        is_frame_suspicious = True
                        head_fail_count += 1

                gaze_ratios.append(avg_gaze_ratio)
                head_ratios.append(head_ratio)
                frame_numbers.append(total_frames)
                confidence_scores.append(face_confidence)
            else:
                is_frame_suspicious = True
                no_face_count += 1
                confidence_scores.append(0.0)

            # Multiple face = CHEATING
            if num_faces > 1:
                is_frame_suspicious = True

            if is_frame_suspicious:
                suspicious_frames += 1

    cap.release()

    if show_progress:
        print()

    cheating_score = 0
    if total_frames > 0:
        cheating_score = (suspicious_frames / total_frames) * 100

    multiple_face_pct = (multiple_face_count / total_frames) * 100 if total_frames > 0 else 0

    # Verdict
    verdict = "Safe"
    cheating_reasons = []

    if multiple_face_pct > 1.0:
        verdict = "High Risk"
        cheating_reasons.append(f"Multiple faces detected ({multiple_face_pct:.1f}% of frames)")
    elif cheating_score > SCORE_HIGH_RISK:
        verdict = "High Risk"
        cheating_reasons.append(f"High suspicious activity ({cheating_score:.1f}%)")
    elif cheating_score > SCORE_MEDIUM_RISK:
        verdict = "Medium Risk"
        cheating_reasons.append(f"Medium suspicious activity ({cheating_score:.1f}%)")

    duration = total_frames / fps if fps > 0 else 0

    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    min_confidence = np.min(confidence_scores) if confidence_scores else 0.0
    max_confidence = np.max(confidence_scores) if confidence_scores else 0.0

    return {
        "status": "success",
        "total_frames": total_frames,
        "suspicious_frames": suspicious_frames,
        "cheating_score": round(cheating_score, 2),
        "verdict": verdict,
        "cheating_reasons": cheating_reasons,
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "confidence": {
            "average": round(avg_confidence, 2),
            "min": round(min_confidence, 2),
            "max": round(max_confidence, 2)
        },
        "details": {
            "eye_fails": eye_fail_count,
            "head_fails": head_fail_count,
            "no_face": no_face_count,
            "multiple_faces": multiple_face_count
        },
        "plot_data": {
            "gaze_ratios": gaze_ratios,
            "head_ratios": head_ratios,
            "frame_numbers": frame_numbers,
            "confidence_scores": confidence_scores,
            "face_counts": face_counts
        }
    }
    
def analyze_speaker_diarization(video_path: str):
    """
    Speaker Diarization - Deteksi berapa banyak orang yang bicara
    Menggunakan Resemblyzer untuk voice embeddings + clustering

    ✅ FIXED: WebM audio extraction + CPU mode for cuDNN compatibility
    """
    try:
        from moviepy.editor import VideoFileClip
        import subprocess
        import torch

        # ✅ FIX 1: Ensure CPU mode for Resemblyzer
        torch.set_num_threads(4)  # Optimize CPU performance

        # ✅ FIX 2: Global VoiceEncoder with CPU device
        global voice_encoder
        if 'voice_encoder' not in globals() or voice_encoder is None:
            print("   Loading VoiceEncoder (CPU mode)...")
            from resemblyzer import VoiceEncoder
            voice_encoder = VoiceEncoder(device='cpu')  # ✅ Force CPU
            print("   ✅ VoiceEncoder loaded on CPU")

        encoder = voice_encoder

        # ✅ FIX 3: Ensure AUDIO_DIR exists
        os.makedirs(AUDIO_DIR, exist_ok=True)
        temp_audio = os.path.join(AUDIO_DIR, "temp_audio_diarization.wav")

        print("   Extracting audio from video...")

        # ✅ FIX 4: Try MoviePy first, fallback to FFmpeg for WebM
        audio_extracted = False

        # Method 1: Try MoviePy (works for most formats)
        try:
            video = VideoFileClip(video_path)

            if video.audio is None:
                print("   ⚠️  MoviePy: No audio track detected")
                video.close()
            else:
                print(f"   ℹ️  Audio duration: {video.audio.duration:.2f}s")
                video.audio.write_audiofile(
                    temp_audio,
                    fps=16000,
                    nbytes=2,
                    codec='pcm_s16le',  # ✅ Explicit codec for WebM
                    verbose=False,
                    logger=None
                )
                video.close()
                audio_extracted = True
                print("   ✅ Audio extracted via MoviePy")
        except Exception as moviepy_error:
            print(f"   ⚠️  MoviePy extraction failed: {str(moviepy_error)[:100]}")
            print("   🔄 Trying FFmpeg direct extraction...")

        # Method 2: Fallback to FFmpeg direct (better for WebM)
        if not audio_extracted:
            try:
                # ✅ FIX 5: Direct FFmpeg extraction (better for WebM/Opus)
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # Convert to PCM
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',  # Mono
                    '-y',  # Overwrite
                    temp_audio
                ]

                result = subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60
                )

                if result.returncode == 0 and os.path.exists(temp_audio):
                    audio_extracted = True
                    print("   ✅ Audio extracted via FFmpeg")
                else:
                    stderr_msg = result.stderr.decode()[:200] if result.stderr else "No error message"
                    print(f"   ❌ FFmpeg failed: {stderr_msg}")

            except FileNotFoundError:
                print("   ❌ FFmpeg not found in PATH")
                print("   💡 Install FFmpeg: https://ffmpeg.org/download.html")
            except subprocess.TimeoutExpired:
                print("   ❌ FFmpeg extraction timeout (>60s)")
            except Exception as ffmpeg_error:
                print(f"   ❌ FFmpeg extraction error: {str(ffmpeg_error)[:100]}")

        # ✅ FIX 6: Check if audio was extracted
        if not audio_extracted:
            return {
                "status": "no_audio",
                "message": "Failed to extract audio from video (WebM may require FFmpeg)",
                "num_speakers": 1,
                "is_cheating": False,
                "confidence": 50,
                "silhouette_score": 0,
                "total_segments": 0
            }

        # ✅ FIX 7: Validate extracted audio file
        if not os.path.exists(temp_audio):
            print("   ❌ Audio file not created")
            return {
                "status": "extraction_failed",
                "message": "Audio extraction failed - file not created",
                "num_speakers": 1,
                "is_cheating": False,
                "confidence": 50,
                "silhouette_score": 0,
                "total_segments": 0
            }

        audio_size = os.path.getsize(temp_audio)
        print(f"   ℹ️  Audio file size: {audio_size / 1024:.1f} KB")

        if audio_size < 1000:  # Less than 1KB
            print("   ⚠️  Audio file too small - likely empty")
            os.remove(temp_audio)
            return {
                "status": "audio_too_small",
                "message": "Extracted audio file is too small (empty or corrupt)",
                "num_speakers": 1,
                "is_cheating": False,
                "confidence": 50,
                "silhouette_score": 0,
                "total_segments": 0
            }

        print("   Loading and preprocessing audio...")
        try:
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(temp_audio)
        except Exception as preprocess_error:
            print(f"   ❌ Preprocessing error: {str(preprocess_error)[:100]}")
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return {
                "status": "preprocessing_failed",
                "message": f"Audio preprocessing failed: {str(preprocess_error)[:100]}",
                "num_speakers": 1,
                "is_cheating": False,
                "confidence": 50,
                "silhouette_score": 0,
                "total_segments": 0
            }

        if wav is None or len(wav) == 0:
            print("   ⚠️  Preprocessed audio is empty")
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return {
                "status": "audio_empty",
                "message": "Preprocessed audio is empty",
                "num_speakers": 1,
                "is_cheating": False,
                "confidence": 50,
                "silhouette_score": 0,
                "total_segments": 0
            }

        print(f"   ℹ️  Audio samples: {len(wav)} ({len(wav)/16000:.2f}s)")

        print("   Extracting speaker embeddings...")
        segment_duration = 0.5
        sample_rate = 16000
        segment_samples = int(segment_duration * sample_rate)

        embeddings = []
        timestamps = []

        step = segment_samples // 2
        for i in range(0, len(wav) - segment_samples, step):
            segment = wav[i:i + segment_samples]
            if len(segment) == segment_samples:
                # ✅ Ensure CPU processing
                with torch.no_grad():  # Disable gradient for inference
                    embed = encoder.embed_utterance(segment)
                embeddings.append(embed)
                timestamps.append(i / sample_rate)

        embeddings = np.array(embeddings)

        print(f"   Analyzing {len(embeddings)} audio segments...")

        if len(embeddings) == 0:
            print("   ⚠️  No valid audio segments extracted (audio too short)")
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return {
                "status": "no_segments",
                "message": "Audio too short - no valid segments (need >0.5s)",
                "num_speakers": 1,
                "is_cheating": False,
                "confidence": 50,
                "silhouette_score": 0,
                "total_segments": 0
            }

        if len(embeddings) < 2:
            num_speakers = 1 if len(embeddings) > 0 else 0
            confidence_score = 60.0
            silhouette = 0.0
        else:
            best_n_speakers = 1
            best_score = -1

            for n in range(2, min(6, len(embeddings))):
                clustering = AgglomerativeClustering(n_clusters=n, linkage='average')
                labels = clustering.fit_predict(embeddings)

                from sklearn.metrics import silhouette_score
                score = silhouette_score(embeddings, labels)

                if score > best_score and score > 0.2:
                    best_score = score
                    best_n_speakers = n

            if best_score > 0.2:
                num_speakers = best_n_speakers
                silhouette = best_score

                if silhouette >= 0.7:
                    confidence_score = 95.0
                elif silhouette >= 0.5:
                    confidence_score = 85.0
                elif silhouette >= 0.35:
                    confidence_score = 75.0
                else:
                    confidence_score = 60.0
            else:
                num_speakers = 1
                silhouette = best_score

                if best_score < 0.1:
                    confidence_score = 90.0
                elif best_score < 0.15:
                    confidence_score = 80.0
                else:
                    confidence_score = 70.0

        # Cleanup
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        is_cheating = num_speakers > 1

        print(f"   ✓ Detected {num_speakers} distinct speaker(s)")
        print(f"   📊 Confidence Score: {confidence_score:.1f}%")
        print(f"   📈 Silhouette Score: {silhouette:.3f}")
        print(f"   🔍 Total segments: {len(embeddings)}")

        if is_cheating:
            print(f"   ⚠️  WARNING: Multiple speakers detected!")

        return {
            "status": "success",
            "num_speakers": num_speakers,
            "total_segments": len(embeddings),
            "is_cheating": is_cheating,
            "confidence": round(confidence_score, 2),
            "silhouette_score": round(silhouette, 3),
            "message": f"Detected {num_speakers} distinct speaker(s) with {confidence_score:.1f}% confidence"
        }

    except Exception as e:
        print(f"   ❌ Speaker diarization error: {e}")
        import traceback
        print(f"   📋 Traceback:")
        traceback.print_exc()

        # Cleanup on error
        temp_audio_path = os.path.join(AUDIO_DIR, "temp_audio_diarization.wav")
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass

        return {
            "status": "error",
            "message": str(e)[:200],
            "num_speakers": 1,
            "is_cheating": False,
            "confidence": 50,
            "silhouette_score": 0,
            "total_segments": 0
        }

def comprehensive_cheating_detection(video_path: str):
    """
    Comprehensive Cheating Detection:
    - Visual: Face detection, eye gaze, head pose, multiple faces
    - Audio: Speaker diarization (multiple speakers)

    Expected for single-person interview: 1 face + 1 speaker
    Cheating if: >1 face OR >1 speaker

    Returns:
        dict: {
            visual: {cheating_score, suspicious_frames, cheating_reasons, confidence},
            audio: {num_speakers, confidence, silhouette_score},
            final_verdict: "Safe" | "Medium Risk" | "High Risk",
            final_avg_confidence: float,
            all_indicators: [...]
        }
    """
    print(f"\n{'='*60}")
    print(f"🎯 COMPREHENSIVE CHEATING DETECTION")
    print(f"   (Video Interview - Expected: 1 Person)")
    print(f"{'='*60}\n")

    # 1. VISUAL ANALYSIS
    print("👁️  STEP 1: Visual Analysis (Face Detection)")
    print("-" * 60)
    visual_result = analyze_video_cheating_detection(video_path, show_progress=True)

    # 2. SPEAKER DIARIZATION
    print("\n🔊 STEP 2: Speaker Diarization (Voice Analysis)")
    print("-" * 60)
    audio_result = analyze_speaker_diarization(video_path)

    # 3. COMBINE RESULTS
    print("\n📊 COMBINING RESULTS...")
    print("-" * 60)

    final_verdict = visual_result['verdict']
    all_indicators = visual_result.get('cheating_reasons', []).copy()

    # Add speaker diarization indicators
    if audio_result.get('status') == 'success':
        num_speakers = audio_result.get('num_speakers', 0)

        if audio_result.get('is_cheating'):
            all_indicators.append(
                f"Multiple speakers detected ({num_speakers} different voices, confidence: {audio_result.get('confidence', 0):.1f}%)"
            )
            final_verdict = "High Risk"

    # Calculate final average confidence
    visual_confidence = visual_result.get('confidence', {}).get('average', 0)
    audio_confidence = audio_result.get('confidence', 0)
    final_avg_confidence = round((visual_confidence + audio_confidence) / 2, 2)

    # Print final result
    print(f"\n{'='*60}")
    print(f"🎯 FINAL VERDICT: {final_verdict}")
    print(f"{'='*60}")

    if all_indicators:
        print(f"\n⚠️  Cheating Indicators Found:")
        for i, indicator in enumerate(all_indicators, 1):
            print(f"   {i}. {indicator}")
    else:
        print("\n✅ No suspicious activity detected")
        print("   ✓ Single person detected (visual)")
        if audio_result.get('status') == 'success':
            print(f"   ✓ Single speaker detected (audio, confidence: {audio_result.get('confidence', 0):.1f}%)")

    print(f"\n📋 Summary:")
    print(f"   Visual:")
    print(f"     • Confidence Score: {visual_confidence:.2f}%")
    print(f"     • Cheating Score: {visual_result['cheating_score']:.2f}%")
    print(f"     • Suspicious Frames: {visual_result['suspicious_frames']}")

    if audio_result.get('status') == 'success':
        print(f"   Audio:")
        print(f"     • Confidence Score: {audio_confidence:.2f}%")
        print(f"     • Number of Speakers: {audio_result['num_speakers']}")
        print(f"     • Silhouette Score: {audio_result.get('silhouette_score', 0):.3f}")

    print(f"\n   🎯 Final Average Confidence: {final_avg_confidence:.2f}%")
    print(f"{'='*60}\n")

    # Return dengan format yang lebih ringkas + final_avg_confidence
    return {
        "visual": {
            "cheating_score": visual_result.get('cheating_score', 0),
            "suspicious_frames": visual_result.get('suspicious_frames', 0),
            "cheating_reasons": visual_result.get('cheating_reasons', []),
            "confidence": visual_result.get('confidence', {
                "average": 0,
                "min": 0,
                "max": 0
            })
        },
        "audio": {
            "num_speakers": audio_result.get('num_speakers', 0),
            "confidence": audio_result.get('confidence', 0),
            "silhouette_score": audio_result.get('silhouette_score', 0)
        },
        "final_verdict": final_verdict,
        "final_avg_confidence": final_avg_confidence,
        "all_indicators": all_indicators
    }


print('✅ Cheating detection functions loaded')
print('   • analyze_video_cheating_detection() - Visual analysis')
print('   • analyze_speaker_diarization() - Audio analysis')
print('   • comprehensive_cheating_detection() - Full detection (with final_avg_confidence)\n')

def aggregate_cheating_results(assessment_results: List[dict]):
    """
    Aggregate cheating detection results dari assessment_results

    """
    if not assessment_results:
        return {
            "avg_cheating_score": 0,
            "avg_visual_confidence": 0,
            "avg_audio_confidence": 0,
            "avg_overall_confidence": 0,
            "total_suspicious_frames": 0,
            "avg_silhouette_score": 0,
            "verdict_distribution": {"Safe": 0, "Medium Risk": 0, "High Risk": 0},
            "final_aggregate_verdict": "No Data",
            "risk_level": "Unknown",
            "overall_performance_status": "Unknown",
            "questions_with_issues": [],
            "all_indicators": [],
            "summary": "No assessment data available for cheating analysis"
        }

    print(f"\n{'='*60}")
    print(f"📊 AGGREGATING CHEATING DETECTION RESULTS")
    print(f"   Total Questions: {len(assessment_results)}")
    print(f"{'='*60}\n")

    # Initialize accumulators
    total_cheating_score = 0
    total_visual_confidence = 0
    total_audio_confidence = 0
    total_overall_confidence = 0
    total_suspicious_frames = 0
    total_silhouette = 0

    verdict_counts = {
        "Safe": 0,
        "Medium Risk": 0,
        "High Risk": 0
    }

    all_indicators = []
    questions_with_issues = []

    valid_audio_count = 0
    valid_cheating_count = 0

    # Aggregate data from all assessment results
    for idx, assessment in enumerate(assessment_results, 1):
        question_id = assessment.get("id", f"question_{idx}")
        question_text = assessment.get("question", "Unknown question")
        result = assessment.get("result", {})

        # Extract cheating detection from result
        cheating_detection = result.get("cheating_detection", {})

        if not cheating_detection:
            continue

        valid_cheating_count += 1

        # Visual metrics
        visual = cheating_detection.get("visual", {})
        cheating_score = visual.get("cheating_score", 0)
        suspicious_frames = visual.get("suspicious_frames", 0)
        visual_conf = visual.get("confidence", {})
        visual_avg_conf = visual_conf.get("average", 0) if isinstance(visual_conf, dict) else 0

        total_cheating_score += cheating_score
        total_visual_confidence += visual_avg_conf
        total_suspicious_frames += suspicious_frames

        # Audio metrics
        audio = cheating_detection.get("audio", {})
        audio_confidence = audio.get("confidence", 0)
        audio_silhouette = audio.get("silhouette_score", 0)
        audio_speakers = audio.get("num_speakers", 0)

        if audio_speakers > 0:  # Valid audio analysis
            total_audio_confidence += audio_confidence
            total_silhouette += audio_silhouette
            valid_audio_count += 1

        # Overall confidence
        total_overall_confidence += cheating_detection.get("final_avg_confidence", 0)

        # Verdict distribution
        verdict = cheating_detection.get("final_verdict", "Safe")
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

        # Collect indicators
        indicators = cheating_detection.get("all_indicators", [])
        if indicators:
            questions_with_issues.append({
                "question_id": question_id,
                "question": question_text,
                "verdict": verdict,
                "cheating_score": cheating_score,
                "visual_confidence": visual_avg_conf,
                "audio_confidence": audio_confidence,
                "num_speakers": audio_speakers,
                "indicators": indicators
            })
            for indicator in indicators:
                all_indicators.append({
                    "question_id": question_id,
                    "question": question_text,
                    "indicator": indicator
                })

    if valid_cheating_count == 0:
        return {
            "avg_cheating_score": 0,
            "avg_visual_confidence": 0,
            "avg_audio_confidence": 0,
            "avg_overall_confidence": 0,
            "total_suspicious_frames": 0,
            "avg_silhouette_score": 0,
            "verdict_distribution": verdict_counts,
            "final_aggregate_verdict": "No Data",
            "risk_level": "Unknown",
            "overall_performance_status": "Unknown",
            "questions_with_issues": [],
            "all_indicators": [],
            "summary": "No valid cheating detection data found in assessment results"
        }

    # Calculate averages
    avg_cheating_score = round(total_cheating_score / valid_cheating_count, 2)
    avg_visual_confidence = round(total_visual_confidence / valid_cheating_count, 2)
    avg_audio_confidence = round(total_audio_confidence / valid_audio_count, 2) if valid_audio_count > 0 else 0
    avg_overall_confidence = round(total_overall_confidence / valid_cheating_count, 2)
    avg_silhouette_score = round(total_silhouette / valid_audio_count, 3) if valid_audio_count > 0 else 0

    # Determine overall performance status based on confidence score
    if avg_overall_confidence >= 80:
        overall_performance_status = "Very High"
    elif avg_overall_confidence >= 70:
        overall_performance_status = "Good"
    elif avg_overall_confidence >= 60:
        overall_performance_status = "Average"
    elif avg_overall_confidence >= 50:
        overall_performance_status = "Below Average"
    else:
        overall_performance_status = "Need Improvement"

    # Determine final aggregate verdict
    high_risk_count = verdict_counts.get("High Risk", 0)
    medium_risk_count = verdict_counts.get("Medium Risk", 0)
    safe_count = verdict_counts.get("Safe", 0)

    # Logic: If ANY question is High Risk OR >50% are Medium+ Risk, verdict is High Risk
    if high_risk_count > 0 or (medium_risk_count + high_risk_count) > valid_cheating_count / 2:
        final_aggregate_verdict = "High Risk"
        risk_level = "Critical"
    elif medium_risk_count > 0:
        final_aggregate_verdict = "Medium Risk"
        risk_level = "Warning"
    else:
        final_aggregate_verdict = "Safe"
        risk_level = "Clear"

    # Generate summary
    summary_parts = []
    summary_parts.append(f"Analyzed {valid_cheating_count} question(s) for cheating detection.")
    summary_parts.append(f"Average cheating score: {avg_cheating_score}%.")
    summary_parts.append(f"Overall confidence: {avg_overall_confidence}%.")

    if high_risk_count > 0:
        summary_parts.append(f"⚠️ {high_risk_count} question(s) flagged as HIGH RISK.")
    if medium_risk_count > 0:
        summary_parts.append(f"⚠️ {medium_risk_count} question(s) flagged as MEDIUM RISK.")
    if safe_count == valid_cheating_count:
        summary_parts.append(f"✅ All questions passed cheating detection.")

    if len(all_indicators) > 0:
        summary_parts.append(f"Total of {len(all_indicators)} cheating indicator(s) detected.")

    summary = " ".join(summary_parts)

    # Print results
    print(f"📊 Aggregate Metrics:")
    print(f"   • Average Cheating Score: {avg_cheating_score}%")
    print(f"   • Average Visual Confidence: {avg_visual_confidence}%")
    print(f"   • Average Audio Confidence: {avg_audio_confidence}%")
    print(f"   • Average Overall Confidence: {avg_overall_confidence}%")
    print(f"   • Total Suspicious Frames: {total_suspicious_frames}")
    if valid_audio_count > 0:
        print(f"   • Average Silhouette Score: {avg_silhouette_score}")
    print(f"   • Overall Performance Status: {overall_performance_status}")

    print(f"\n📈 Verdict Distribution:")
    print(f"   • Safe: {safe_count}")
    print(f"   • Medium Risk: {medium_risk_count}")
    print(f"   • High Risk: {high_risk_count}")

    print(f"\n🎯 FINAL AGGREGATE VERDICT: {final_aggregate_verdict} ({risk_level})")

    if questions_with_issues:
        print(f"\n⚠️  Questions with Issues ({len(questions_with_issues)}):")
        for q_issue in questions_with_issues:
            print(f"   • Q{q_issue['question_id']}: {q_issue['verdict']} (Cheating: {q_issue['cheating_score']}%)")
            for indicator in q_issue['indicators']:
                print(f"      - {indicator}")

    print(f"\n{'='*60}\n")

    return {
        "avg_cheating_score": avg_cheating_score,
        "avg_visual_confidence": avg_visual_confidence,
        "avg_audio_confidence": avg_audio_confidence,
        "avg_overall_confidence": avg_overall_confidence,
        "total_suspicious_frames": total_suspicious_frames,
        "avg_silhouette_score": avg_silhouette_score,
        "verdict_distribution": verdict_counts,
        "final_aggregate_verdict": final_aggregate_verdict,
        "risk_level": risk_level,
        "overall_performance_status": overall_performance_status,
        "questions_with_issues": questions_with_issues,
        "all_indicators": all_indicators,
        "summary": summary
    }


print('✅ Aggregate function loaded (with performance status)')
print('   • aggregate_cheating_results() - Now includes overall_performance_status\n')