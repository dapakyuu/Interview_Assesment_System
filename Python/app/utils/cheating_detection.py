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

###################################################################
# 1️⃣ SPEAKER DIARIZATION (SILERO)
###################################################################

def perform_speaker_diarization_silero(video_path):
    """
    Detect multiple speakers using Silero VAD (Voice Activity Detection)
    FIXED: Better algorithm to distinguish between natural pauses vs multiple speakers
    """
    try:
        print('   🎤 Performing speaker diarization (Silero VAD)...')
        # Load Silero VAD model
        try:
            model = load_silero_vad()
            print('   │ ✅ Silero VAD model loaded')
        except Exception as e:
            print(f'   │ ⚠️  Could not load Silero VAD: {str(e)[:50]}')
            return {
                'is_single_speaker': True,
                'speaker_count': 1,
                'duration': 0,
                'method': 'silero_vad_unavailable',
                'error': str(e)
            }

        # Try to load audio
        try:
            print('   │ Attempting to load audio...')
            waveform, sample_rate = torchaudio.load(video_path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            print(f'   │ ✅ Audio loaded: {waveform.shape[0]} channels @ {sample_rate}Hz')

        except Exception as e:
            print(f'   │ ⚠️  torchaudio load failed: {str(e)[:50]}')

            # Fallback: Use pydub + ffmpeg
            try:
                print('   │ Fallback: Using pydub to extract audio...')

                audio = AudioSegment.from_file(video_path)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                    samples = samples.mean(axis=1)

                samples = samples / 32768.0
                waveform = torch.from_numpy(samples).unsqueeze(0)
                sample_rate = audio.frame_rate

                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000

                print(f'   │ ✅ Audio extracted via pydub: {waveform.shape[0]} channels @ {sample_rate}Hz')

            except Exception as e2:
                print(f'   │ ⚠️  All audio loading methods failed')
                return {
                    'is_single_speaker': True,
                    'speaker_count': 1,
                    'duration': 0,
                    'method': 'audio_loading_failed',
                    'error': f'{str(e)[:30]} | {str(e2)[:30]}'
                }

        duration_seconds = waveform.shape[1] / sample_rate
        print(f'   │ ℹ️  Audio duration: {duration_seconds:.1f}s')

        # Apply Silero VAD
        print('   │ Analyzing speech patterns...')

        CHUNK_SIZE = int(sample_rate * 0.032)  # 32ms chunks
        chunks = waveform.squeeze(0).split(CHUNK_SIZE)

        speech_segments = []  # List of (start_idx, end_idx) tuples
        current_speech_start = None

        for i, chunk in enumerate(chunks):
            if len(chunk) < CHUNK_SIZE:
                chunk = torch.nn.functional.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

            try:
                speech_prob = model(chunk.unsqueeze(0), sample_rate)
                is_speech = speech_prob > 0.5

                if is_speech and current_speech_start is None:
                    # Start of speech segment
                    current_speech_start = i
                elif not is_speech and current_speech_start is not None:
                    # End of speech segment
                    speech_segments.append((current_speech_start, i))
                    current_speech_start = None
            except:
                pass

        # Close last segment if still open
        if current_speech_start is not None:
            speech_segments.append((current_speech_start, len(chunks)))

        print(f'   │ ℹ️  Detected {len(speech_segments)} speech segments')

        # ✅ FIXED: Better multiple speaker detection logic
        # Key indicators:
        # 1. Number of distinct speech segments (pauses > 2s indicate speaker change)
        # 2. Average segment length (short segments = conversation, long = monologue)
        # 3. Variance in segment lengths (varied = conversation, uniform = single speaker)

        if len(speech_segments) == 0:
            speaker_count = 1
            confidence = 'low'
            print(f'   │    ⚠️  No speech segments detected')
        else:
            # Calculate segment statistics
            segment_lengths = [(end - start) * 0.032 for start, end in speech_segments]  # in seconds
            avg_segment_length = np.mean(segment_lengths)
            segment_variance = np.var(segment_lengths)

            # Calculate silence gaps between segments
            silence_gaps = []
            for i in range(len(speech_segments) - 1):
                gap = (speech_segments[i+1][0] - speech_segments[i][1]) * 0.032
                silence_gaps.append(gap)

            long_pauses = sum(1 for gap in silence_gaps if gap > 2.0)  # Pauses > 2s

            print(f'   │ ℹ️  Avg segment: {avg_segment_length:.1f}s | Long pauses: {long_pauses}')

            # ✅ DECISION LOGIC (FIXED)
            # Single speaker indicators:
            # - Few long pauses (natural thinking/breathing)
            # - Relatively uniform segment lengths
            # - Average segment length > 3 seconds

            # Multiple speaker indicators:
            # - Many long pauses (turn-taking)
            # - High variance in segment lengths
            # - Many short segments (back-and-forth conversation)

            if duration_seconds < 30:
                # Short videos: likely single speaker
                speaker_count = 1
                confidence = 'medium'
            elif long_pauses < 5 and avg_segment_length > 3:
                # Few long pauses + long segments = single speaker monologue
                speaker_count = 1
                confidence = 'high'
            elif long_pauses > 15 and avg_segment_length < 2:
                # Many pauses + short segments = conversation
                speaker_count = 2
                confidence = 'high'
            elif len(speech_segments) > 30 and segment_variance > 5:
                # Many varied segments = possible conversation
                speaker_count = 2
                confidence = 'medium'
            else:
                # Default: assume single speaker
                speaker_count = 1
                confidence = 'medium'

            is_single_speaker = (speaker_count == 1)

            print(f'   │ ✅ Analysis complete: {speaker_count} speaker(s)')
            print(f'   │    Confidence: {confidence.upper()}')
            print(f'   │    Reasoning: {"Monologue pattern" if speaker_count == 1 else "Conversation pattern"}')

        return {
            'is_single_speaker': is_single_speaker,
            'speaker_count': speaker_count,
            'duration': round(duration_seconds, 2),
            'speech_segments': len(speech_segments),
            'avg_segment_length': round(avg_segment_length, 2) if len(speech_segments) > 0 else 0,
            'long_pauses': long_pauses if len(speech_segments) > 0 else 0,
            'method': 'silero_vad_fixed',
            'confidence': confidence
        }

    except Exception as e:
        print(f'   ⚠️  Silero VAD error: {str(e)}')
        traceback.print_exc()

        return {
            'is_single_speaker': True,
            'speaker_count': 1,
            'error': str(e),
            'method': 'silero_vad_exception'
        }

###################################################################
# 2️⃣ MEDIA PIPE EYE + GAZE DETECTION
###################################################################

def detect_eyes_in_video(video_path, sample_rate=5):
    """Detect eyes using MediaPipe - FIXED for compatibility"""
    try:
        # ✅ OpenCV dari MediaPipe (sudah compatible)
        print('   👁️  Eye detection analysis...')

        # MediaPipe solutions
        mp_face_detection = mp.solutions.face_detection

        # Open video dengan OpenCV
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print('   ⚠️  Could not open video file')
            return {
                'is_suspicious': False,
                'error': 'Video could not be opened',
                'message': 'Eye detection failed - video read error'
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0 or fps > 120:
            fps = 30  # Fallback

        if total_frames == 0:
            cap.release()
            print('   ⚠️  Could not determine total frames')
            return {
                'is_suspicious': False,
                'error': 'Could not determine frame count',
                'message': 'Eye detection skipped - frame count unknown'
            }

        frame_count = 0
        eye_detected_frames = 0
        eyes_open_frames = 0
        eyes_closed_frames = 0
        suspicious_frames = 0

        sample_interval = max(1, int(fps / sample_rate))

        print(f'   │ FPS: {fps:.1f} | Total Frames: {total_frames} | Interval: {sample_interval}')

        try:
            # ✅ MediaPipe FaceDetection
            with mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            ) as face_detection:

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % sample_interval != 0:
                        frame_count += 1
                        continue

                    try:
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Detect faces
                        results = face_detection.process(rgb_frame)

                        if results.detections:
                            eye_detected_frames += 1

                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                h, w, c = frame.shape

                                # Face position
                                face_center_y = (bbox.ymin + bbox.height) * h

                                # Check if looking down (suspicious)
                                if face_center_y > h * 0.6:
                                    suspicious_frames += 1

                                # Check eyes visibility
                                # MediaPipe detects 6 keypoints (left eye, right eye, nose, mouth, etc)
                                if len(detection.location_data.relative_keypoints) >= 2:
                                    eyes_open_frames += 1
                                else:
                                    eyes_closed_frames += 1

                    except Exception as e:
                        print(f'   │ ⚠️  Frame {frame_count} error: {str(e)[:40]}')
                        continue

                    frame_count += 1

                    # Progress update
                    if frame_count % (sample_interval * 30) == 0 and total_frames > 0:
                        progress = (frame_count / total_frames) * 100
                        print(f'   │ ⏳ Processing: {progress:.1f}%', end='\r')

        except Exception as e:
            print(f'   ⚠️  Face detection error: {str(e)}')

        finally:
            cap.release()

        # Calculate statistics
        total_sampled_frames = frame_count
        face_detection_rate = (eye_detected_frames / total_sampled_frames * 100) if total_sampled_frames > 0 else 0
        suspicious_rate = (suspicious_frames / eye_detected_frames * 100) if eye_detected_frames > 0 else 0

        print(f'\n   ✅ Eye Detection Complete:')
        print(f'      Face: {face_detection_rate:.1f}% | Eyes open: {eyes_open_frames} | Eyes closed: {eyes_closed_frames}')

        # Determine if suspicious
        is_suspicious = False
        suspicious_reasons = []

        if face_detection_rate < 50:
            is_suspicious = True
            suspicious_reasons.append("Face not consistently visible")

        if suspicious_rate > 30:
            is_suspicious = True
            suspicious_reasons.append("Frequent downward gaze (reading)")

        if eyes_closed_frames > eyes_open_frames and eyes_open_frames > 0:
            is_suspicious = True
            suspicious_reasons.append("Eyes frequently closed")

        return {
            'face_detection_rate': round(face_detection_rate, 2),
            'eyes_open_frames': eyes_open_frames,
            'eyes_closed_frames': eyes_closed_frames,
            'suspicious_gaze_rate': round(suspicious_rate, 2),
            'is_suspicious': is_suspicious,
            'suspicious_reasons': suspicious_reasons,
            'total_frames_analyzed': total_sampled_frames
        }

    except Exception as e:
        print(f'   ⚠️  Eye detection error: {str(e)}')
        traceback.print_exc()

        return {
            'is_suspicious': False,
            'error': str(e),
            'message': 'Eye detection failed - using conservative assessment'
        }

print('✅ Eye detection function loaded (Fixed)')

###################################################################
# 3️⃣ ADVANCED CHEATING DETECTION
###################################################################

def advanced_cheating_detection(video_path, transcription_text):
    """✅ FIXED: Proper cheating score calculation with baseline"""
    try:
        print('   🚨 Advanced Cheating Detection:')

        cheating_indicators = []
        cheating_score = 100  # ✅ START at 100 (assume clean), DEDUCT for suspicious behavior

        confidence_components = {
            'diarization_confidence': 0,
            'diarization_data_quality': 0,
            'eye_detection_confidence': 0,
            'eye_detection_coverage': 0,
            'text_pattern_confidence': 0,
            'text_pattern_diversity': 0,
            'audio_quality_confidence': 0,
            'audio_snr': 0
        }

        total_checks = 4

        # ============================================================
        # 1️⃣ DIARIZATION CHECK
        # ============================================================
        print('   │ 1️⃣  Speaker Diarization Check')
        diar_result = perform_speaker_diarization_silero(video_path)

        if 'confidence' in diar_result:
            conf_map = {'high': 90, 'medium': 70, 'low': 50}
            base_conf = conf_map.get(diar_result['confidence'], 50)

            duration = diar_result.get('duration', 0)
            speech_segments = diar_result.get('speech_segments', 0)
            avg_segment_length = diar_result.get('avg_segment_length', 0)

            data_quality = 50
            if duration > 10:
                data_quality += 20
            if speech_segments > 5:
                data_quality += 15
            if avg_segment_length > 2:
                data_quality += 15

            confidence_components['diarization_confidence'] = int(
                (base_conf * 0.7) + (data_quality * 0.3)
            )
            confidence_components['diarization_data_quality'] = data_quality

            print(f'   │    📊 Diarization: {confidence_components["diarization_confidence"]}% (base: {base_conf}, quality: {data_quality})')
        else:
            confidence_components['diarization_confidence'] = 50
            confidence_components['diarization_data_quality'] = 30

        # ✅ DEDUCT score if multiple speakers detected
        if not diar_result.get('is_single_speaker', True):
            cheating_indicators.append(
                f"Multiple speakers detected ({diar_result.get('speaker_count', 2)} speakers)"
            )
            cheating_score -= 40  # ✅ DEDUCT from 100
            print(f'   │    ⚠️  Multiple speakers: {diar_result.get("speaker_count", 2)} (-40 points)')
        else:
            print(f'   │    ✅ Single speaker confirmed')

        # ============================================================
        # 2️⃣ EYE DETECTION CHECK
        # ============================================================
        print('   │ 2️⃣  Eye Detection & Gaze Analysis')
        eye_result = detect_eyes_in_video(video_path, sample_rate=5)

        if 'face_detection_rate' in eye_result:
            face_rate = eye_result['face_detection_rate']
            frames_analyzed = eye_result.get('total_frames_analyzed', 0)
            eyes_open = eye_result.get('eyes_open_frames', 0)

            if face_rate > 90:
                base_eye_conf = 95
            elif face_rate > 75:
                base_eye_conf = 85
            elif face_rate > 60:
                base_eye_conf = 75
            elif face_rate > 45:
                base_eye_conf = 65
            elif face_rate > 30:
                base_eye_conf = 55
            else:
                base_eye_conf = 40

            coverage_quality = min(100, (frames_analyzed / 300) * 100)

            visibility_quality = 50
            if eyes_open > 100:
                visibility_quality = 90
            elif eyes_open > 50:
                visibility_quality = 75
            elif eyes_open > 20:
                visibility_quality = 60

            confidence_components['eye_detection_confidence'] = int(
                (base_eye_conf * 0.5) + (coverage_quality * 0.25) + (visibility_quality * 0.25)
            )
            confidence_components['eye_detection_coverage'] = int(coverage_quality)

            print(f'   │    📊 Eye Detection: {confidence_components["eye_detection_confidence"]}% (base: {base_eye_conf}, coverage: {coverage_quality:.0f}, visibility: {visibility_quality})')
        else:
            confidence_components['eye_detection_confidence'] = 50
            confidence_components['eye_detection_coverage'] = 30

        # ✅ DEDUCT score for suspicious eye behavior
        if eye_result.get('is_suspicious'):
            suspicious_count = 0

            if eye_result.get('face_detection_rate', 100) < 30:
                cheating_indicators.append("Eye detection: Very low face visibility")
                suspicious_count += 1
                cheating_score -= 15  # ✅ DEDUCT
                print(f'   │    ⚠️  Low face visibility (-15 points)')

            if eye_result.get('suspicious_gaze_rate', 0) > 50:
                cheating_indicators.append("Eye detection: Frequent downward gaze")
                suspicious_count += 1
                cheating_score -= 15  # ✅ DEDUCT
                print(f'   │    ⚠️  Downward gaze (-15 points)')

            if suspicious_count == 0:
                print(f'   │    ✅ Eye gaze analysis normal')
        else:
            print(f'   │    ✅ Eye gaze analysis normal')

        # ============================================================
        # 3️⃣ TEXT PATTERN CHECK
        # ============================================================
        print('   │ 3️⃣  Text Pattern Analysis')
        words = transcription_text.split()
        word_count = len(words)

        unique_words = len(set(word.lower() for word in words))
        repetition_ratio = (len(words) - unique_words) / len(words) if words else 1

        if word_count >= 100:
            base_text_conf = 95
        elif word_count >= 50:
            base_text_conf = 85
        elif word_count >= 30:
            base_text_conf = 75
        elif word_count >= 20:
            base_text_conf = 65
        elif word_count >= 10:
            base_text_conf = 55
        elif word_count >= 5:
            base_text_conf = 45
        else:
            base_text_conf = 30

        diversity_score = int((1 - repetition_ratio) * 100)

        confidence_components['text_pattern_confidence'] = int(
            (base_text_conf * 0.6) + (diversity_score * 0.4)
        )
        confidence_components['text_pattern_diversity'] = diversity_score

        print(f'   │    📊 Text Pattern: {confidence_components["text_pattern_confidence"]}% (base: {base_text_conf}, diversity: {diversity_score})')

        # ✅ DEDUCT score for suspicious text patterns
        if len(words) < 3:
            cheating_indicators.append("Answer extremely short (possible AI generation)")
            cheating_score -= 20  # ✅ DEDUCT
            print(f'   │    ⚠️  Extremely short answer: {len(words)} words (-20 points)')

        if repetition_ratio > 0.65:
            cheating_indicators.append(f"Very high word repetition ({repetition_ratio*100:.1f}%)")
            cheating_score -= 15  # ✅ DEDUCT
            print(f'   │    ⚠️  High repetition rate: {repetition_ratio*100:.1f}% (-15 points)')
        else:
            print(f'   │    ✅ Text pattern normal')

        # ============================================================
        # 4️⃣ AUDIO QUALITY CHECK
        # ============================================================
        print('   │ 4️⃣  Audio Quality Check')
        try:
            y, sr = librosa.load(video_path, sr=16000, duration=30)

            S = librosa.feature.melspectrogram(y=y, sr=sr)
            noise_level = np.mean(S)
            signal_level = np.max(S)
            snr = signal_level / (noise_level + 1e-10)

            if snr > 50:
                base_audio_conf = 95
            elif snr > 30:
                base_audio_conf = 85
            elif snr > 20:
                base_audio_conf = 75
            elif snr > 10:
                base_audio_conf = 65
            else:
                base_audio_conf = 50

            if noise_level < 20:
                noise_penalty = 0
            elif noise_level < 40:
                noise_penalty = 10
            elif noise_level < 60:
                noise_penalty = 20
            else:
                noise_penalty = 30

            final_audio_conf = max(30, base_audio_conf - noise_penalty)

            confidence_components['audio_quality_confidence'] = int(final_audio_conf)
            confidence_components['audio_snr'] = int(min(100, snr))

            print(f'   │    📊 Audio Quality: {final_audio_conf}% (SNR: {snr:.1f}, noise: {noise_level:.1f})')

            # ✅ DEDUCT score for high noise
            if noise_level > 80:
                cheating_indicators.append(f"Very high background noise detected")
                cheating_score -= 10  # ✅ DEDUCT
                print(f'   │    ⚠️  High noise level: {noise_level:.1f} (-10 points)')
            else:
                print(f'   │    ✅ Audio quality normal (noise: {noise_level:.1f})')

        except Exception as e:
            print(f'   │    ℹ️  Audio analysis skipped: {str(e)}')
            fallback_audio = 50 + min(20, word_count // 5)
            confidence_components['audio_quality_confidence'] = fallback_audio
            confidence_components['audio_snr'] = 30

        # ============================================================
        # ✅ FINALIZE CHEATING SCORE (ensure 0-100 range)
        # ============================================================
        cheating_score = max(0, min(100, cheating_score))

        # ✅ INVERT score: High score = High cheating risk
        # Current: 100 (clean) → Want: 0 (clean)
        cheating_score = 100 - cheating_score  # ✅ INVERT!

        # ============================================================
        # CALCULATE OVERALL CONFIDENCE SCORE
        # ============================================================
        weighted_confidence = (
            confidence_components['diarization_confidence'] * 0.25 +
            confidence_components['eye_detection_confidence'] * 0.25 +
            confidence_components['text_pattern_confidence'] * 0.25 +
            confidence_components['audio_quality_confidence'] * 0.25
        )

        quality_adjustment = (
            confidence_components['diarization_data_quality'] * 0.1 +
            confidence_components['eye_detection_coverage'] * 0.1 +
            confidence_components['text_pattern_diversity'] * 0.1 +
            confidence_components['audio_snr'] * 0.1
        ) / 4

        overall_confidence = min(100, weighted_confidence + quality_adjustment)

        if overall_confidence >= 85:
            confidence_level = "Very High"
        elif overall_confidence >= 75:
            confidence_level = "High"
        elif overall_confidence >= 60:
            confidence_level = "Medium"
        elif overall_confidence >= 45:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"

        # ✅ Determine cheating status (FIXED thresholds)
        is_cheating = cheating_score > 40  # ✅ Lower threshold (was 60)
        cheating_status = "Ya" if is_cheating else "Tidak"

        print(f'   │ 📊 Final Cheating Score: {cheating_score}/100')
        print(f'   │ 🎯 Overall Confidence: {overall_confidence:.1f}% ({confidence_level})')
        print(f'   │ 🚨 Cheating Detection: {cheating_status}')

        if cheating_indicators:
            print(f'   │ ⚠️  Indicators ({len(cheating_indicators)}):')
            for indicator in cheating_indicators:
                print(f'   │    - {indicator}')
        else:
            print(f'   │ ✅ No suspicious indicators found')

        return {
            'is_cheating': is_cheating,
            'cheating_status': cheating_status,
            'cheating_score': cheating_score,
            'indicators': cheating_indicators,
            'confidence_score': round(overall_confidence, 2),
            'confidence_level': confidence_level,
            'confidence_components': confidence_components,
            'details': {
                'diarization': diar_result,
                'eye_detection': eye_result,
                'word_count': len(words),
                'repetition_ratio': round(repetition_ratio, 3),
                'unique_words': unique_words,
                'diversity_score': diversity_score
            }
        }

    except Exception as e:
        print(f'   ⚠️  Cheating detection error: {str(e)}')
        return {
            'is_cheating': False,
            'cheating_status': 'Tidak',
            'cheating_score': 0,
            'indicators': [],
            'confidence_score': 0,
            'confidence_level': 'N/A',
            'confidence_components': {},
            'error': str(e)
        }

###################################################################
# 4️⃣ AGGREGATE ANALYSIS
###################################################################

def calculate_aggregate_cheating_analysis(assessment_results):
    """Enhanced aggregate analysis with MORE LENIENT thresholds"""
    if not assessment_results:
        return {
            "overall_cheating_status": "Tidak",
            "overall_cheating_score": 0,
            "total_videos": 0,
            "videos_flagged": 0,
            "flagged_percentage": 0,
            "confidence_level": "N/A",
            "risk_level": "LOW RISK",
            "recommendation": "No data to analyze",
            "summary": "No assessment results available",
            "average_confidence_score": 0,
            "overall_confidence_level": "N/A"
        }

    total_videos = len(assessment_results)
    cheating_scores = []
    confidence_scores = []
    videos_flagged = 0
    flagged_video_ids = []
    cheating_indicators_summary = {}

    for video in assessment_results:
        result = video.get("result", {})

        cheating_score = result.get("cheating_score", 0)
        cheating_scores.append(cheating_score)

        confidence_score = result.get("cheating_confidence_score", 0)
        confidence_scores.append(confidence_score)

        if result.get("cheating_detection") == "Ya":
            videos_flagged += 1
            flagged_video_ids.append(video.get("id"))

            indicators = result.get("cheating_details", {}).get("diarization", {})
            if not indicators.get("is_single_speaker", True):
                cheating_indicators_summary["multiple_speakers"] = \
                    cheating_indicators_summary.get("multiple_speakers", 0) + 1

            eye_data = result.get("cheating_details", {}).get("eye_detection", {})
            if eye_data.get("is_suspicious", False):
                cheating_indicators_summary["suspicious_eye_behavior"] = \
                    cheating_indicators_summary.get("suspicious_eye_behavior", 0) + 1

    avg_cheating_score = sum(cheating_scores) / total_videos if total_videos > 0 else 0
    avg_confidence_score = sum(confidence_scores) / total_videos if total_videos > 0 else 0
    max_cheating_score = max(cheating_scores) if cheating_scores else 0

    flagged_percentage = (videos_flagged / total_videos * 100) if total_videos > 0 else 0

    if avg_confidence_score >= 85:
        overall_confidence_level = "Very High"
    elif avg_confidence_score >= 75:
        overall_confidence_level = "High"
    elif avg_confidence_score >= 60:
        overall_confidence_level = "Medium"
    elif avg_confidence_score >= 45:
        overall_confidence_level = "Low"
    else:
        overall_confidence_level = "Very Low"

    # ✅ FIXED: More lenient decision thresholds
    if flagged_percentage >= 70 or avg_cheating_score > 65 or max_cheating_score > 80:
        overall_status = "Ya"
        confidence = "High"
        risk_level = "HIGH RISK"
        recommendation = "TIDAK LULUS - Strong evidence of cheating"
    elif flagged_percentage >= 50 or avg_cheating_score >= 50:
        overall_status = "Ya"
        confidence = "Medium"
        risk_level = "MEDIUM RISK"
        recommendation = "PERTIMBANGAN - Suspicious patterns detected"
    else:
        overall_status = "Tidak"
        confidence = "High" if flagged_percentage == 0 else "Medium"
        risk_level = "LOW RISK"
        recommendation = "LULUS - No significant cheating indicators"

    # Summary text
    if overall_status == "Ya":
        summary_text = f"Detected suspicious patterns in {videos_flagged} out of {total_videos} video(s). "
        if cheating_indicators_summary:
            summary_text += "Main concerns: " + ", ".join([
                f"{k.replace('_', ' ')}: {v}"
                for k, v in cheating_indicators_summary.items()
            ]) + "."
    else:
        summary_text = f"All {total_videos} video(s) passed cheating detection without significant concerns."

    return {
        "overall_cheating_status": overall_status,
        "overall_cheating_score": round(avg_cheating_score, 2),
        "average_confidence_score": round(avg_confidence_score, 2),
        "overall_confidence_level": overall_confidence_level,
        "max_cheating_score": max_cheating_score,
        "total_videos": total_videos,
        "videos_flagged": videos_flagged,
        "flagged_percentage": round(flagged_percentage, 2),
        "confidence_level": confidence,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "summary": summary_text,
        "pattern_analysis": {
            "indicators_summary": cheating_indicators_summary,
            "score_distribution": {
                "min": min(cheating_scores) if cheating_scores else 0,
                "max": max_cheating_score,
                "avg": round(avg_cheating_score, 2)
            }
        }
    }

###################################################################
# 5️⃣ IMPROVEMENT TIPS
###################################################################

def get_confidence_improvement_tips(confidence_components):
    """
    Provides actionable tips to improve confidence score
    """
    tips = []

    diar = confidence_components.get('diarization_confidence', 0)
    eye = confidence_components.get('eye_detection_confidence', 0)
    text = confidence_components.get('text_pattern_confidence', 0)
    audio = confidence_components.get('audio_quality_confidence', 0)

    if diar < 80:
        tips.append({
            'component': 'Speaker Detection',
            'current': f'{diar:.1f}%',
            'tips': [
                '✅ Record in quiet environment',
                '✅ Ensure only one person speaks',
                '✅ Avoid background conversations'
            ]
        })

    if eye < 80:
        tips.append({
            'component': 'Eye Detection',
            'current': f'{eye:.1f}%',
            'tips': [
                '✅ Position camera at eye level',
                '✅ Good lighting on face',
                '✅ Look at camera frequently',
                '✅ Avoid reading from notes'
            ]
        })

    if text < 80:
        tips.append({
            'component': 'Text Pattern',
            'current': f'{text:.1f}%',
            'tips': [
                '✅ Speak more (aim for 50+ words)',
                '✅ Use varied vocabulary',
                '✅ Avoid repeating same words',
                '✅ Speak clearly and naturally'
            ]
        })

    if audio < 80:
        tips.append({
            'component': 'Audio Quality',
            'current': f'{audio:.1f}%',
            'tips': [
                '✅ Use good microphone',
                '✅ Record in quiet room',
                '✅ Reduce background noise',
                '✅ Maintain consistent volume'
            ]
        })

    return tips