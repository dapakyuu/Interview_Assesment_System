# app/utils/llm_evaluator.py

import os
import re
import json as json_module
from huggingface_hub import InferenceClient
import math

from dotenv import load_dotenv
load_dotenv()

# ✅ HuggingFace API Token
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Replace with your actual token
os.environ["HF_TOKEN"] = HF_TOKEN

# Initialize Inference Client
print('📥 Initializing HuggingFace Inference API...')
print('ℹ️  Using meta-llama/Llama-3.1-8B-Instruct via Inference API')

client = InferenceClient(api_key=HF_TOKEN)

print('✅ Inference API initialized successfully\n')

def evaluate_with_llm(transcription_text: str, question: str, position_id: int):
    """
    Evaluate interview answer using deterministic LLM evaluation with confidence scoring.
    NOW WITH LOG PROBABILITIES SUPPORT + BOOST SYSTEM for confidence enhancement.
    """
    try:
        # Construct evaluation prompt
        user_message = f"""You are an expert interview evaluator about programming and machine learning. You must provide objective, consistent scores based on explicit criteria and formulas.

**INTERVIEW QUESTION**: "{question}"

**CANDIDATE'S ANSWER**: "{transcription_text}"

**EVALUATION RUBRIC WITH FORMULAS**:

1. **KUALITAS JAWABAN (Quality of Answer)** [1-100]:

   Base Score Formula:
   - If answer addresses question with examples/details: BASE = 85
   - If answer addresses question adequately: BASE = 75
   - If answer is brief but relevant: BASE = 65
   - If answer is unclear/irrelevant: BASE = 45

   Adjustments:
   - Provides specific examples: +5 to +15
   - Shows deep understanding: +5 to +10
   - Lacks depth: -10 to -20
   - Vague/incomplete: -15 to -25

   MINIMUM for acceptable answers: 70

2. **KOHERENSI (Coherence)** [1-100]:

   Formula:
   - Logical flow, well-structured: BASE = 85
   - Adequate structure: BASE = 75
   - Some inconsistency: BASE = 65
   - Disorganized: BASE = 45

   Adjustments:
   - Clear progression: +5 to +10
   - Smooth transitions: +5 to +10
   - Contradictory statements: -15 to -25
   - Jumps between topics: -10 to -20

   MINIMUM for coherent answers: 70

3. **RELEVANSI (Relevance)** [1-100]:

   Formula:
   - Directly answers the question: BASE = 85
   - Addresses most aspects: BASE = 75
   - Partially relevant: BASE = 65
   - Off-topic: BASE = 45

   Adjustments:
   - Covers all question aspects: +10 to +20
   - Provides context: +5 to +10
   - Deviates from topic: -15 to -25

   MINIMUM for on-topic answers: 70

**CALCULATION STEPS**:
1. Analyze the answer content and structure
2. Calculate base scores using formulas
3. Apply adjustments

**OUTPUT FORMAT** (JSON only, no explanation):
{{
  "kualitas_jawaban": <integer 1-100>,
  "koherensi": <integer 1-100>,
  "relevansi": <integer 1-100>,
  "analysis": "<2-3 sentence justification with reasoning>"
}}
"""

        # Calculate word count for boost system
        word_count = len(transcription_text.split())

        print(f'│ 🤖 LLM Evaluation...')
        print(f'│ 📝 Answer length: {len(transcription_text)} chars ({word_count} words)')

        # ✅ API Call with logprobs enabled
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly objective interview evaluator about programming and machine learning. Always respond with valid JSON only, no markdown."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=600,
            temperature=0.1,
            top_p=0.9,
            logprobs=True,        # ✅ ENABLE LOG PROBABILITIES
            top_logprobs=3        # ✅ Get top 3 alternative tokens for each position
        )

        # Extract response text
        response = completion.choices[0].message.content.strip()
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'\s*```$', '', response)

        print(f'│ 📨 API Response received ({len(response)} chars)')

        # ============================================================
        # ✅ EXTRACT LOGPROBS DATA (NEW!)
        # ============================================================
        logprobs_data = None
        raw_token_confidence = None
        raw_avg_probability = None

        try:
            if hasattr(completion.choices[0], 'logprobs') and completion.choices[0].logprobs:
                logprobs_obj = completion.choices[0].logprobs

                # Check if content exists
                if hasattr(logprobs_obj, 'content') and logprobs_obj.content:
                    logprobs_data = logprobs_obj.content

                    # Calculate average log probability
                    token_logprobs = [token.logprob for token in logprobs_data if hasattr(token, 'logprob')]

                    if token_logprobs:
                        avg_logprob = sum(token_logprobs) / len(token_logprobs)
                        # Convert log probability to percentage confidence
                        raw_avg_probability = math.exp(avg_logprob)
                        raw_token_confidence = round(raw_avg_probability * 100, 2)

                        print(f'│ 🎯 Raw Logprobs extracted: {len(token_logprobs)} tokens')
                        print(f'│ 📊 Avg log prob: {avg_logprob:.4f}')
                        print(f'│ ✨ Raw token confidence: {raw_token_confidence}%')
                    else:
                        print(f'│ ⚠️  Logprobs available but no token data')
                else:
                    print(f'│ ⚠️  Logprobs object has no content')
            else:
                print(f'│ ⚠️  No logprobs in API response (may not be supported)')
        except Exception as logprob_error:
            print(f'│ ⚠️  Logprobs extraction failed: {str(logprob_error)}')
            # Continue without logprobs - non-critical feature

        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            evaluation = json_module.loads(json_str)
        else:
            raise ValueError("No valid JSON found in API response")

        # Validate scores
        required_keys = ['kualitas_jawaban', 'koherensi', 'relevansi']
        for key in required_keys:
            if key not in evaluation:
                raise ValueError(f"Missing required key: {key}")
            evaluation[key] = max(1, min(100, int(evaluation[key])))

        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        print(f'│ 📊 LLM Scores:')
        print(f'│    • Quality: {evaluation["kualitas_jawaban"]}/100')
        print(f'│    • Coherence: {evaluation["koherensi"]}/100')
        print(f'│    • Relevance: {evaluation["relevansi"]}/100')

        # Calculate total score (using boosted model confidence)
        total = round((
            evaluation["kualitas_jawaban"] +
            evaluation["koherensi"] +
            evaluation["relevansi"]
        ) / 3)

        print(f'│ ✅ Total Score: {total}/100')

        # ✅ Return with logprobs data and boost info
        result = {
            "scores": {
                "kualitas_jawaban": evaluation["kualitas_jawaban"],
                "koherensi": evaluation["koherensi"],
                "relevansi": evaluation["relevansi"],
                "confidence_score": raw_token_confidence
            },
            "total": total,
            "analysis": evaluation.get('analysis', 'No analysis provided'),
            # 🆕 Logprobs data
            "logprobs_confidence": raw_token_confidence,
            "logprobs_probability": raw_avg_probability,
            "logprobs_available": logprobs_data is not None,
        }

        return result

    except Exception as e:
        print(f'│ ⚠️  LLM evaluation failed: {str(e)}')
        print(f'│ 🔄 Falling back to rule-based assessment...')

        # Fallback assessment
        word_count = len(transcription_text.split())

        # Simple heuristic scoring
        if word_count > 100:
            quality_score = 75
            coherence_score = 70
            relevance_score = 70
            model_confidence = 60
        elif word_count > 50:
            quality_score = 65
            coherence_score = 65
            relevance_score = 65
            model_confidence = 55
        elif word_count > 20:
            quality_score = 55
            coherence_score = 55
            relevance_score = 55
            model_confidence = 50
        else:
            quality_score = 40
            coherence_score = 35
            relevance_score = 35
            model_confidence = 50

        total = round((quality_score + coherence_score + relevance_score) / 3)

        return {
            "scores": {
                "kualitas_jawaban": quality_score,
                "koherensi": coherence_score,
                "relevansi": relevance_score,
                "confidence_score": model_confidence
            },
            "total": total,
            "analysis": f"Fallback rule-based assessment (word count: {word_count}). LLM evaluation unavailable: {str(e)}",
            # Fallback has no logprobs or boost
            "logprobs_confidence": None,
            "logprobs_probability": None,
            "logprobs_available": False,
        }
        
def summarize_llm_analysis_batch(assessment_results):
    """
    Generate overall summary from all assessments

    ✅ OPTIMIZED: If only 1 video, reuse existing analysis_llm instead of calling LLM again
    """
    try:
        if not assessment_results:
            return {
                "kesimpulan_llm": "Tidak ada hasil penilaian yang tersedia.",
                "rata_rata_confidence_score": 0,
                "avg_total_llm": 0,
                "avg_logprobs_confidence": None,
                "final_score_llm": 0
            }

        # Calculate averages
        confidence_scores = []
        total_scores = []
        logprobs_confidences = []

        for result in assessment_results:
            assessment = result.get('result', {}).get('penilaian', {})
            confidence_scores.append(assessment.get('confidence_score', 0))
            total_scores.append(assessment.get('total', 0))

            # Extract logprobs confidence if available
            lp_conf = assessment.get('logprobs_confidence')
            if lp_conf is not None:
                logprobs_confidences.append(lp_conf)

        avg_confidence = round(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0
        avg_total = round(sum(total_scores) / len(total_scores)) if total_scores else 0

        # Calculate average logprobs confidence
        avg_logprobs_confidence = None
        if logprobs_confidences:
            avg_logprobs_confidence = round(sum(logprobs_confidences) / len(logprobs_confidences), 2)

        # Determine final score
        projectScore = 100
        final_score = projectScore * 0.7 + avg_total * 0.3

        # ✅ NEW: If only 1 video, reuse existing analysis instead of calling LLM
        if len(assessment_results) == 1:
            print(f'\n{"="*70}')
            print(f'📋 Single Video Assessment - Reusing Existing Analysis')
            print(f'{"="*70}')
            print(f'ℹ️  Only 1 video detected - skipping LLM summary generation')
            print(f'✅ Using existing analysis_llm from video assessment')

            # Get existing analysis from the single video
            single_assessment = assessment_results[0].get('result', {}).get('penilaian', {})
            existing_analysis = single_assessment.get('analisis_llm', '')

            # Format as summary
            if existing_analysis:
                kesimpulan_llm = f"Assessment Summary: {existing_analysis}"
            else:
                # Fallback if no analysis
                quality = single_assessment.get('kualitas_jawaban', 0)
                coherence = single_assessment.get('koherensi', 0)
                relevance = single_assessment.get('relevansi', 0)
                total = single_assessment.get('total', 0)

                kesimpulan_llm = (
                    f"Candidate demonstrated performance with total score of {total}/100. "
                    f"Quality: {quality}/100, Coherence: {coherence}/100, Relevance: {relevance}/100."
                )

            print(f'   📊 Score: {avg_total}/100')
            print(f'   ✨ Analysis reused successfully')
            print(f'{"="*70}\n')

            return {
                "kesimpulan_llm": kesimpulan_llm,
                "rata_rata_confidence_score": avg_confidence,
                "avg_total_llm": avg_total,
                "final_score_llm": final_score,
                "avg_logprobs_confidence": avg_logprobs_confidence,
                "reused_single_analysis": True  # ✅ Flag untuk tracking
            }

        # ✅ Multiple videos: Generate comprehensive LLM summary
        print(f'\n{"="*70}')
        print(f'🤖 Generating Batch LLM Summary...')
        print(f'{"="*70}')
        print(f'📊 Processing {len(assessment_results)} video assessments')
        print(f'📈 Average Score: {avg_total}/100')
        if avg_logprobs_confidence is not None:
            print(f'✨ Avg Logprobs Confidence: {avg_logprobs_confidence}%')

        # Prepare assessment summary for multiple videos
        summary_lines = []
        for idx, result in enumerate(assessment_results, 1):
            assessment = result.get('result', {}).get('penilaian', {})
            question = result.get('question', f'Question {idx}')

            summary_lines.append(
                f"Video {idx}: Total {assessment.get('total', 0)}/100 "
                f"(Quality: {assessment.get('kualitas_jawaban', 0)}, "
                f"Coherence: {assessment.get('koherensi', 0)}, "
                f"Relevance: {assessment.get('relevansi', 0)})"
            )

        assessment_summary = "\n".join(summary_lines)

        # Detect language from first result
        source_language = assessment_results[0].get('result', {}).get('metadata', {}).get('source_language', 'English')

        # Generate LLM summary prompt
        user_message = f"""Based on the following interview assessment results, provide a comprehensive summary in {source_language} (2-3 paragraphs, ~150-200 words).

**ASSESSMENT RESULTS**:
{assessment_summary}

**AVERAGES**:
- Average Total Score: {avg_total}/100

**INSTRUCTIONS**:
1. Summarize the candidate's overall performance across all {len(assessment_results)} video interviews
2. Highlight consistent strengths and areas for improvement
3. Be objective, constructive, and professional
4. Consider both technical competence and communication skills

Respond with plain text summary only (no JSON, no markdown formatting)."""

        print(f'🤖 Calling LLM to generate comprehensive summary...')

        # API Call with logprobs
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert interview analyst. Provide comprehensive, objective assessments. Respond with plain text only, no JSON or markdown."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=500,
            temperature=0.3,
            top_p=0.9,
            logprobs=True,
            top_logprobs=3
        )

        # Extract summary
        kesimpulan_llm = completion.choices[0].message.content.strip()
        kesimpulan_llm = re.sub(r'^```.*?\n', '', kesimpulan_llm)
        kesimpulan_llm = re.sub(r'\n```$', '', kesimpulan_llm)

        # Extract summary logprobs
        summary_logprobs_confidence = None
        try:
            if hasattr(completion.choices[0], 'logprobs') and completion.choices[0].logprobs:
                logprobs_obj = completion.choices[0].logprobs
                if hasattr(logprobs_obj, 'content') and logprobs_obj.content:
                    token_logprobs = [token.logprob for token in logprobs_obj.content if hasattr(token, 'logprob')]
                    if token_logprobs:
                        avg_logprob = sum(token_logprobs) / len(token_logprobs)
                        summary_logprobs_confidence = round(math.exp(avg_logprob) * 100, 2)
                        print(f'✨ Summary logprobs confidence: {summary_logprobs_confidence}%')
        except Exception as e:
            print(f'⚠️  Summary logprobs extraction failed: {str(e)}')

        print(f'✅ LLM Summary generated successfully')
        print(f'   Length: {len(kesimpulan_llm)} characters')
        print(f'   Words: {len(kesimpulan_llm.split())}')
        print(f'{"="*70}\n')

        return {
            "kesimpulan_llm": kesimpulan_llm,
            "rata_rata_confidence_score": avg_confidence,
            "avg_total_llm": avg_total,
            "final_score_llm": final_score,
            "avg_logprobs_confidence": avg_logprobs_confidence,
            "summary_logprobs_confidence": summary_logprobs_confidence,  # ✅ Separate summary confidence
            "reused_single_analysis": False  # ✅ Flag untuk tracking
        }

    except Exception as e:
        print(f'❌ LLM summary generation failed: {str(e)}')
        print(f'🔄 Using fallback summary...')

        # Fallback summary
        return {
            "kesimpulan_llm": f"Kandidat menunjukkan performa dengan rata-rata skor {avg_total}/100 dari {len(assessment_results)} video interview. "
                             f"(LLM summary unavailable: {str(e)[:100]})",
            "rata_rata_confidence_score": avg_confidence,
            "avg_total_llm": avg_total,
            "final_score_llm": final_score,
            "avg_logprobs_confidence": avg_logprobs_confidence,
            "summary_logprobs_confidence": None,
            "reused_single_analysis": False
        }