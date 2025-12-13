# MIT License

Copyright (c) 2025 Interview Assessment System Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Third-Party Licenses

This project uses the following open-source libraries:

### Python Libraries

**faster-whisper (Systran)**

- License: MIT
- URL: https://github.com/SYSTRAN/faster-whisper
- Purpose: Speech-to-text transcription (CTranslate2 implementation)

**MediaPipe (Google)**

- License: Apache 2.0
- URL: https://github.com/google/mediapipe
- Purpose: Face mesh detection (468 landmarks)

**Resemblyzer**

- License: MIT
- URL: https://github.com/resemble-ai/Resemblyzer
- Purpose: Speaker diarization and voice encoding

**FastAPI**

- License: MIT
- URL: https://github.com/tiangolo/fastapi
- Purpose: REST API backend framework

**PyTorch**

- License: BSD-3-Clause
- URL: https://github.com/pytorch/pytorch
- Purpose: Deep learning framework (for models)

**OpenCV (cv2)**

- License: Apache 2.0
- URL: https://github.com/opencv/opencv-python
- Purpose: Video processing and frame extraction

**NumPy**

- License: BSD-3-Clause
- URL: https://github.com/numpy/numpy
- Purpose: Numerical computations

**FFmpeg**

- License: LGPL v2.1+ / GPL v2+ (depending on build)
- URL: https://ffmpeg.org
- Purpose: Audio extraction from video files

**Uvicorn**

- License: BSD-3-Clause
- URL: https://github.com/encode/uvicorn
- Purpose: ASGI server for FastAPI

---

## API Services

**Hugging Face Inference API**

- Terms: https://huggingface.co/terms-of-service
- Privacy: https://huggingface.co/privacy
- Used for: LLM assessment (Meta-Llama 3.1-8B-Instruct)
- Tier: FREE tier (1000 requests/hour limit)
- Data: Requests sent to HF servers, not stored by us

**DeepL API**

- Terms: https://www.deepl.com/pro-license
- Privacy: https://www.deepl.com/privacy
- Used for: EN ↔ ID translation
- Tier: FREE tier (500,000 characters/month)
- Data: Text sent to DeepL servers for translation

**Note on External APIs:**

- When you use this system, video transcripts and text may be sent to external APIs (HuggingFace, DeepL)
- Ensure you have proper consent from interview subjects
- Consider using local models if data privacy is critical
- You can disable translation by using single language mode

---

## AI Models

**Whisper Large-v3 (OpenAI)**

- License: MIT
- URL: https://github.com/openai/whisper
- Model: openai/whisper-large-v3
- Implementation: faster-whisper (CTranslate2)
- Purpose: Multilingual speech recognition
- Commercial use: ✅ Allowed (MIT license)

**Meta-Llama 3.1-8B-Instruct (Meta)**

- License: Llama 3.1 Community License
- URL: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Terms: https://llama.meta.com/llama-downloads/
- Purpose: Interview assessment and evaluation
- Commercial use: ✅ Allowed with <700M monthly active users
- Usage: Via HuggingFace Inference API (FREE tier)

**MediaPipe Face Mesh (Google)**

- License: Apache 2.0
- URL: https://github.com/google/mediapipe
- Purpose: Facial landmark detection (468 points)
- Commercial use: ✅ Allowed (Apache 2.0)

**Resemblyzer Voice Encoder**

- License: MIT
- URL: https://github.com/resemble-ai/Resemblyzer
- Model: GE2E (Generalized End-to-End)
- Purpose: Speaker diarization
- Commercial use: ✅ Allowed (MIT license)

---

## Attribution

If you use this software in your research, academic work, or publications, please cite:

```bibtex
@software{interview_assessment_system_2025,
  title={AI-Powered Interview Assessment System},
  author={Interview Assessment System Contributors},
  year={2025},
  month={December},
  version={1.0.0},
  url={https://github.com/YourUsername/Interview_Assesment_System},
  note={Open-source interview assessment using Whisper, Llama 3.1, and MediaPipe}
}
```

**Academic Use:**

- You may use this system for research purposes
- Please acknowledge the underlying models (Whisper, Llama, MediaPipe)
- Share findings and improvements with the community

**Commercial Use:**

- ✅ Allowed under MIT license
- ✅ No attribution required (but appreciated)
- ⚠️ Check Llama 3.1 license restrictions (<700M MAU)
- ⚠️ API rate limits apply (HF FREE tier, DeepL FREE tier)

---

## Data Privacy & Security

### User Responsibilities

**Legal Compliance:**

- ✅ Obtain informed consent from interview subjects before recording
- ✅ Comply with data protection laws (GDPR, CCPA, PIPEDA, etc.)
- ✅ Implement appropriate security measures for video storage
- ✅ Delete recordings after retention period expires
- ✅ Inform subjects about AI processing and external APIs used

**Data Handling:**

- Videos are processed locally on your server
- Transcripts sent to HuggingFace API (for LLM assessment)
- Text sent to DeepL API (for translation)
- Results stored locally in `results/` folder
- No data sent to project maintainers

### Our Commitments

**Privacy:**

- ❌ We don't collect any user data
- ❌ No telemetry or analytics tracking
- ❌ No phone-home functionality
- ✅ All processing happens on your infrastructure
- ✅ You control where data is stored

**Security:**

- ✅ HTTPS for API calls (HF, DeepL)
- ✅ Session-based file isolation (`uploads/{session_id}/`)
- ✅ Automatic temp file cleanup
- ⚠️ No built-in authentication (add your own if needed)
- ⚠️ No encryption at rest (implement if required)

**Recommendations for Production:**

1. **Add authentication** - Implement API keys or OAuth2
2. **Enable HTTPS** - Use reverse proxy (Nginx) with SSL certificates
3. **Encrypt storage** - Use encrypted filesystems for sensitive data
4. **Rate limiting** - Prevent abuse and control costs
5. **Audit logging** - Track who processed what and when
6. **Regular updates** - Keep dependencies updated for security patches

---

## Disclaimer & Important Warnings

### NO WARRANTY

This software is provided **"AS IS"** without any warranties, express or implied, including but not limited to:

- ❌ No warranty of merchantability
- ❌ No warranty of fitness for a particular purpose
- ❌ No warranty of non-infringement
- ❌ No guarantee of accuracy or reliability
- ❌ No guarantee of uptime or availability

The authors and contributors are **NOT LIABLE** for:

- Any damages (direct, indirect, incidental, special, or consequential)
- Data loss or corruption
- Business interruption
- Loss of profits
- Any claims arising from the use of this software

### AI Assessment Limitations

**Critical Understanding:**

⚠️ **AI assessments are NOT perfect** and should **NEVER** be used as the sole decision-making criteria for hiring, evaluation, or any consequential decision.

**Best Practices:**

1. ✅ **Use as supplementary tool only** - Combine with human judgment
2. ✅ **Validate results** - Review AI assessments manually
3. ✅ **Be aware of biases** - AI models can perpetuate biases in training data
4. ✅ **Comply with employment laws** - Some jurisdictions regulate AI in hiring
5. ✅ **Provide transparency** - Inform candidates about AI usage
6. ✅ **Allow appeals** - Give candidates opportunity to dispute AI assessments

**Known Limitations:**

- 🔴 Transcription accuracy ~95% (varies by accent, audio quality)
- 🔴 Cheating detection is rule-based (not ML-trained, ~80% accuracy)
- 🔴 LLM assessments can be subjective or biased
- 🔴 Non-verbal analysis may not account for cultural differences
- 🔴 System performance depends on hardware and video quality

### Legal Considerations

**Employment Law:**

- Some jurisdictions restrict or regulate AI use in hiring (e.g., EU AI Act, NYC Local Law 144)
- Consult legal counsel before using in employment decisions
- Ensure compliance with anti-discrimination laws
- Document decision-making process

**Data Protection:**

- Biometric data (facial landmarks) may be regulated (GDPR Article 9, BIPA)
- Implement data minimization principles
- Provide data subject access rights
- Maintain records of processing activities

**Intellectual Property:**

- Ensure you have rights to process interview videos
- Respect privacy and publicity rights of individuals
- Don't use for unauthorized surveillance

---

## Questions & Contact

**Licensing Questions:**

- Open GitHub Issue with "licensing" label
- Check FAQ: [troubleshooting/faq.md](../troubleshooting/faq.md)

**Legal Concerns:**

- This is an open-source project with no legal department
- Consult your own legal counsel for compliance questions
- MIT license is permissive - most uses are allowed

**Contributions:**

- See [Contributing Guide](../development/contributing.md)
- Code contributions licensed under MIT automatically
- By contributing, you agree to license under MIT

**Commercial Support:**

- ❌ No official commercial support available
- ✅ Community support via GitHub Issues
- ✅ Self-service via documentation

---

## Summary

**License:** MIT License (very permissive)

**You CAN:**

- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

**You MUST:**

- ✅ Include original license and copyright notice

**You CANNOT:**

- ❌ Hold authors liable
- ❌ Use authors' names for endorsement (without permission)

**Additional Considerations:**

- ⚠️ External APIs (HuggingFace, DeepL) have their own terms
- ⚠️ Llama 3.1 has usage restrictions (<700M MAU for commercial)
- ⚠️ Comply with local laws (employment, privacy, biometrics)

---

**Last Updated:** December 13, 2025

**License Version:** MIT License (unchanged since v1.0)
