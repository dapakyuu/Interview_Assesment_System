// =========================================================
// GLOBAL VARIABLES
// =========================================================
let interviewData = null;

// const API_BASE_URL = "http://127.0.0.1:8888";
const API_BASE_URL =
  "https://c9032efccb59.ngrok-free.app";

// =========================================================
// DATA LOADING
// =========================================================
async function loadJSONData() {
  try {
    // Get session ID from URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get("session");

    if (!sessionId) {
      throw new Error(
        "Session ID tidak ditemukan di URL. Pastikan URL memiliki parameter ?session=xxx"
      );
    }

    console.log(`🔍 Loading data for session: ${sessionId}`);

    // Show loading indicator
    showLoadingIndicator();

    // Fetch data from API
    const response = await fetch(`${API_BASE_URL}/results/${sessionId}.json`, {
      headers: {
        "ngrok-skip-browser-warning": "true",
      },
    });

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(
          "Data hasil interview tidak ditemukan. Mungkin masih dalam proses atau session tidak valid."
        );
      }
      throw new Error(
        `Failed to load data: ${response.status} ${response.statusText}`
      );
    }

    interviewData = await response.json();

    // Validate data structure
    if (!interviewData.success) {
      throw new Error("Data tidak valid atau processing gagal");
    }

    console.log("✅ Data berhasil dimuat:", interviewData);

    // Hide loading and load dashboard
    hideLoadingIndicator();
    loadDashboardData();

    // Update header with candidate name
    updateCandidateName(interviewData.name);
  } catch (error) {
    console.error("❌ Error loading data:", error);
    hideLoadingIndicator();
    showErrorMessage(error.message);
  }
}

function showLoadingIndicator() {
  const container = document.querySelector(".container");
  if (!container) return;

  const loadingDiv = document.createElement("div");
  loadingDiv.id = "dashboardLoading";
  loadingDiv.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.95);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    `;
  loadingDiv.innerHTML = `
        <div style="text-align: center;">
            <img src="Assest/Loader-cat.gif" alt="Loading..." style="width: 200px; height: 200px; margin-bottom: 20px; object-fit: contain;" />
            <h2 style="color: #2d3748; margin: 0;">Loading Dashboard...</h2>
            <p style="color: #718096; margin-top: 10px;">Mengambil data hasil interview</p>
        </div>
    `;
  document.body.appendChild(loadingDiv);
}

function hideLoadingIndicator() {
  const loading = document.getElementById("dashboardLoading");
  if (loading) loading.remove();
}

function showErrorMessage(message) {
  const container = document.querySelector(".container");
  if (!container) return;

  const errorDiv = document.createElement("div");
  errorDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        text-align: center;
        max-width: 500px;
        z-index: 10000;
    `;
  errorDiv.innerHTML = `
        <i class="fas fa-exclamation-circle" style="font-size: 64px; color: #e74c3c; margin-bottom: 20px;"></i>
        <h2 style="color: #2d3748; margin: 0 0 15px 0;">Gagal Memuat Data</h2>
        <p style="color: #718096; margin: 0 0 25px 0;">${message}</p>
        <button onclick="window.location.href='Upload.html'" class="btn btn-primary">
            <i class="fas fa-arrow-left"></i> Kembali ke Upload
        </button>
        <button onclick="location.reload()" class="btn btn-outline" style="margin-left: 10px;">
            <i class="fas fa-sync"></i> Coba Lagi
        </button>
    `;
  document.body.appendChild(errorDiv);
}

function updateCandidateName(name) {
  const headerLeft = document.querySelector(".header-left");
  if (headerLeft && name) {
    const nameTag = document.createElement("p");
    nameTag.style.cssText =
      "color: #667eea; font-weight: 600; margin-top: 5px;";
    nameTag.innerHTML = `<i class="fas fa-user"></i> ${name}`;
    headerLeft.appendChild(nameTag);
  }
}

// =========================================================
// CALCULATE AGGREGATE DATA
// =========================================================
function calculateAggregateData() {
  if (!interviewData?.content || interviewData.content.length === 0) {
    console.warn("⚠️ Data content kosong");
    return null;
  }

  const totalQuestions = interviewData.content.length;
  const aggregate = {
    avgConfidence: 0,
    avgKualitas: 0,
    avgRelevansi: 0,
    avgKoherensi: 0,
    avgTotal: 0,
  };

  // Hitung total dari semua pertanyaan
  interviewData.content.forEach((item) => {
    const p = item.result.penilaian;
    aggregate.avgConfidence += p.confidence_score;
    aggregate.avgKualitas += p.kualitas_jawaban;
    aggregate.avgRelevansi += p.relevansi;
    aggregate.avgKoherensi += p.koherensi;
    aggregate.avgTotal += p.total;
  });

  // Hitung rata-rata
  aggregate.avgConfidence = Math.round(
    aggregate.avgConfidence / totalQuestions
  );
  aggregate.avgKualitas = Math.round(aggregate.avgKualitas / totalQuestions);
  aggregate.avgRelevansi = Math.round(aggregate.avgRelevansi / totalQuestions);
  aggregate.avgKoherensi = Math.round(aggregate.avgKoherensi / totalQuestions);
  aggregate.avgTotal = Math.round(aggregate.avgTotal / totalQuestions);

  return aggregate;
}

// =========================================================
// MAIN DASHBOARD LOADER
// =========================================================
function loadDashboardData() {
  if (!interviewData) {
    console.error("❌ Data interview belum dimuat!");
    return;
  }

  updateAspectDetails();
  updateSummaryCards();
  updateCheatingDisplay();
  updateNonVerbalDisplay();
  updateFinalDecision();
  updateFinalRating();
  updateTranscriptDisplay();
  createRadarChart();

  // ✅ TAMBAHKAN FUNGSI BARU
  updateCheatingConfidenceCard();
  updateNonVerbalConfidenceCard();
  updateTranscriptionConfidenceCard();
  console.log("✅ Dashboard berhasil dimuat");
}

// =========================================================
// UPDATE FUNCTIONS
// =========================================================
function updateCheatingDisplay() {
  const cheatingElement = document.getElementById("cheating-detect");
  if (!cheatingElement || !interviewData?.aggregate_cheating_analysis) return;

  // ✅ Ambil dari aggregate analysis (bukan per-video)
  const agg = interviewData.aggregate_cheating_analysis;

  // Tentukan warna berdasarkan risk level
  let statusColor = "#28a745"; // Green (LOW RISK)
  let bgColor = "#d4edda";

  if (agg.risk_level === "HIGH RISK") {
    statusColor = "#dc3545"; // Red
    bgColor = "#f8d7da";
  } else if (agg.risk_level === "MEDIUM RISK") {
    statusColor = "#ffc107"; // Yellow
    bgColor = "#fff3cd";
  }

  // Build HTML berdasarkan status
  if (agg.overall_cheating_status.toLowerCase() === "ya") {
    cheatingElement.innerHTML = `
      <div class="content-text" style="background: ${bgColor}; padding: 15px; border-radius: 8px;">
        <div style="font-size: 18px; font-weight: bold; color: ${statusColor}; margin-bottom: 8px;">
          YA
        </div>
        <div style="font-size: 13px; color: #666; margin-bottom: 5px;">
          <strong>Risk Level:</strong> ${agg.risk_level}
        </div>
        <div style="font-size: 13px; color: #666; margin-bottom: 5px;">
          <strong>Confidence:</strong> ${agg.confidence_level}
        </div>
        <div style="font-size: 12px; color: ${statusColor}; margin-top: 8px; padding: 8px; background: white; border-radius: 4px;">
          <strong>📊 Summary:</strong><br/>
          ${agg.summary}
        </div>
        <div style="font-size: 12px; color: #555; margin-top: 8px;">
          <strong>⚠️ Recommendation:</strong> ${agg.recommendation}
        </div>
        <div style="font-size: 11px; color: #888; margin-top: 8px; border-top: 1px solid #ddd; padding-top: 8px;">
          Videos Flagged: ${agg.videos_flagged}/${agg.total_videos} (${agg.flagged_percentage}%) | 
          Avg Score: ${agg.overall_cheating_score}/100
        </div>
      </div>
    `;
  } else {
    cheatingElement.innerHTML = `
      <div class="content-text" style="background: ${bgColor}; padding: 15px; border-radius: 8px;">
        <div style="font-size: 18px; font-weight: bold; color: ${statusColor}; margin-bottom: 8px;">
          TIDAK
        </div>
        <div style="font-size: 13px; color: #666; margin-bottom: 5px;">
          <strong>Risk Level:</strong> ${agg.risk_level}
        </div>
        <div style="font-size: 12px; color: #555; margin-top: 8px;">
          ${agg.summary}
        </div>
        <div style="font-size: 11px; color: #888; margin-top: 8px; border-top: 1px solid #ddd; padding-top: 8px;">
          Videos Analyzed: ${agg.total_videos} | 
          Avg Score: ${agg.overall_cheating_score}/100
        </div>
      </div>
    `;
  }
}

// ============================================================================
// ✅ NEW FUNCTION: Update Cheating Confidence Score Card
// ============================================================================
function updateCheatingConfidenceCard() {
  const confidenceCard = document.getElementById("cheating-confidence-card");
  if (!confidenceCard || !interviewData?.aggregate_cheating_analysis) return;

  const agg = interviewData.aggregate_cheating_analysis;
  const confScore = agg.average_confidence_score || 0;

  // Determine color based on confidence level
  let confColor = "#28a745"; // Green
  let bgColor = "#d4edda";
  let statusText = "High Confidence";

  if (confScore < 45) {
    confColor = "#dc3545"; // Red
    bgColor = "#f8d7da";
    statusText = "Very Low Confidence";
  } else if (confScore < 60) {
    confColor = "#ffc107"; // Yellow
    bgColor = "#fff3cd";
    statusText = "Low Confidence";
  } else if (confScore < 75) {
    confColor = "#17a2b8"; // Blue
    bgColor = "#d1ecf1";
    statusText = "Medium Confidence";
  } else if (confScore < 85) {
    confColor = "#28a745"; // Green
    bgColor = "#d4edda";
    statusText = "High Confidence";
  } else {
    confColor = "#155724"; // Dark Green
    bgColor = "#c3e6cb";
    statusText = "Very High Confidence";
  }

  confidenceCard.innerHTML = `
    <div style="padding: 20px;">
      <!-- Header -->
      <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
          Detection Confidence Score
        </div>
        <div style="font-size: 48px; font-weight: bold; color: ${confColor};">
          ${confScore}%
        </div>
        <div style="font-size: 13px; color: ${confColor}; font-weight: 600; margin-top: 4px;">
          ${agg.overall_confidence_level || statusText}
        </div>
      </div>

      <!-- Progress Bar -->
      <div style="margin: 20px 0;">
        <div style="height: 12px; background: #e0e0e0; border-radius: 6px; overflow: hidden;">
          <div style="height: 100%; background: ${confColor}; width: ${confScore}%; transition: width 0.5s ease;"></div>
        </div>
      </div>

      <!-- Confidence Breakdown -->
      <div style="background: ${bgColor}; padding: 15px; border-radius: 8px; margin-top: 15px;">
        <div style="font-size: 13px; font-weight: 600; color: #333; margin-bottom: 10px;">
          📊 Confidence Breakdown:
        </div>
        
        ${getConfidenceBreakdown(agg)}
      </div>

      <!-- Explanation -->
      <div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-left: 3px solid ${confColor}; border-radius: 4px;">
        <div style="font-size: 12px; color: #666; line-height: 1.6;">
          <strong>ℹ️ What this means:</strong><br/>
          This score represents how confident the AI system is in detecting potential cheating behaviors. 
          ${getConfidenceExplanation(confScore)}
        </div>
      </div>
    </div>
  `;
}

// Helper function: Get confidence breakdown from individual videos
function getConfidenceBreakdown(agg) {
  // If we have per-video confidence data
  if (interviewData?.content && interviewData.content.length > 0) {
    let html = '<div style="display: flex; flex-direction: column; gap: 8px;">';

    interviewData.content.forEach((item, index) => {
      const videoConf = item.result?.cheating_confidence_score || 0;
      const videoConfLevel = item.result?.cheating_confidence_level || "N/A";

      let barColor = "#28a745";
      if (videoConf < 60) barColor = "#ffc107";
      if (videoConf < 45) barColor = "#dc3545";

      html += `
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="flex: 0 0 80px; font-size: 11px; color: #666;">
            Video ${index + 1}:
          </div>
          <div style="flex: 1; height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden;">
            <div style="height: 100%; background: ${barColor}; width: ${videoConf}%;"></div>
          </div>
          <div style="flex: 0 0 60px; font-size: 11px; font-weight: 600; color: ${barColor}; text-align: right;">
            ${videoConf}%
          </div>
        </div>
      `;
    });

    html += "</div>";
    return html;
  }

  // Fallback: show only average
  return `
    <div style="text-align: center; font-size: 12px; color: #666;">
      Average confidence across ${agg.total_videos} video(s): <strong>${agg.average_confidence_score}%</strong>
    </div>
  `;
}

// Helper function: Explain what the confidence score means
function getConfidenceExplanation(score) {
  if (score >= 85) {
    return "The system has very high certainty in its cheating detection analysis. The detection methods produced consistent and reliable results.";
  } else if (score >= 75) {
    return "The system has high certainty in its analysis. Most detection methods produced reliable results.";
  } else if (score >= 60) {
    return "The system has moderate certainty. Some detection methods may have produced inconsistent results.";
  } else if (score >= 45) {
    return "The system has low certainty. Detection results may be unreliable due to video quality or technical limitations.";
  } else {
    return "The system has very low certainty. Results should be interpreted with caution and may require manual review.";
  }
}
// Helper function: Get confidence breakdown from individual videos
function getConfidenceBreakdown(agg) {
  // If we have per-video confidence data
  if (interviewData?.content && interviewData.content.length > 0) {
    let html = '<div style="display: flex; flex-direction: column; gap: 8px;">';

    interviewData.content.forEach((item, index) => {
      const videoConf = item.result?.cheating_confidence_score || 0;
      const videoConfLevel = item.result?.cheating_confidence_level || "N/A";

      let barColor = "#28a745";
      if (videoConf < 60) barColor = "#ffc107";
      if (videoConf < 45) barColor = "#dc3545";

      html += `
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="flex: 0 0 80px; font-size: 11px; color: #666;">
            Video ${index + 1}:
          </div>
          <div style="flex: 1; height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden;">
            <div style="height: 100%; background: ${barColor}; width: ${videoConf}%;"></div>
          </div>
          <div style="flex: 0 0 60px; font-size: 11px; font-weight: 600; color: ${barColor}; text-align: right;">
            ${videoConf}%
          </div>
        </div>
      `;
    });

    html += "</div>";
    return html;
  }

  // Fallback: show only average
  return `
    <div style="text-align: center; font-size: 12px; color: #666;">
      Average confidence across ${agg.total_videos} video(s): <strong>${agg.average_confidence_score}%</strong>
    </div>
  `;
}

// Helper function: Explain what the confidence score means
function getConfidenceExplanation(score) {
  if (score >= 85) {
    return "The system has very high certainty in its cheating detection analysis. The detection methods produced consistent and reliable results.";
  } else if (score >= 75) {
    return "The system has high certainty in its analysis. Most detection methods produced reliable results.";
  } else if (score >= 60) {
    return "The system has moderate certainty. Some detection methods may have produced inconsistent results.";
  } else if (score >= 45) {
    return "The system has low certainty. Detection results may be unreliable due to video quality or technical limitations.";
  } else {
    return "The system has very low certainty. Results should be interpreted with caution and may require manual review.";
  }
}

// ...existing code... (line 1-409)

// ============================================================================
// ✅ NEW: Non-Verbal Confidence Card
// ============================================================================
function updateNonVerbalConfidenceCard() {
  const confidenceCard = document.getElementById("nonverbal-confidence-card");
  if (!confidenceCard || !interviewData?.aggregate_non_verbal_analysis) return;

  const agg = interviewData.aggregate_non_verbal_analysis;
  const confScore = agg.overall_confidence_score || 0;
  const confLevel = agg.overall_performance_status || "N/A";

  let confColor = "#28a745";
  let bgColor = "#d4edda";

  if (confScore < 45) {
    confColor = "#dc3545";
    bgColor = "#f8d7da";
  } else if (confScore < 60) {
    confColor = "#ffc107";
    bgColor = "#fff3cd";
  } else if (confScore < 75) {
    confColor = "#17a2b8";
    bgColor = "#d1ecf1";
  } else if (confScore < 85) {
    confColor = "#28a745";
    bgColor = "#d4edda";
  }

  confidenceCard.innerHTML = `
    <div style="padding: 20px;">
      <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
          Non-Verbal Analysis Confidence
        </div>
        <div style="font-size: 48px; font-weight: bold; color: ${confColor};">
          ${confScore}%
        </div>
        <div style="font-size: 13px; color: ${confColor}; font-weight: 600; margin-top: 4px;">
          ${confLevel}
        </div>
      </div>

      <div style="margin: 20px 0;">
        <div style="height: 12px; background: #e0e0e0; border-radius: 6px; overflow: hidden;">
          <div style="height: 100%; background: ${confColor}; width: ${confScore}%; transition: width 0.5s ease;"></div>
        </div>
      </div>

      <div style="background: ${bgColor}; padding: 15px; border-radius: 8px;">
        <div style="font-size: 13px; font-weight: 600; color: #333; margin-bottom: 10px;">
          📊 Component Breakdown:
        </div>
        ${getNonVerbalBreakdown()}
      </div>

      <div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-left: 3px solid ${confColor}; border-radius: 4px;">
        <div style="font-size: 12px; color: #666; line-height: 1.6;">
          <strong>ℹ️ Analysis Quality:</strong><br/>
          ${getNonVerbalExplanation(confScore)}
        </div>
      </div>
    </div>
  `;
}

function getNonVerbalBreakdown() {
  if (interviewData?.content && interviewData.content.length > 0) {
    let html = '<div style="display: flex; flex-direction: column; gap: 8px;">';

    interviewData.content.forEach((item, index) => {
      const nvConf = item.result?.non_verbal_confidence_score || 0;

      let barColor = "#28a745";
      if (nvConf < 60) barColor = "#ffc107";
      if (nvConf < 45) barColor = "#dc3545";

      html += `
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="flex: 0 0 80px; font-size: 11px; color: #666;">
            Video ${index + 1}:
          </div>
          <div style="flex: 1; height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden;">
            <div style="height: 100%; background: ${barColor}; width: ${nvConf}%;"></div>
          </div>
          <div style="flex: 0 0 60px; font-size: 11px; font-weight: 600; color: ${barColor}; text-align: right;">
            ${nvConf}%
          </div>
        </div>
      `;
    });

    html += "</div>";
    return html;
  }

  return '<div style="text-align: center; font-size: 12px; color: #666;">No detailed breakdown available</div>';
}

function getNonVerbalExplanation(score) {
  if (score >= 85)
    return "Excellent non-verbal analysis with high-quality facial, speech, and eye tracking data.";
  if (score >= 75)
    return "Good analysis quality with reliable non-verbal indicators detected.";
  if (score >= 60)
    return "Moderate quality - some non-verbal features may be limited.";
  if (score >= 45) return "Low quality - limited non-verbal data captured.";
  return "Very low quality - non-verbal analysis may be unreliable.";
}

// ============================================================================
// ✅ UPDATED: Transcription Confidence Card (was Translation Confidence Card)
// ============================================================================
function updateTranscriptionConfidenceCard() {
  const confidenceCard = document.getElementById(
    "transcription-confidence-card"
  );
  if (!confidenceCard || !interviewData?.content) return;

  // Calculate average transcription confidence
  let totalConf = 0;
  let count = 0;

  interviewData.content.forEach((item) => {
    if (item.result?.transkripsi_confidence) {
      totalConf += item.result.transkripsi_confidence;
      count++;
    }
  });

  const avgConf = count > 0 ? Math.round(totalConf / count) : 0;

  let confColor = "#28a745";
  let bgColor = "#d4edda";
  let confLevel = "Very High";

  if (avgConf < 45) {
    confColor = "#dc3545";
    bgColor = "#f8d7da";
    confLevel = "Very Low";
  } else if (avgConf < 60) {
    confColor = "#ffc107";
    bgColor = "#fff3cd";
    confLevel = "Low";
  } else if (avgConf < 75) {
    confColor = "#17a2b8";
    bgColor = "#d1ecf1";
    confLevel = "Medium";
  } else if (avgConf < 85) {
    confColor = "#28a745";
    bgColor = "#d4edda";
    confLevel = "High";
  }

  confidenceCard.innerHTML = `
    <div style="padding: 20px;">
      <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
          Transcription Confidence Score
        </div>
        <div style="font-size: 48px; font-weight: bold; color: ${confColor};">
          ${avgConf}%
        </div>
        <div style="font-size: 13px; color: ${confColor}; font-weight: 600; margin-top: 4px;">
          ${confLevel}
        </div>
      </div>

      <div style="margin: 20px 0;">
        <div style="height: 12px; background: #e0e0e0; border-radius: 6px; overflow: hidden;">
          <div style="height: 100%; background: ${confColor}; width: ${avgConf}%; transition: width 0.5s ease;"></div>
        </div>
      </div>

      <div style="background: ${bgColor}; padding: 15px; border-radius: 8px;">
        <div style="font-size: 13px; font-weight: 600; color: #333; margin-bottom: 10px;">
          📊 Per-Video Confidence:
        </div>
        ${getTranscriptionBreakdown()}
      </div>

      <div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-left: 3px solid ${confColor}; border-radius: 4px;">
        <div style="font-size: 12px; color: #666; line-height: 1.6;">
          <strong>ℹ️ Transcription Reliability:</strong><br/>
          ${getTranscriptionExplanation(avgConf)}
        </div>
      </div>
    </div>
  `;
}

function getTranscriptionBreakdown() {
  if (interviewData?.content && interviewData.content.length > 0) {
    let html = '<div style="display: flex; flex-direction: column; gap: 8px;">';

    interviewData.content.forEach((item, index) => {
      const transConf = item.result?.transkripsi_confidence || 0;

      let barColor = "#28a745";
      if (transConf < 60) barColor = "#ffc107";
      if (transConf < 45) barColor = "#dc3545";

      html += `
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="flex: 0 0 80px; font-size: 11px; color: #666;">
            Video ${index + 1}:
          </div>
          <div style="flex: 1; height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden;">
            <div style="height: 100%; background: ${barColor}; width: ${transConf}%;"></div>
          </div>
          <div style="flex: 0 0 60px; font-size: 11px; font-weight: 600; color: ${barColor}; text-align: right;">
            ${transConf}%
          </div>
        </div>
      `;
    });

    html += "</div>";
    return html;
  }

  return '<div style="text-align: center; font-size: 12px; color: #666;">No transcription data available</div>';
}

function getTranscriptionExplanation(score) {
  if (score >= 85)
    return "Excellent transcription quality with very high accuracy from the speech-to-text model.";
  if (score >= 75)
    return "Good transcription quality - the model is confident in the recognized speech.";
  if (score >= 60)
    return "Acceptable transcription - some words may have lower confidence due to audio quality.";
  if (score >= 45)
    return "Low quality transcription - audio quality or speech clarity may be poor. Manual review recommended.";
  return "Very low quality - transcription may be unreliable due to poor audio or unclear speech.";
}

function updateNonVerbalDisplay() {
  const nonVerbalElement = document.getElementById("nonverbal-analysis");
  if (!nonVerbalElement) return;

  // Ambil summary dari aggregate non-verbal batch
  const summary =
    interviewData.aggregate_non_verbal_analysis?.summary ||
    "Tidak ada ringkasan non-verbal.";

  let output = `
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
      <h3 style="color: #2c3e50;"> Ringkasan Analisis Non-Verbal </h3>
      <p>${summary}</p>
    </div>
  `;

  nonVerbalElement.innerHTML = output.trim();
}

function updateFinalDecision() {
  const decisionElement = document.getElementById("final-decision");
  if (!decisionElement || !interviewData?.content) return;

  const result = interviewData.content[0].result;
  decisionElement.innerHTML = `<div class="content-text">${result.keputusan_akhir}</div>`;
}

function updateFinalRating() {
  const scoreImageElement = document.getElementById("scoreImage");
  if (!scoreImageElement) return;

  const rating = interviewData.content[0].result.penilaian_akhir;
  scoreImageElement.src = `Assest/rating-dark-${rating}.png`;
  scoreImageElement.alt = `Rating ${rating} dari 5`;
}

function updateAspectDetails() {
  const detailsList = document.getElementById("aspectDetailsList");
  if (!detailsList) return;

  const aggregate = calculateAggregateData();
  const aspects = [
    // { label: "Confidence Score", score: aggregate.avgConfidence },
    { label: "Kualitas Jawaban", score: aggregate.avgKualitas },
    { label: "Relevansi", score: aggregate.avgRelevansi },
    { label: "Koherensi", score: aggregate.avgKoherensi },
    // { label: "Tempo Bicara", score: aggregate.avgTempo },
  ];

  detailsList.innerHTML = aspects
    .map(
      (aspect) => `
        <div class="aspect-detail-item">
            <div class="aspect-label">${aspect.label}</div>
            <div class="aspect-bar-container">
                <div class="aspect-progress-bar">
                    <div class="aspect-progress-fill" style="width: ${aspect.score}%"></div>
                </div>
                <div class="aspect-scale">
                    <span>0</span>
                    <span>50</span>
                    <span>100</span>
                </div>
            </div>
            <div class="aspect-score">${aspect.score}</div>
        </div>
    `
    )
    .join("");
}

function updateSummaryCards() {
  // ✅ Ambil data dari llm_results (hanya 2 field: confidence & kesimpulan)
  const llmResults = interviewData.llm_results;

  if (!llmResults) {
    console.warn("⚠️ llm_results tidak tersedia");
    // Fallback ke perhitungan manual jika llm_results tidak ada
    const aggregate = calculateAggregateData();
    document.getElementById("highestScore").textContent =
      aggregate.avgConfidence;
    document.getElementById("analisisllm").textContent =
      "Kesimpulan LLM tidak tersedia.";
    return;
  }

  // ✅ 1. Update Confidence Score LLM
  const avgConfidence = llmResults.rata_rata_confidence_score || 0;
  document.getElementById("highestScore").textContent = avgConfidence;

  // ✅ 2. Update Kesimpulan LLM
  const kesimpulanLLM =
    llmResults.kesimpulan_llm || "Kesimpulan LLM tidak tersedia.";
  document.getElementById("analisisllm").textContent = kesimpulanLLM;

  console.log("✅ Summary cards updated from llm_results:", {
    avgConfidence,
    kesimpulanLength: kesimpulanLLM.length,
  });
}

function updateTranscriptDisplay() {
  const transcriptContainer = document.querySelector("#transcript-container");
  if (!transcriptContainer) return;

  transcriptContainer.innerHTML = "";

  interviewData.content.forEach((item, index) => {
    // Buat section untuk setiap transkrip
    const transcriptSection = document.createElement("div");
    transcriptSection.className = "transcript-section";

    // Buat header dropdown untuk transkrip ini
    const transcriptHeader = document.createElement("div");
    transcriptHeader.className = "transcript-header";
    transcriptHeader.innerHTML = `
      <div class="transcript-header-title">
        <i class="fas fa-file-alt"></i>
        <span>Transkrip Video ${index + 1}</span>
      </div>
      <i class="fas fa-chevron-down transcript-toggle"></i>
    `;

    // Buat content container untuk transkrip ini
    const transcriptContent = document.createElement("div");
    transcriptContent.className = "transcript-content";

    // Buat wrapper untuk smooth transition
    const contentWrapper = document.createElement("div");
    contentWrapper.className = "transcript-content-wrapper";

    // Isi dengan card Indonesia dan English
    const row = document.createElement("div");
    row.className = "dashboard-grid";
    row.style.marginBottom = "20px";

    row.innerHTML = `
      <div class="card" style="grid-column: span 1;">
        <div class="card-title">
          <i class="fas fa-video"></i>
          Transkrip Video ${index + 1} - Indonesia
        </div>
        <div class="card-content">
          <div style="margin-bottom: 15px; padding: 15px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
              <div style="width: 36px; height: 36px; background: #0052d3; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 16px;">
                ${index + 1}
              </div>
              <div style="font-weight: 600; color: black; font-size: 15px;">
                Video ${index + 1}
              </div>
            </div>
            
            <div style="font-size: 14px; color: #2d3748; line-height: 1.8; padding: 12px; background: white; border-radius: 6px; text-align: justify;">
              ${
                item.result.transkripsi_id ||
                "Transkrip Indonesia tidak tersedia"
              }
            </div>
          </div>
        </div>
      </div>

      <div class="card" style="grid-column: span 1;">
        <div class="card-title">
          <i class="fas fa-video"></i>
          Transkrip Video ${index + 1} - English
        </div>
        <div class="card-content">
          <div style="margin-bottom: 15px; padding: 15px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
              <div style="width: 36px; height: 36px; background: #fff3d9; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: black; font-weight: bold; font-size: 16px;">
                ${index + 1}
              </div>
              <div style="font-weight: 600; color: black; font-size: 15px;">
                Video ${index + 1}
              </div>
            </div>
            
            <div style="font-size: 14px; color: #2d3748; line-height: 1.8; padding: 12px; background: white; border-radius: 6px; text-align: justify;">
              ${
                item.result.transkripsi_en || "English transcript not available"
              }
            </div>
          </div>
        </div>
      </div>
    `;

    contentWrapper.appendChild(row);
    transcriptContent.appendChild(contentWrapper);

    // Toggle functionality untuk transkrip ini
    transcriptHeader.addEventListener("click", () => {
      const toggle = transcriptHeader.querySelector(".transcript-toggle");
      transcriptContent.classList.toggle("active");
      toggle.classList.toggle("active");
    });

    // Append header dan content ke section
    transcriptSection.appendChild(transcriptHeader);
    transcriptSection.appendChild(transcriptContent);

    // Append section ke container utama
    transcriptContainer.appendChild(transcriptSection);
  });
}

// =========================================================
// RADAR CHART
// =========================================================
function createRadarChart() {
  const ctx = document.getElementById("radarChart");
  if (!ctx) return;

  const aggregate = calculateAggregateData();
  const chartData = [
    // aggregate.avgConfidence,
    aggregate.avgKualitas,
    aggregate.avgRelevansi,
    aggregate.avgKoherensi,
    // aggregate.avgTempo,
  ];

  new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["Kualitas", "Relevansi", "Koherensi"],
      datasets: [
        {
          label: "Skor Kandidat",
          data: chartData,
          backgroundColor: "rgba(102, 126, 234, 0.2)",
          borderColor: "#667eea",
          borderWidth: 2,
          pointBackgroundColor: "#667eea",
          pointBorderColor: "#fff",
          pointHoverBackgroundColor: "#fff",
          pointHoverBorderColor: "#667eea",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          ticks: { stepSize: 20 },
        },
      },
      plugins: {
        legend: { position: "top" },
      },
    },
  });
}

// =========================================================
// UTILITY FUNCTIONS
// =========================================================
function downloadJSON() {
  if (!interviewData) {
    alert("Data belum dimuat!");
    return;
  }

  // Konversi data ke JSON string dengan format yang rapi
  const jsonString = JSON.stringify(interviewData, null, 2);

  // Buat blob dari JSON string
  const blob = new Blob([jsonString], { type: "application/json" });

  // Buat URL untuk blob
  const url = URL.createObjectURL(blob);

  // Buat element anchor untuk download
  const a = document.createElement("a");
  a.href = url;
  a.download = `interview_data_${new Date().getTime()}.json`;

  // Trigger download
  document.body.appendChild(a);
  a.click();

  // Cleanup
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  console.log("✅ JSON berhasil didownload");
}

async function downloadPDF() {
  if (!interviewData) {
    alert("Data belum dimuat!");
    return;
  }

  // Import jsPDF
  const script = document.createElement("script");
  script.src =
    "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
  document.head.appendChild(script);

  script.onload = function () {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    // 🎨 Tema warna
    const blue = [0, 82, 211]; // #0052d3
    const cream = [255, 243, 217]; // #fff3d9

    // 📐 Layout
    const tableX = 20;
    const tableWidth = 170;
    const result = interviewData.content[0].result;
    const aggregate = calculateAggregateData();
    let yPos = 10;

    // ===== HEADER =====
    doc.setFontSize(20);
    doc.setTextColor(...blue);
    doc.text("LAPORAN ANALISIS INTERVIEW", 105, yPos, { align: "center" });
    yPos += 15;
    doc.setDrawColor(...blue);
    doc.line(20, yPos, 190, yPos);
    yPos += 10;

    // ===== RINGKASAN PENILAIAN =====
    doc.setFontSize(14);
    doc.setTextColor(...blue);
    doc.text("Ringkasan Penilaian", 20, yPos);
    yPos += 5;

    doc.setFontSize(10);
    doc.setTextColor(0, 0, 0);
    doc.text(`Rata-rata Skor: ${aggregate.avgTotal}`, 20, yPos);
    doc.text(`Penilaian Akhir: ${result.penilaian_akhir}/5`, 80, yPos);
    doc.text(`Keputusan: ${result.keputusan_akhir}`, 140, yPos);
    yPos += 10;

    // ===== SKOR PER ASPEK =====
    yPos = drawTableHeader(
      doc,
      "Skor Per Aspek Kompetensi",
      yPos,
      tableX,
      blue
    );
    const aspects = [
      // [
      //   "Confidence Score",
      //   aggregate.avgConfidence,
      //   getScoreCategoryText(aggregate.avgConfidence),
      // ],
      [
        "Kualitas Jawaban",
        aggregate.avgKualitas,
        getScoreCategoryText(aggregate.avgKualitas),
      ],
      [
        "Relevansi",
        aggregate.avgRelevansi,
        getScoreCategoryText(aggregate.avgRelevansi),
      ],
      [
        "Koherensi",
        aggregate.avgKoherensi,
        getScoreCategoryText(aggregate.avgKoherensi),
      ],
      // [
      //   "Tempo Bicara",
      //   aggregate.avgTempo,
      //   getScoreCategoryText(aggregate.avgTempo),
      // ],
    ];
    yPos = drawTable(
      doc,
      aspects,
      ["Aspek Penilaian", "Skor", "Kategori"],
      yPos,
      tableX,
      tableWidth,
      blue,
      cream
    );

    // ===== HASIL ANALISIS =====
    yPos = drawTableHeader(doc, "Hasil Analisis", yPos + 10, tableX, blue);
    const analysisData = [
      ["Cheating Detection", result.cheating_detection || "Tidak"],
    ];
    if (result.alasan_cheating) {
      analysisData.push(["Alasan Cheating", result.alasan_cheating]);
    }
    analysisData.push(["Analisis Non-Verbal", result.analisis_non_verbal]);
    yPos = drawTable(
      doc,
      analysisData,
      ["Kategori", "Hasil"],
      yPos,
      tableX,
      tableWidth,
      blue,
      cream
    );

    // ===== DETAIL PERTANYAAN =====
    yPos += 10;
    if (yPos > 240) {
      doc.addPage();
      yPos = 25;
    }
    doc.setFontSize(12);
    doc.setTextColor(...blue);
    doc.text(
      `Detail Pertanyaan (${interviewData.content.length} Pertanyaan)`,
      20,
      yPos
    );
    yPos += 7.5;

    interviewData.content.forEach((item, idx) => {
      if (yPos > 240) {
        doc.addPage();
        yPos = 25;
      }

      doc.setFontSize(10);
      doc.setTextColor(45, 55, 72);
      doc.text(`Pertanyaan ${idx + 1}:`, 20, yPos);
      yPos += 5;

      doc.setFontSize(9);
      const qLines = doc.splitTextToSize(item.question, 170);
      doc.text(qLines, 20, yPos);
      yPos += qLines.length * 5 + 5;

      const scores = [
        // ["Confidence Score", item.result.penilaian.confidence_score],
        ["Kualitas Jawaban", item.result.penilaian.kualitas_jawaban],
        ["Relevansi", item.result.penilaian.relevansi],
        ["Koherensi", item.result.penilaian.koherensi],
        // ["Tempo Bicara", item.result.penilaian.tempo_bicara],
        ["Total Skor", item.result.penilaian.total],
      ];
      yPos = drawTable(
        doc,
        scores,
        ["Aspek", "Nilai"],
        yPos,
        tableX,
        tableWidth,
        blue,
        cream
      );
      yPos += 8;
    });

    // ===== FOOTER =====
    const pageCount = doc.internal.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFontSize(8);
      doc.setTextColor(80, 80, 80);
      doc.text(
        "Laporan ini dibuat secara otomatis oleh AI Interview Platform",
        105,
        285,
        { align: "center" }
      );
      doc.text(`Tanggal: ${new Date().toLocaleString("id-ID")}`, 105, 290, {
        align: "center",
      });
      doc.text(`Halaman ${i} dari ${pageCount}`, 190, 290, { align: "right" });
    }

    // ===== DOWNLOAD =====
    doc.save(`interview_report_${new Date().getTime()}.pdf`);
  };
}

// =========================================
// 🔹 Helper: Draw Section Header
// =========================================
function drawTableHeader(doc, title, yPos, xStart, blue) {
  doc.setFontSize(12);
  doc.setTextColor(...blue);
  doc.text(title, xStart, yPos);
  return yPos + 6;
}

// =========================================
// 🔹 Helper: Draw Table (Auto Page + Full Move)
// =========================================
function drawTable(doc, rows, headers, yPos, xStart, tableWidth, blue, cream) {
  const rowHeight = 7;
  const headerHeight = 8;
  const pageHeight = doc.internal.pageSize.getHeight();
  const bottomMargin = 20;
  const colCount = headers.length;
  const colWidth = tableWidth / colCount;

  // 🔹 Total tinggi tabel
  const tableHeight = headerHeight + rows.length * rowHeight;
  if (yPos + tableHeight > pageHeight - bottomMargin) {
    doc.addPage();
    yPos = 25;
  }

  // ===== HEADER =====
  doc.setFillColor(...blue);
  doc.setTextColor(255, 255, 255);
  doc.rect(xStart, yPos, tableWidth, headerHeight, "F");
  doc.setFontSize(9);
  let xCursor = xStart + 5;
  headers.forEach((h, i) => {
    doc.text(h, xCursor, yPos + 5);
    xCursor += colWidth;
  });
  yPos += headerHeight;
  doc.setTextColor(0, 0, 0);

  // ===== ROWS =====
  rows.forEach((row, idx) => {
    // 🔹 Jika tabel baris berikutnya melewati batas halaman
    if (yPos + rowHeight > pageHeight - bottomMargin) {
      doc.addPage();
      yPos = 25;

      // Header ulang di halaman baru
      doc.setFillColor(...blue);
      doc.setTextColor(255, 255, 255);
      doc.rect(xStart, yPos, tableWidth, headerHeight, "F");
      xCursor = xStart + 5;
      headers.forEach((h, i) => {
        doc.text(h, xCursor, yPos + 5);
        xCursor += colWidth;
      });
      yPos += headerHeight;
      doc.setTextColor(0, 0, 0);
    }

    // 🔹 Latar baris selang-seling
    if (idx % 2 === 0) {
      doc.setFillColor(...cream);
      doc.rect(xStart, yPos, tableWidth, rowHeight, "F");
    }

    xCursor = xStart + 5;
    row.forEach((cell, i) => {
      const textLines = doc.splitTextToSize(String(cell), colWidth - 10);
      doc.text(textLines, xCursor, yPos + 5);
      xCursor += colWidth;
    });

    yPos += rowHeight;
  });

  return yPos;
}

// =========================================
// 🔹 Helper: Get Score Category
// =========================================
function getScoreCategoryText(score) {
  if (score >= 90) return "Luar Biasa";
  if (score >= 80) return "Bagus";
  if (score >= 70) return "Cukup Bagus";
  return "Perlu Peningkatan";
}

// =========================================================
// INITIALIZATION
// =========================================================
document.addEventListener("DOMContentLoaded", () => {
  console.log("🚀 Inisialisasi Dashboard...");

  // Check if session parameter exists
  const urlParams = new URLSearchParams(window.location.search);
  const sessionId = urlParams.get("session");

  if (!sessionId) {
    showErrorMessage(
      "Session ID tidak ditemukan di URL. Akses dashboard melalui halaman upload setelah processing selesai."
    );
    return;
  }

  console.log(`📋 Session ID: ${sessionId}`);
  loadJSONData();
});
