/* ============================
   VARIABLES
   ============================ */
// Video
let fileInputVideo, previewGridVideo, statusAreaVideo, participantNameInput;
let selectedFilesVideo = [];
let videoQuestions = []; // NEW: Array untuk menyimpan pertanyaan setiap video
let objectUrlsVideo = [];
let clearAllVideoBtn;
let isSubmittingVideo = false;
let lastSubmitTime = 0;

// JSON
let fileInputJSON, previewGridJSON, statusAreaJSON;
let selectedFilesJSON = [];
let clearAllJSONBtn;
let isSubmittingJSON = false;
let lastSubmitTimeJSON = 0;

// Config
const DEFAULT_BASE_URL = "http://127.0.0.1:5500";
const SUBMIT_DEBOUNCE_MS = 3000;
const SESSION_STORAGE_KEY = "video_processing_session";

// const VIDEO_ENDPOINT = "http://127.0.0.1:8888/upload";
// const API_BASE_URL = "http://127.0.0.1:8888";

// jika menggunakan ngrok, ganti dengan URL ngrok Anda
const VIDEO_ENDPOINT =
  "https://c9032efccb59.ngrok-free.app/upload";
const API_BASE_URL =
  "https://c9032efccb59.ngrok-free.app";

/* ============================
   HELPERS
   ============================ */
function getBaseUrl() {
  return DEFAULT_BASE_URL.replace(/\/+$/, "");
}

function showLoading(message = "Processing, please wait...") {
  const o = document.getElementById("loadingOverlay");
  const m = document.getElementById("loadingMessage");
  if (m) m.textContent = message;
  if (o) o.style.display = "flex";

  // Ensure cat gif is loaded (in case it's dynamically changed)
  const img = o?.querySelector("img");
  if (img && !img.src.includes("Loader-cat.gif")) {
    img.src = "Assest/Loader-cat.gif";
  }
}

function hideLoading() {
  const o = document.getElementById("loadingOverlay");
  if (o) o.style.display = "none";
}

let __statusReadyInjected = false;
function injectStatusReadyStyle() {
  if (__statusReadyInjected) return;
  __statusReadyInjected = true;
  const s = document.createElement("style");
  s.textContent = `.status-ready{background:linear-gradient(180deg,#28a745,#1e7e34);color:#fff!important;padding:4px 8px;border-radius:6px;font-weight:700;display:inline-block;box-shadow:0 2px 6px rgba(30,126,52,.25)}`;
  document.head.appendChild(s);
}

function setStatusReady(el, message) {
  if (!el) return;
  injectStatusReadyStyle();
  if (typeof message === "string") el.textContent = message;
  el.classList.add("status-ready");
}

function resetStatus(el, message) {
  if (!el) return;
  el.classList.remove("status-ready");
  if (typeof message === "string") el.textContent = message;
}

function updateClearButtonsVisibility() {
  try {
    // Update Clear buttons
    if (clearAllVideoBtn)
      clearAllVideoBtn.style.display = selectedFilesVideo.length ? "" : "none";
    if (clearAllJSONBtn)
      clearAllJSONBtn.style.display = selectedFilesJSON.length ? "" : "none";

    // Update Send buttons mode
    updateSendVideoButtonMode();
    updateSendJSONButtonMode();
  } catch (e) {}
}

function updateSendVideoButtonMode() {
  const sendVideoBtn = document.querySelector(
    "#videoCard .btn-primary, #videoCard .btn-outline"
  );
  if (!sendVideoBtn) return;

  if (selectedFilesVideo.length === 0) {
    // MODE: Select Files
    sendVideoBtn.innerHTML = '<i class="fas fa-folder-open"></i> Pilih Video';
    sendVideoBtn.onclick = function (e) {
      e.preventDefault();
      fileInputVideo?.click();
    };
    sendVideoBtn.classList.remove("btn-primary");
    sendVideoBtn.classList.add("btn-outline");
  } else {
    // MODE: Send Videos
    sendVideoBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Kirim Video';
    sendVideoBtn.onclick = function (e) {
      e.preventDefault();
      buildAndSendVideo();
    };
    sendVideoBtn.classList.remove("btn-outline");
    sendVideoBtn.classList.add("btn-primary");
  }
}

function updateSendJSONButtonMode() {
  const sendJSONBtn = document.querySelector(
    "#jsonCard .btn-primary, #jsonCard .btn-outline"
  );
  if (!sendJSONBtn) return;

  if (selectedFilesJSON.length === 0) {
    // MODE: Select File
    sendJSONBtn.innerHTML = '<i class="fas fa-folder-open"></i> Pilih JSON';
    sendJSONBtn.onclick = function (e) {
      e.preventDefault();
      fileInputJSON?.click();
    };
    sendJSONBtn.classList.remove("btn-primary");
    sendJSONBtn.classList.add("btn-outline");
  } else {
    // MODE: Send JSON
    sendJSONBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Kirim JSON';
    sendJSONBtn.onclick = function (e) {
      e.preventDefault();
      buildAndSendJSONFiles();
    };
    sendJSONBtn.classList.remove("btn-outline");
    sendJSONBtn.classList.add("btn-primary");
  }
}

/* ============================
   VIDEO HANDLERS
   ============================ */
function handleFilesVideo(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("video/"));
  if (!files.length) return alert("Tidak ada file video terdeteksi.");

  selectedFilesVideo = selectedFilesVideo.concat(files);
  // NEW: Inisialisasi pertanyaan kosong untuk video baru
  files.forEach(() => videoQuestions.push(""));
  renderPreviewsVideo();
  updateClearButtonsVisibility();
}

function renderPreviewsVideo() {
  objectUrlsVideo.forEach((u) => {
    try {
      URL.revokeObjectURL(u);
    } catch (e) {}
  });
  objectUrlsVideo = [];

  if (!previewGridVideo) return;
  previewGridVideo.innerHTML = "";

  selectedFilesVideo.forEach((file, idx) => {
    const item = document.createElement("div");
    item.className = "preview-item";

    // NEW: Input pertanyaan - DIPINDAHKAN KE ATAS
    const questionWrapper = document.createElement("div");
    questionWrapper.style.cssText = "margin-bottom: 12px;";

    const questionLabel = document.createElement("label");
    questionLabel.style.cssText =
      "font-weight: 600; display: block; margin-bottom: 4px;";
    questionLabel.innerHTML = 'Pertanyaan<span class="required">*</span>:';

    const questionInput = document.createElement("input");
    questionInput.type = "text";
    questionInput.className = "small-input";
    questionInput.placeholder = "Masukkan pertanyaan untuk video ini";
    questionInput.value = videoQuestions[idx] || "";
    questionInput.style.cssText =
      "width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px;";
    questionInput.addEventListener("input", (e) => {
      videoQuestions[idx] = e.target.value;
    });

    questionWrapper.appendChild(questionLabel);
    questionWrapper.appendChild(questionInput);

    // Video preview
    const video = document.createElement("video");
    video.className = "preview-video";
    video.controls = true;
    video.preload = "metadata";
    const blobUrl = URL.createObjectURL(file);
    objectUrlsVideo[idx] = blobUrl;
    video.src = blobUrl;

    const wrapper = document.createElement("div");
    wrapper.style.cssText =
      "position:relative;width:100%;padding-top:56.25%;overflow:hidden";
    video.style.cssText =
      "position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain";
    wrapper.appendChild(video);

    const meta = document.createElement("div");
    meta.className = "file-meta";
    meta.innerHTML = `<span title="${file.name}">${
      file.name.length > 30 ? file.name.slice(0, 27) + "..." : file.name
    }</span><span>${(file.size / 1024 / 1024).toFixed(2)} MB</span>`;

    const actions = document.createElement("div");
    actions.style.marginTop = "8px";
    actions.innerHTML = `<button class="btn btn-outline" onclick="removeFileVideo(${idx})"><i class="fas fa-trash"></i> Hapus</button>`;

    // CHANGED: Urutan elemen - pertanyaan di atas, video di tengah
    item.appendChild(questionWrapper); // 1. Pertanyaan di atas
    item.appendChild(wrapper); // 2. Video di tengah
    item.appendChild(meta); // 3. Meta info
    item.appendChild(actions); // 4. Actions di bawah
    previewGridVideo.appendChild(item);
  });

  if (statusAreaVideo)
    resetStatus(
      statusAreaVideo,
      `${selectedFilesVideo.length} file siap diproses.`
    );

  // Update button mode after rendering
  updateSendVideoButtonMode();
}

function removeFileVideo(index) {
  try {
    if (objectUrlsVideo[index]) URL.revokeObjectURL(objectUrlsVideo[index]);
  } catch (e) {}
  selectedFilesVideo.splice(index, 1);
  videoQuestions.splice(index, 1); // NEW: Hapus pertanyaan juga
  objectUrlsVideo.splice(index, 1);
  renderPreviewsVideo();
  updateClearButtonsVisibility();
}

function clearAllVideo() {
  objectUrlsVideo.forEach((u) => {
    try {
      URL.revokeObjectURL(u);
    } catch (e) {}
  });
  selectedFilesVideo = [];
  videoQuestions = []; // NEW: Reset pertanyaan
  objectUrlsVideo = [];
  if (previewGridVideo) previewGridVideo.innerHTML = "";
  if (statusAreaVideo) resetStatus(statusAreaVideo, "Tidak ada file dipilih.");
  if (participantNameInput) participantNameInput.value = "";
  updateClearButtonsVisibility();
}

/* ============================
   SESSION & POLLING
   ============================ */
function checkOngoingSession() {
  const savedSession = localStorage.getItem(SESSION_STORAGE_KEY);
  if (!savedSession) {
    // NO SESSION - Force hide loading
    hideLoading();
    return;
  }

  try {
    const { sessionId, candidateName, startTime, videoCount } =
      JSON.parse(savedSession);
    const elapsed = Date.now() - startTime;

    if (elapsed > 30 * 60 * 1000) {
      console.log("Session expired (>30 min), removing...");
      localStorage.removeItem(SESSION_STORAGE_KEY);
      hideLoading(); // Force hide
      return;
    }

    console.log(`🔄 Resuming session: ${sessionId}`);

    // Start polling FIRST without showing loading
    // Only show loading AFTER we confirm session exists on server
    isSubmittingVideo = true;

    // Check session status first
    verifyAndResumeSession(sessionId, candidateName, videoCount);
  } catch (e) {
    console.error("Failed to parse session:", e);
    localStorage.removeItem(SESSION_STORAGE_KEY);
    hideLoading(); // Force hide
  }
}

async function verifyAndResumeSession(sessionId, candidateName, videoCount) {
  try {
    // Check if session exists on server FIRST
    const statusRes = await fetch(
      // `http://127.0.0.1:8888/status/${sessionId}`,
      `${API_BASE_URL}/status/${sessionId}`,
      {
        method: "GET",
        cache: "no-cache",
        headers: {
          Accept: "application/json",
          "ngrok-skip-browser-warning": "true",
        },
      }
    );

    if (!statusRes.ok) {
      console.warn(
        `Session not found on server (${statusRes.status}), clearing...`
      );
      localStorage.removeItem(SESSION_STORAGE_KEY);
      hideLoading();
      isSubmittingVideo = false;

      if (statusAreaVideo) {
        statusAreaVideo.textContent =
          "Session tidak ditemukan di server. Silakan upload ulang.";
        statusAreaVideo.style.color = "#f39c12";
      }
      return;
    }

    const statusData = await statusRes.json();

    // Check if already completed
    if (statusData.status === "completed") {
      console.log("Session already completed, redirecting...");
      localStorage.removeItem(SESSION_STORAGE_KEY);
      hideLoading();
      window.location.href = `halaman_dasboard.html?session=${sessionId}`;
      return;
    }

    // Check if error
    if (statusData.status === "error") {
      console.log("Session has error, clearing...");
      localStorage.removeItem(SESSION_STORAGE_KEY);
      hideLoading();
      isSubmittingVideo = false;

      if (statusAreaVideo) {
        statusAreaVideo.textContent =
          "❌ Processing error: " + (statusData.error || "Unknown");
        statusAreaVideo.style.color = "#e74c3c";
      }

      alert(
        "❌ Processing sebelumnya gagal!\n\n" +
          "Error: " +
          (statusData.error || "Unknown") +
          "\n\n" +
          "Silakan upload ulang."
      );
      return;
    }

    // Session is valid and processing - NOW show loading
    console.log("✅ Valid session found, starting polling...");
    showLoading(
      `Processing transcriptions...\nCandidate: ${candidateName}\nVideos: ${videoCount}\n\nPlease wait, this may take several minutes.`
    );

    // Start polling
    resumePolling(sessionId, candidateName);
  } catch (err) {
    console.error("Failed to verify session:", err);
    localStorage.removeItem(SESSION_STORAGE_KEY);
    hideLoading();
    isSubmittingVideo = false;

    if (statusAreaVideo) {
      statusAreaVideo.textContent =
        "❌ Gagal menghubungi server: " + err.message;
      statusAreaVideo.style.color = "#e74c3c";
    }
  }
}

async function resumePolling(sessionId, candidateName) {
  const pollStatus = async () => {
    try {
      const statusRes = await fetch(
        // `http://127.0.0.1:8888/status/${sessionId}`,
        `${API_BASE_URL}/status/${sessionId}`,
        {
          method: "GET",
          cache: "no-cache",
          headers: {
            Accept: "application/json",
            "ngrok-skip-browser-warning": "true",
          },
        }
      );

      if (!statusRes.ok) {
        console.warn(`Status check failed: ${statusRes.status}`);

        // If 404 after many attempts, session might be invalid
        if (statusRes.status === 404) {
          return false; // Continue polling for a while
        }

        return false;
      }

      const statusData = await statusRes.json();
      console.log("Status:", statusData);

      if (statusData.status === "completed") {
        localStorage.removeItem(SESSION_STORAGE_KEY);
        showLoading("Processing complete!\nRedirecting to dashboard...");
        await new Promise((r) => setTimeout(r, 1500));
        hideLoading();
        isSubmittingVideo = false;
        window.location.href = `halaman_dasboard.html?session=${sessionId}`;
        return true;
      } else if (statusData.status === "error") {
        // ERROR DETECTED - Hide overlay and show error
        localStorage.removeItem(SESSION_STORAGE_KEY);
        hideLoading();
        isSubmittingVideo = false;

        const errorDetail =
          statusData.error_detail || statusData.error || "Unknown error";
        alert(
          "❌ Processing gagal!\n\n" +
            "Error: " +
            errorDetail +
            "\n\n" +
            "Silakan:\n" +
            "1. Cek log server untuk detail\n" +
            "2. Pastikan video format valid\n" +
            "3. Coba upload ulang dengan file berbeda"
        );

        // Update status area
        if (statusAreaVideo) {
          statusAreaVideo.textContent =
            "❌ Processing error: " + (statusData.error || "Unknown");
          statusAreaVideo.style.color = "#e74c3c";
        }

        return true; // Stop polling
      } else if (
        statusData.status === "processing" ||
        statusData.status === "uploading"
      ) {
        const progress = statusData.progress || "";
        const message = statusData.message || "Processing...";
        showLoading(
          `Processing transcriptions...\nCandidate: ${candidateName}\n${message}\nProgress: ${progress}\n\nPlease wait, this may take several minutes.`
        );
        return false;
      } else if (statusData.status === "not_found") {
        console.warn("Session not found yet, will retry...");
        return false;
      }
      return false;
    } catch (err) {
      console.error("Polling error:", err);
      return false; // Continue polling despite network errors
    }
  };

  let attempts = 0;
  const maxAttempts = 360; // 30 minutes max (5s interval)

  const poll = async () => {
    if (++attempts > maxAttempts) {
      // TIMEOUT - Hide overlay and notify user
      localStorage.removeItem(SESSION_STORAGE_KEY);
      hideLoading();
      isSubmittingVideo = false;

      alert(
        "⏱️ Processing timeout!\n\n" +
          "Proses memakan waktu lebih dari 30 menit.\n\n" +
          "Kemungkinan:\n" +
          "1. Server masih memproses (cek dashboard)\n" +
          "2. Video terlalu besar atau panjang\n" +
          "3. Server error (cek log server)\n\n" +
          "Silakan:\n" +
          "- Cek dashboard untuk hasil\n" +
          "- Atau upload ulang dengan file lebih kecil"
      );
      // Update status
      if (statusAreaVideo) {
        statusAreaVideo.textContent =
          "⏱️ Processing timeout - cek dashboard atau upload ulang";
        statusAreaVideo.style.color = "#f39c12";
      }

      return;
    }

    if (!(await pollStatus())) {
      setTimeout(poll, 5000);
    }
  };

  poll();
}

async function buildAndSendVideo() {
  event?.preventDefault?.();

  const now = Date.now();
  if (isSubmittingVideo) {
    console.warn("Already submitting");
    return false;
  }
  if (now - lastSubmitTime < SUBMIT_DEBOUNCE_MS) {
    alert(
      `Mohon tunggu ${SUBMIT_DEBOUNCE_MS / 1000} detik sebelum mengirim ulang.`
    );
    return;
  }

  if (!selectedFilesVideo.length)
    return alert("Pilih minimal satu file video.");
  const name = participantNameInput.value.trim();
  if (!name) return alert("Nama Peserta wajib diisi.");

  // NEW: Validasi pertanyaan
  const emptyQuestions = [];
  videoQuestions.forEach((q, idx) => {
    if (!q.trim()) {
      emptyQuestions.push(idx + 1);
    }
  });

  if (emptyQuestions.length > 0) {
    return alert(
      `Pertanyaan wajib diisi untuk semua video!\n\nVideo tanpa pertanyaan: ${emptyQuestions.join(
        ", "
      )}`
    );
  }

  isSubmittingVideo = true;
  lastSubmitTime = now;

  const tempSessionId =
    "temp_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);

  try {
    // PHASE 1: Upload - SHOW loading here
    showLoading(
      `Uploading ${selectedFilesVideo.length} video(s)...\n\nPlease wait, do not close this page.`
    );
    resetStatus(statusAreaVideo, "Mengunggah video ke server...");

    const formData = new FormData();
    formData.append("candidate_name", name);

    // NEW: Tambahkan pertanyaan untuk setiap video
    selectedFilesVideo.forEach((file, idx) => {
      formData.append("videos", file);
      formData.append("questions", videoQuestions[idx].trim());
    });

    console.log(
      `📤 Uploading ${selectedFilesVideo.length} video(s) with questions...`
    );

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000);

    const uploadPromise = new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const percentComplete = Math.round((e.loaded / e.total) * 100);
          showLoading(
            `Uploading ${selectedFilesVideo.length} video(s)...\n${percentComplete}% complete\n\nPlease wait, do not close this page.`
          );
        }
      });

      xhr.addEventListener("load", () => {
        clearTimeout(timeoutId);
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            resolve(result);
          } catch (e) {
            reject(
              new Error(`Failed to parse server response: ${xhr.responseText}`)
            );
          }
        } else {
          reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
        }
      });

      xhr.addEventListener("error", () => {
        clearTimeout(timeoutId);
        reject(new Error("Network error during upload"));
      });

      xhr.addEventListener("abort", () => {
        clearTimeout(timeoutId);
        reject(
          new Error("Upload timeout - file too large or network too slow")
        );
      });

      xhr.open("POST", VIDEO_ENDPOINT);
      xhr.setRequestHeader("ngrok-skip-browser-warning", "true");
      xhr.send(formData);
    });

    const result = await uploadPromise;
    console.log("✅ Upload complete:", result);

    // Validate response
    if (!result.success) {
      throw new Error(result.error || "Upload failed - server returned error");
    }

    // Get session ID
    const sessionId = result.session_id || tempSessionId;
    console.log(`📊 Session ID: ${sessionId}`);

    // PHASE 2: Save session
    const sessionData = {
      sessionId,
      candidateName: name,
      videoCount: selectedFilesVideo.length,
      startTime: Date.now(),
    };
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessionData));
    console.log("💾 Session saved to localStorage");

    // Update loading message
    showLoading(
      `Upload complete!\n\nProcessing transcriptions...\nCandidate: ${name}\nVideos: ${selectedFilesVideo.length}\n\nThis page will auto-reload.\nProcessing will continue in background.`
    );

    // Wait then reload
    await new Promise((r) => setTimeout(r, 3000));

    console.log("🔄 Reloading page to start background polling...");
    window.location.reload();

    return false;
  } catch (err) {
    console.error("❌ Upload error:", err);

    // ALWAYS hide loading on error
    hideLoading();
    isSubmittingVideo = false;

    if (statusAreaVideo) {
      statusAreaVideo.textContent = "❌ Upload gagal: " + err.message;
      statusAreaVideo.style.color = "#e74c3c";
    }

    let errorMessage = "Gagal mengirim video ke server.\n\n";

    if (err.message.includes("Network error")) {
      errorMessage += "Penyebab: Koneksi jaringan terputus.\n\n";
      errorMessage += "Solusi:\n";
      errorMessage +=
        "1. Pastikan server FastAPI berjalan di http://127.0.0.1:8888\n";
      errorMessage += "2. Cek koneksi internet Anda\n";
      errorMessage += "3. Coba refresh halaman dan upload ulang";
    } else if (err.message.includes("timeout")) {
      errorMessage +=
        "Penyebab: Upload memakan waktu terlalu lama (>10 menit).\n\n";
      errorMessage += "Solusi:\n";
      errorMessage += "1. Gunakan file video yang lebih kecil\n";
      errorMessage += "2. Compress video terlebih dahulu\n";
      errorMessage += "3. Upload lebih sedikit video sekaligus";
    } else if (err.message.includes("Failed to parse")) {
      errorMessage += "Penyebab: Server mengirim respons yang tidak valid.\n\n";
      errorMessage += "Solusi:\n";
      errorMessage += "1. Restart server FastAPI\n";
      errorMessage += "2. Cek log server untuk error\n";
      errorMessage += "3. Pastikan semua dependencies terinstall";
    } else if (err.message.includes("Upload failed")) {
      errorMessage += `Penyebab: ${err.message}\n\n`;
      errorMessage += "Solusi:\n";
      errorMessage += "1. Cek log server FastAPI untuk detail error\n";
      errorMessage += "2. Pastikan format video didukung (.webm, .mp4)\n";
      errorMessage += "3. Coba dengan file yang lebih kecil";
    } else {
      errorMessage += `Error: ${err.message}\n\n`;
      errorMessage += "Silakan coba lagi atau hubungi administrator.";
    }

    alert(errorMessage);

    return false;
  }
}

/* ============================
   JSON HANDLERS
   ============================ */
function handleFilesJSON(fileList) {
  const files = Array.from(fileList).filter(
    (f) =>
      f.name.toLowerCase().endsWith(".json") || f.type === "application/json"
  );
  if (!files.length) return alert("Tidak ada file JSON terdeteksi.");
  if (files.length > 1)
    alert("Hanya satu file JSON diperbolehkan. Menggunakan file pertama.");
  selectedFilesJSON = [files[0]];
  renderPreviewsJSON();
}

async function renderPreviewsJSON() {
  if (!previewGridJSON) return;
  previewGridJSON.innerHTML = "";

  for (let idx = 0; idx < selectedFilesJSON.length; idx++) {
    const file = selectedFilesJSON[idx];
    const item = document.createElement("div");
    item.className = "preview-item";

    const meta = document.createElement("div");
    meta.className = "file-meta";
    meta.innerHTML = `<span title="${file.name}">${
      file.name.length > 30 ? file.name.slice(0, 27) + "..." : file.name
    }</span><span>${(file.size / 1024).toFixed(1)} KB</span>`;

    const actions = document.createElement("div");
    actions.style.marginTop = "8px";
    actions.innerHTML = `<button class="btn btn-outline" onclick="removeFileJSON(${idx})"><i class="fas fa-trash"></i> Hapus</button>`;

    item.appendChild(meta);

    try {
      const txt = await file.text();
      let pretty = txt;
      try {
        pretty = JSON.stringify(JSON.parse(txt), null, 2);
      } catch (e) {}

      const pre = document.createElement("pre");
      pre.className = "json-preview";
      pre.textContent = pretty;
      pre.style.cssText =
        "max-height:240px;overflow:auto;background:#f7f7f7;padding:8px;border-radius:4px;border:1px solid #e6e6e6;font-family:monospace;font-size:12px;white-space:pre-wrap;margin-top:8px";
      item.appendChild(pre);
    } catch (err) {
      const errNote = document.createElement("div");
      errNote.className = "muted";
      errNote.textContent = "Preview gagal dibuat.";
      item.appendChild(errNote);
    }

    item.appendChild(actions);
    previewGridJSON.appendChild(item);
  }

  if (statusAreaJSON)
    resetStatus(
      statusAreaJSON,
      `${selectedFilesJSON.length} file JSON siap diproses.`
    );
  updateClearButtonsVisibility();

  // Update button mode after rendering
  updateSendJSONButtonMode();
}

function removeFileJSON(index) {
  selectedFilesJSON.splice(index, 1);
  renderPreviewsJSON();
  updateClearButtonsVisibility();
}

function clearAllJSON() {
  selectedFilesJSON = [];
  previewGridJSON.innerHTML = "";
  resetStatus(statusAreaJSON, "Belum ada file JSON dipilih.");
  updateClearButtonsVisibility();
}

async function buildAndSendJSONFiles() {
  event?.preventDefault?.();

  const now = Date.now();
  if (isSubmittingJSON) {
    console.warn("Already submitting JSON");
    return false;
  }
  if (now - lastSubmitTimeJSON < SUBMIT_DEBOUNCE_MS) {
    alert(
      `Mohon tunggu ${SUBMIT_DEBOUNCE_MS / 1000} detik sebelum mengirim ulang.`
    );
    return;
  }

  if (!selectedFilesJSON.length) {
    return alert("Pilih file JSON terlebih dahulu.");
  }

  isSubmittingJSON = true;
  lastSubmitTimeJSON = now;

  try {
    // Read JSON file
    const file = selectedFilesJSON[0];
    const jsonText = await file.text();
    let jsonData;

    try {
      jsonData = JSON.parse(jsonText);
    } catch (e) {
      throw new Error("File JSON tidak valid atau rusak.");
    }

    // Validate JSON structure
    if (
      !jsonData.success ||
      !jsonData.data ||
      !jsonData.data.candidate ||
      !jsonData.data.reviewChecklists ||
      !jsonData.data.reviewChecklists.interviews
    ) {
      throw new Error(
        "Struktur JSON tidak valid. Pastikan JSON mengandung data candidate dan interviews."
      );
    }

    const candidateName = jsonData.data.candidate.name;
    const interviews = jsonData.data.reviewChecklists.interviews;

    if (!candidateName) {
      throw new Error("Nama kandidat tidak ditemukan di JSON.");
    }

    if (!interviews || !Array.isArray(interviews) || interviews.length === 0) {
      throw new Error("Tidak ada data interview di JSON.");
    }

    console.log(`📋 Processing JSON for: ${candidateName}`);
    console.log(`📹 Found ${interviews.length} interview videos`);

    // Show loading
    showLoading(
      `Sending JSON data...\nCandidate: ${candidateName}\nVideos: ${interviews.length}\n\nPlease wait...`
    );
    resetStatus(statusAreaJSON, "Mengirim data ke server...");

    // Send to server
    const response = await fetch(`${VIDEO_ENDPOINT}_json`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(jsonData),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Server error: ${response.status} - ${errorText || response.statusText}`
      );
    }

    const result = await response.json();
    console.log("✅ JSON processing started:", result);

    if (!result.success) {
      throw new Error(result.error || "Server returned error");
    }

    const sessionId = result.session_id;
    console.log(`📊 Session ID: ${sessionId}`);

    // Save session
    const sessionData = {
      sessionId,
      candidateName,
      videoCount: interviews.length,
      startTime: Date.now(),
    };
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessionData));
    console.log("💾 Session saved to localStorage");

    // Update loading message
    showLoading(
      `JSON uploaded!\n\nDownloading and processing videos...\nCandidate: ${candidateName}\nVideos: ${interviews.length}\n\nThis page will auto-reload.\nProcessing will continue in background.`
    );

    // Clear JSON selection
    clearAllJSON();

    // Wait then reload
    await new Promise((r) => setTimeout(r, 3000));

    console.log("🔄 Reloading page to start background polling...");
    window.location.reload();

    return false;
  } catch (err) {
    console.error("❌ JSON upload error:", err);

    // ALWAYS hide loading on error
    hideLoading();
    isSubmittingJSON = false;

    // Update status area
    if (statusAreaJSON) {
      statusAreaJSON.textContent = "❌ Upload gagal: " + err.message;
      statusAreaJSON.style.color = "#e74c3c";
    }

    // Show user-friendly error message
    let errorMessage = "Gagal mengirim JSON ke server.\n\n";

    if (err.message.includes("tidak valid")) {
      errorMessage += `Penyebab: ${err.message}\n\n`;
      errorMessage += "Solusi:\n";
      errorMessage += "1. Pastikan file JSON memiliki struktur yang benar\n";
      errorMessage +=
        "2. Cek field: candidate.name dan reviewChecklists.interviews\n";
      errorMessage += "3. Validasi JSON di jsonlint.com";
    } else if (err.message.includes("Server error")) {
      errorMessage += `Penyebab: ${err.message}\n\n`;
      errorMessage += "Solusi:\n";
      errorMessage +=
        "1. Pastikan server FastAPI berjalan di http://127.0.0.1:8888\n";
      errorMessage += "2. Cek log server untuk detail error\n";
      errorMessage += "3. Restart server jika perlu";
    } else {
      errorMessage += `Error: ${err.message}\n\n`;
      errorMessage += "Silakan coba lagi atau hubungi administrator.";
    }

    alert(errorMessage);

    return false;
  }
}

/* ============================
   INITIALIZATION
   ============================ */
function initUploadModule() {
  fileInputVideo = document.getElementById("fileInputVideo");
  previewGridVideo = document.getElementById("previewGridVideo");
  statusAreaVideo = document.getElementById("statusAreaVideo");
  participantNameInput = document.getElementById("participantName");
  clearAllVideoBtn = document.getElementById("clearAllVideoBtn");
  clearAllJSONBtn = document.getElementById("clearAllJSONBtn");

  if (fileInputVideo) {
    const newFileInput = fileInputVideo.cloneNode(true);
    fileInputVideo.parentNode.replaceChild(newFileInput, fileInputVideo);
    fileInputVideo = newFileInput;
    fileInputVideo.addEventListener("change", (e) => {
      handleFilesVideo(e.target.files);
      e.target.value = "";
    });
  }

  const uploadAreaVideo = document.getElementById("uploadAreaVideo");
  if (uploadAreaVideo) {
    let isClickHandled = false;
    uploadAreaVideo.addEventListener("click", (e) => {
      if (isClickHandled) return;
      e.stopPropagation();
      isClickHandled = true;
      fileInputVideo?.click();
      setTimeout(() => (isClickHandled = false), 500);
    });
    uploadAreaVideo.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadAreaVideo.style.opacity = "0.8";
    });
    uploadAreaVideo.addEventListener("dragleave", (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadAreaVideo.style.opacity = "1";
    });
    uploadAreaVideo.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadAreaVideo.style.opacity = "1";
      if (e.dataTransfer?.files?.length) handleFilesVideo(e.dataTransfer.files);
    });
  }

  fileInputJSON = document.getElementById("fileInputJSON");
  previewGridJSON = document.getElementById("previewGridJSON");
  statusAreaJSON = document.getElementById("statusAreaJSON");
  const uploadAreaJSON = document.getElementById("uploadAreaJSON");

  if (fileInputJSON) {
    const newFileInputJSON = fileInputJSON.cloneNode(true);
    fileInputJSON.parentNode.replaceChild(newFileInputJSON, fileInputJSON);
    fileInputJSON = newFileInputJSON;
    fileInputJSON.addEventListener("change", (e) => {
      handleFilesJSON(e.target.files);
      e.target.value = "";
    });
  }

  if (uploadAreaJSON) {
    let isClickHandledJSON = false;
    uploadAreaJSON.addEventListener("click", (e) => {
      if (isClickHandledJSON) return;
      e.stopPropagation();
      isClickHandledJSON = true;
      fileInputJSON?.click();
      setTimeout(() => (isClickHandledJSON = false), 500);
    });
    uploadAreaJSON.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadAreaJSON.style.opacity = "0.8";
    });
    uploadAreaJSON.addEventListener("dragleave", (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadAreaJSON.style.opacity = "1";
    });
    uploadAreaJSON.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadAreaJSON.style.opacity = "1";
      if (e.dataTransfer?.files?.length) handleFilesJSON(e.dataTransfer.files);
    });
  }

  // FORCE HIDE LOADING on page load
  hideLoading();

  updateClearButtonsVisibility();

  // Set initial button modes
  updateSendVideoButtonMode();
  updateSendJSONButtonMode();
}

let isInitialized = false;
function initialize() {
  if (isInitialized) return;
  isInitialized = true;

  // STEP 1: Always hide loading first
  hideLoading();

  // STEP 2: Initialize UI
  initUploadModule();

  // STEP 3: Check for ongoing session (will show loading only if valid session exists)
  checkOngoingSession();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initialize, { once: true });
} else {
  initialize();
}
