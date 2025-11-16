/* ============================
   VARIABLES
   ============================ */
// Video
let fileInputVideo, previewGridVideo, statusAreaVideo, participantNameInput;
let selectedFilesVideo = [];
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
const VIDEO_ENDPOINT = "http://127.0.0.1:8888/upload";
const DEFAULT_BASE_URL = "http://127.0.0.1:5500";
const SUBMIT_DEBOUNCE_MS = 3000;
const SESSION_STORAGE_KEY = "video_processing_session";

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
    if (clearAllVideoBtn)
      clearAllVideoBtn.style.display = selectedFilesVideo.length ? "" : "none";
    if (clearAllJSONBtn)
      clearAllJSONBtn.style.display = selectedFilesJSON.length ? "" : "none";
  } catch (e) {}
}

/* ============================
   VIDEO HANDLERS
   ============================ */
function handleFilesVideo(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("video/"));
  if (!files.length) return alert("Tidak ada file video terdeteksi.");

  selectedFilesVideo = selectedFilesVideo.concat(files);
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

    item.appendChild(wrapper);
    item.appendChild(meta);
    item.appendChild(actions);
    previewGridVideo.appendChild(item);
  });

  if (statusAreaVideo)
    resetStatus(
      statusAreaVideo,
      `${selectedFilesVideo.length} file siap diproses.`
    );
}

function removeFileVideo(index) {
  try {
    if (objectUrlsVideo[index]) URL.revokeObjectURL(objectUrlsVideo[index]);
  } catch (e) {}
  selectedFilesVideo.splice(index, 1);
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
  if (!savedSession) return;

  try {
    const { sessionId, candidateName, startTime, videoCount } =
      JSON.parse(savedSession);
    const elapsed = Date.now() - startTime;

    if (elapsed > 30 * 60 * 1000) {
      localStorage.removeItem(SESSION_STORAGE_KEY);
      return;
    }

    console.log(`🔄 Resuming session: ${sessionId}`);
    showLoading(
      `Processing transcriptions...\nCandidate: ${candidateName}\nVideos: ${videoCount}\n\nPlease wait, this may take several minutes.`
    );
    isSubmittingVideo = true;

    // Start polling after small delay to let server initialize
    setTimeout(() => {
      resumePolling(sessionId, candidateName);
    }, 2000);
  } catch (e) {
    console.error("Failed to parse session:", e);
    localStorage.removeItem(SESSION_STORAGE_KEY);
  }
}

async function resumePolling(sessionId, candidateName) {
  const pollStatus = async () => {
    try {
      const statusRes = await fetch(
        `http://127.0.0.1:8888/status/${sessionId}`,
        {
          method: "GET",
          cache: "no-cache",
          headers: {
            Accept: "application/json",
          },
        }
      );

      if (!statusRes.ok) {
        console.warn(`Status check failed: ${statusRes.status}`);
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
        localStorage.removeItem(SESSION_STORAGE_KEY);
        hideLoading();
        alert(
          "Processing gagal!\n\nError: " +
            (statusData.error || "Unknown error") +
            "\n\nSilakan coba lagi."
        );
        isSubmittingVideo = false;
        return true;
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
        return false; // Keep trying for a while
      }
      return false;
    } catch (err) {
      console.error("Polling error:", err);
      return false;
    }
  };

  let attempts = 0;
  const poll = async () => {
    if (++attempts > 360) {
      localStorage.removeItem(SESSION_STORAGE_KEY);
      hideLoading();
      alert(
        "Processing timeout!\n\nProses memakan waktu terlalu lama.\nSilakan cek hasil di dashboard atau coba lagi."
      );
      isSubmittingVideo = false;
      return;
    }
    if (!(await pollStatus())) setTimeout(poll, 5000);
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

  isSubmittingVideo = true;
  lastSubmitTime = now;

  // Generate temporary session ID immediately
  const tempSessionId =
    "temp_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);

  try {
    // PHASE 1: Upload (show loading immediately)
    showLoading(
      `Uploading ${selectedFilesVideo.length} video(s)...\n\nPlease wait, do not close this page.`
    );
    resetStatus(statusAreaVideo, "Mengunggah video ke server...");

    const formData = new FormData();
    formData.append("candidate_name", name);
    selectedFilesVideo.forEach((file) => formData.append("videos", file));

    console.log(`📤 Uploading ${selectedFilesVideo.length} video(s)...`);

    // Upload with longer timeout for large files
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minutes

    // Use XHR instead of fetch to avoid CORS issues
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
            // If can't parse, assume success with temp ID
            resolve({ success: true, session_id: tempSessionId });
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
        reject(new Error("Upload timeout"));
      });

      xhr.open("POST", VIDEO_ENDPOINT);
      xhr.send(formData);
    });

    const result = await uploadPromise;
    console.log("✅ Upload complete:", result);

    // Get session ID (from server or use temp)
    const sessionId = result.session_id || tempSessionId;
    console.log(`📊 Session ID: ${sessionId}`);

    // PHASE 2: Save session and show processing message
    const sessionData = {
      sessionId,
      candidateName: name,
      videoCount: selectedFilesVideo.length,
      startTime: Date.now(),
    };
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessionData));
    console.log("💾 Session saved to localStorage");

    // Update loading message for processing phase
    showLoading(
      `Upload complete!\n\nProcessing transcriptions...\nCandidate: ${name}\nVideos: ${selectedFilesVideo.length}\n\nThis page will auto-reload.\nProcessing will continue in background.`
    );

    // Wait a moment, then reload page to start polling
    await new Promise((r) => setTimeout(r, 3000));

    console.log("🔄 Reloading page to start background polling...");
    window.location.reload();

    return false;
  } catch (err) {
    console.error("❌ Upload error:", err);
    hideLoading();

    // Even if upload failed, if we have temp session, save it
    if (
      err.message.includes("Network error") ||
      err.message.includes("timeout")
    ) {
      // Save temp session anyway - server might have received it
      const sessionData = {
        sessionId: tempSessionId,
        candidateName: name,
        videoCount: selectedFilesVideo.length,
        startTime: Date.now(),
      };
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessionData));

      alert(
        "Upload mungkin berhasil meskipun ada network error.\n\nHalaman akan di-reload untuk check status."
      );

      setTimeout(() => {
        window.location.reload();
      }, 2000);
    } else {
      statusAreaVideo.textContent = "Terjadi kesalahan saat mengirim.";
      alert("Error: " + err.message);
      isSubmittingVideo = false;
    }
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

  updateClearButtonsVisibility();
}

let isInitialized = false;
function initialize() {
  if (isInitialized) return;
  isInitialized = true;
  initUploadModule();
  checkOngoingSession();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initialize, { once: true });
} else {
  initialize();
}
