/* ============================
   Video upload variables
   ============================ */
let fileInputVideo;
let previewGridVideo;
let statusAreaVideo;
let participantNameInput;

let selectedFilesVideo = [];
let objectUrlsVideo = []; // track created blob URLs so we can revoke reliably

// Replace previous SERVER_MODE/FS fallback logic with simple server upload endpoints
const FILE_UPLOAD_ENDPOINT = "http://127.0.0.1:8888/upload_file";
const DELETE_ENDPOINT = "http://127.0.0.1:8888/delete_file";
const VIDEO_ENDPOINT = "http://127.0.0.1:8888/upload";
const JSON_ENDPOINT = VIDEO_ENDPOINT;

// state
let uploadedUrls = [];
let uploadedNames = [];

// NEW: flag to prevent duplicate submission
let isSubmittingVideo = false;
let lastSubmitTime = 0;
const SUBMIT_DEBOUNCE_MS = 3000; // 3 detik debounce

// DEFAULT base URL — ganti dengan domain produksi ketika di-deploy
const DEFAULT_BASE_URL = "http://127.0.0.1:5500"; // <- ubah ini nanti ke https://your-domain.com

// Get base URL (gunakan konstanta, bukan input)
function getBaseUrl() {
  return DEFAULT_BASE_URL.replace(/\/+$/, "");
}

// Loading overlay helpers
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

// --- NEW: inject .status-ready style once and helpers to mark/reset status ---
let __statusReadyInjected = false;
function injectStatusReadyStyle() {
  if (__statusReadyInjected) return;
  __statusReadyInjected = true;
  const s = document.createElement("style");
  s.textContent = `
    .status-ready {
      background: linear-gradient(180deg,#28a745,#1e7e34);
      color: #fff !important;
      padding: 4px 8px;
      border-radius: 6px;
      font-weight: 700;
      display: inline-block;
      box-shadow: 0 2px 6px rgba(30,126,52,0.25);
    }
  `;
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

// Upload single file: accept either index or File object; on failure remove from preview
async function uploadOne(indexOrFile) {
  // resolve file and original index snapshot
  let originalFile = null;
  let originalIndex = -1;
  if (typeof indexOrFile === "number") {
    originalIndex = indexOrFile;
    originalFile = selectedFilesVideo[originalIndex] || null;
  } else {
    originalFile = indexOrFile || null;
    // don't rely on index here; will look up current index later
  }

  if (!originalFile) return null;

  // show uploading state and ensure previous "ready" styling is removed
  resetStatus(statusAreaVideo);
  statusAreaVideo.textContent = `Mengunggah file ke server...`;

  try {
    const fd = new FormData();
    fd.append("file", originalFile);
    const res = await fetch(FILE_UPLOAD_ENDPOINT, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) throw new Error("Upload failed: " + res.status);
    const j = await res.json();
    const url = j?.url || null;
    const name = j?.name || null;

    // find current index of this file (may have moved); update arrays at that index
    const curIndex = selectedFilesVideo.indexOf(originalFile);
    if (curIndex >= 0) {
      if (url) uploadedUrls[curIndex] = url;
      if (name) uploadedNames[curIndex] = name;
    } else {
      // fallback: if originalIndex still valid, set there
      if (originalIndex >= 0) {
        if (url) uploadedUrls[originalIndex] = url;
        if (name) uploadedNames[originalIndex] = name;
      }
    }

    renderPreviewsVideo();
    resetStatus(statusAreaVideo, `File terunggah ke server.`);
    return url;
  } catch (err) {
    console.error("Upload to server error", err);
    // remove file from selection if still present
    try {
      const curIndex = selectedFilesVideo.indexOf(originalFile);
      if (curIndex >= 0) {
        // attempt server delete if it has a name
        const serverName = uploadedNames[curIndex];
        if (serverName) {
          try {
            await fetch(DELETE_ENDPOINT, {
              method: "DELETE",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name: serverName }),
            });
          } catch (e) {
            // ignore
          }
        }
        // revoke object URL if any
        try {
          if (objectUrlsVideo[curIndex]) {
            URL.revokeObjectURL(objectUrlsVideo[curIndex]);
          }
        } catch (e) {
          /* ignore */
        }
        // remove entries
        selectedFilesVideo.splice(curIndex, 1);
        uploadedUrls.splice(curIndex, 1);
        uploadedNames.splice(curIndex, 1);
        objectUrlsVideo.splice(curIndex, 1);
        // re-render previews and update clear button visibility
        renderPreviewsVideo();
        updateClearButtonsVisibility();
      }
    } catch (e) {
      /* ignore removal errors */
    }

    resetStatus(
      statusAreaVideo,
      `Gagal unggah file ke server. File dihapus dari preview.`
    );
    return null;
  }
}

// Saat user drop / pilih file: HANYA preview, TIDAK upload
function handleFilesVideo(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("video/"));
  if (files.length === 0) {
    alert("Tidak ada file video terdeteksi.");
    return;
  }

  // append files and ensure arrays alignment
  const startIndex = selectedFilesVideo.length;
  selectedFilesVideo = selectedFilesVideo.concat(files);
  // extend storage arrays (kosong dulu, akan diisi saat upload)
  for (let i = 0; i < files.length; i++) {
    uploadedUrls[startIndex + i] = null;
    uploadedNames[startIndex + i] = null;
  }
  renderPreviewsVideo();

  // update clear button visibility
  updateClearButtonsVisibility();

  // REMOVED: Auto-upload - sekarang hanya upload saat tombol "Kirim" ditekan
  // files.forEach((f) => {
  //   uploadOne(f);
  // });
}

// --- CHANGED: defer DOM element binding & event listener setup until DOM ready ---
function initUploadModule() {
  // bind elements (safe even if script included at bottom; helps live-server timing issues)
  fileInputVideo = document.getElementById("fileInputVideo");
  previewGridVideo = document.getElementById("previewGridVideo");
  statusAreaVideo = document.getElementById("statusAreaVideo");
  participantNameInput = document.getElementById("participantName");

  // bind clear buttons
  clearAllVideoBtn = document.getElementById("clearAllVideoBtn");
  clearAllJSONBtn = document.getElementById("clearAllJSONBtn");

  if (fileInputVideo) {
    fileInputVideo.addEventListener("change", (e) =>
      handleFilesVideo(e.target.files)
    );
  }

  const uploadAreaVideo = document.getElementById("uploadAreaVideo");
  if (uploadAreaVideo) {
    uploadAreaVideo.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadAreaVideo.style.opacity = "0.8";
    });
    uploadAreaVideo.addEventListener("dragleave", () => {
      uploadAreaVideo.style.opacity = "1";
    });
    uploadAreaVideo.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadAreaVideo.style.opacity = "1";
      if (e.dataTransfer?.files) handleFilesVideo(e.dataTransfer.files);
    });
  }

  // JSON area binding (keep original behaviour)
  // --- CHANGED: bind globals so renderPreviewsJSON can access them ---
  fileInputJSON = document.getElementById("fileInputJSON");
  previewGridJSON = document.getElementById("previewGridJSON");
  statusAreaJSON = document.getElementById("statusAreaJSON");
  const uploadAreaJSON = document.getElementById("uploadAreaJSON");

  if (fileInputJSON) {
    fileInputJSON.addEventListener("change", (e) =>
      handleFilesJSON(e.target.files)
    );
  }
  if (uploadAreaJSON) {
    uploadAreaJSON.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadAreaJSON.style.opacity = "0.8";
    });
    uploadAreaJSON.addEventListener(
      "dragleave",
      () => (uploadAreaJSON.style.opacity = "1")
    );
    uploadAreaJSON.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadAreaJSON.style.opacity = "1";
      if (e.dataTransfer?.files) handleFilesJSON(e.dataTransfer.files);
    });
  }

  // ensure initial visibility correct
  updateClearButtonsVisibility();
}

// initialize when DOM ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initUploadModule);
} else {
  initUploadModule();
}

// --- CHANGED: renderPreviewsVideo now uses a 16:9 wrapper so previews keep 16:9 aspect ratio ---
function renderPreviewsVideo() {
  // revoke any previously created object URLs that are no longer used
  if (objectUrlsVideo.length) {
    for (let u of objectUrlsVideo) {
      try {
        if (u) URL.revokeObjectURL(u);
      } catch (e) {
        /* ignore revoke errors */
      }
    }
  }
  objectUrlsVideo = [];

  if (!previewGridVideo) return;
  previewGridVideo.innerHTML = "";
  selectedFilesVideo.forEach((file, idx) => {
    const item = document.createElement("div");
    item.className = "preview-item";

    // video element
    const video = document.createElement("video");
    video.className = "preview-video";
    video.controls = true;
    video.preload = "metadata";

    // create and store object URL
    const blobUrl = URL.createObjectURL(file);
    objectUrlsVideo[idx] = blobUrl;
    video.src = blobUrl;

    // Wrap the video in a responsive 16:9 container
    const wrapper = document.createElement("div");
    // 16:9 padding-top = 9/16 * 100% = 56.25%
    wrapper.style.position = "relative";
    wrapper.style.width = "100%";
    wrapper.style.paddingTop = "56.25%";
    wrapper.style.overflow = "hidden";
    // video fills the wrapper
    video.style.position = "absolute";
    video.style.top = "0";
    video.style.left = "0";
    video.style.width = "100%";
    video.style.height = "100%";
    video.style.objectFit = "contain"; // use "cover" if you prefer crop-fill
    wrapper.appendChild(video);

    // try to load metadata to ensure preview shows; log errors for debugging
    video.addEventListener("error", (ev) => {
      console.error("Video preview error for file:", file.name, ev);
    });
    video.addEventListener("loadedmetadata", () => {
      // no-op, kept for robustness
    });

    const meta = document.createElement("div");
    meta.className = "file-meta";
    meta.innerHTML = `<span title="${file.name}">${
      file.name.length > 30 ? file.name.slice(0, 27) + "..." : file.name
    }</span>
								  <span>${(file.size / 1024 / 1024).toFixed(2)} MB</span>`;

    const actions = document.createElement("div");
    actions.style.marginTop = "8px";
    actions.innerHTML = `<button class="btn btn-outline" onclick="removeFileVideo(${idx})"><i class="fas fa-trash"></i> Hapus</button>`;

    // append wrapper (containing video) instead of raw video
    item.appendChild(wrapper);
    item.appendChild(meta);
    item.appendChild(actions);
    previewGridVideo.appendChild(item);
  });

  if (statusAreaVideo) {
    resetStatus(
      statusAreaVideo,
      `${selectedFilesVideo.length} file siap diproses.`
    );
  }
}

// Hapus file: request server to delete file (if uploaded)
async function removeFileVideo(index) {
  const serverName = uploadedNames[index];
  if (serverName) {
    try {
      await fetch(DELETE_ENDPOINT, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: serverName }),
      });
    } catch (e) {
      console.warn("Gagal menghapus file di server", e);
    }
  }

  // revoke object URL if present
  try {
    if (objectUrlsVideo[index]) {
      URL.revokeObjectURL(objectUrlsVideo[index]);
    }
  } catch (e) {
    /* ignore */
  }

  selectedFilesVideo.splice(index, 1);
  uploadedUrls.splice(index, 1);
  uploadedNames.splice(index, 1);
  objectUrlsVideo.splice(index, 1);
  renderPreviewsVideo();

  // update clear button visibility
  updateClearButtonsVisibility();
}

// Hapus semua: delete all uploaded files on server (best-effort)
async function clearAllVideo() {
  for (const n of uploadedNames) {
    if (!n) continue;
    try {
      await fetch(DELETE_ENDPOINT, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: n }),
      });
    } catch (e) {
      console.warn("Gagal menghapus di server", n, e);
    }
  }

  // revoke any blob URLs
  try {
    for (let u of objectUrlsVideo) {
      if (u) URL.revokeObjectURL(u);
    }
  } catch (e) {
    /* ignore */
  }
  selectedFilesVideo = [];
  uploadedUrls = [];
  uploadedNames = [];
  objectUrlsVideo = [];
  if (previewGridVideo) previewGridVideo.innerHTML = "";
  if (statusAreaVideo) resetStatus(statusAreaVideo, "Tidak ada file dipilih.");
  if (participantNameInput) participantNameInput.value = "";

  // update clear button visibility
  updateClearButtonsVisibility();
}

// Membaca file menjadi Base64 (Data URL) tanpa header data:*/*;base64,
function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // hasil: data:video/mp4;base64,AAAA...
      const result = reader.result || "";
      const base64 = String(result).split(",")[1] || "";
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Ganti buildPayloadVideo: upload file ke cloud terlebih dahulu dan gunakan URL di payload
async function buildPayloadVideo() {
  if (!selectedFilesVideo.length) {
    alert("Pilih minimal satu file video.");
    throw new Error("No files");
  }
  const name = participantNameInput.value.trim();
  if (!name) {
    alert("Nama Peserta wajib diisi.");
    throw new Error("No name");
  }
  // ensure not styled as "ready" while building
  resetStatus(
    statusAreaVideo,
    "Mengunggah file ke server (jika perlu) dan membangun payload..."
  );

  const interviews = [];
  for (let i = 0; i < selectedFilesVideo.length; i++) {
    const f = selectedFilesVideo[i];
    resetStatus(
      statusAreaVideo,
      `Memproses file ${i + 1} / ${selectedFilesVideo.length} ...`
    );

    let url = uploadedUrls[i] || null;
    if (!url) {
      // coba upload otomatis jika belum diunggah
      url = await uploadOne(i);
    }

    if (url) {
      interviews.push({
        positionId: i + 1,
        isVideoExist: true,
        recordedVideoUrl: url,
      });
    } else {
      interviews.push({
        positionId: i + 1,
        isVideoExist: false,
        recordedVideoUrl: null,
      });
    }
  }

  const payload = {
    success: true,
    data: {
      candidate: { name: name },
      reviewChecklists: {
        project: [],
        interviews: interviews,
      },
    },
  };

  // highlight ready state in green
  setStatusReady(statusAreaVideo, "Payload video siap.");
  return payload;
}

// Kirim Video - upload semua video + buat JSON di server + proses transcription
async function buildAndSendVideo() {
  // Prevent duplicate submission
  const now = Date.now();
  if (isSubmittingVideo) {
    console.warn("⚠️ Submission already in progress. Ignoring duplicate call.");
    return;
  }

  // Debounce
  if (now - lastSubmitTime < SUBMIT_DEBOUNCE_MS) {
    console.warn(
      `⚠️ Please wait ${SUBMIT_DEBOUNCE_MS / 1000} seconds between submissions.`
    );
    alert(
      `Mohon tunggu ${SUBMIT_DEBOUNCE_MS / 1000} detik sebelum mengirim ulang.`
    );
    return;
  }

  // Validasi
  if (!selectedFilesVideo.length) {
    alert("Pilih minimal satu file video.");
    return;
  }
  const name = participantNameInput.value.trim();
  if (!name) {
    alert("Nama Peserta wajib diisi.");
    return;
  }

  isSubmittingVideo = true;
  lastSubmitTime = now;

  try {
    showLoading("Uploading videos and processing...");
    resetStatus(statusAreaVideo, "Mengunggah video ke server...");

    // Upload semua video dan kirim ke endpoint /upload yang akan:
    // 1. Upload semua video
    // 2. Build JSON otomatis
    // 3. Proses transcription
    // 4. Return hasil

    const formData = new FormData();
    formData.append("candidate_name", name);

    // Append semua video files
    selectedFilesVideo.forEach((file, idx) => {
      formData.append("videos", file);
    });

    console.log(
      `🔵 Uploading ${selectedFilesVideo.length} video(s) to server...`
    );

    const res = await fetch(VIDEO_ENDPOINT, {
      method: "POST",
      body: formData,
      // IMPORTANT: Jangan set Content-Type header, biarkan browser set otomatis untuk multipart/form-data
    });

    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      alert("Gagal mengirim data ke server: " + text);
      hideLoading();
      isSubmittingVideo = false;
      return;
    }

    const result = await res.json().catch(() => null);
    hideLoading();

    console.log("✅ Server response:", result);

    statusAreaVideo.textContent = "Selesai diproses.";

    // Jika server menyertakan properti redirect, lakukan navigasi
    if (result && result.redirect) {
      let redirectUrl = result.redirect;
      if (redirectUrl && redirectUrl.startsWith("/")) {
        const origin =
          window.location && window.location.origin
            ? window.location.origin
            : getBaseUrl();
        redirectUrl = origin.replace(/\/+$/, "") + redirectUrl;
      }

      // Reset flag sebelum redirect
      isSubmittingVideo = false;
      window.location.href = redirectUrl;
      return;
    }

    alert("Server processed videos. Result: " + JSON.stringify(result));
    isSubmittingVideo = false;
  } catch (err) {
    console.error(err);
    hideLoading();
    statusAreaVideo.textContent = "Terjadi kesalahan saat mengirim.";
    isSubmittingVideo = false;
  }
}

// Download: build payload locally (no server) and force download
async function downloadPayloadVideo() {
  try {
    const payload = await buildPayloadVideo();
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `upload_video_payload_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    /* ditangani di buildPayloadVideo */
  }
}

/* ============================
   JSON upload variables
   ============================ */
// --- CHANGED: declare vars only; bindings are done in initUploadModule to avoid duplicate listeners ---
let fileInputJSON;
let previewGridJSON;
let statusAreaJSON;
let selectedFilesJSON = [];

// Note: initUploadModule already locates fileInputJSON / previewGridJSON / statusAreaJSON
// and attaches listeners. Removing the top-level getElementById/addEventListener prevents
// the same handler from being invoked twice (one from global scope and one from initUploadModule).

function handleFilesJSON(fileList) {
  // filter hanya file JSON
  const files = Array.from(fileList).filter(
    (f) =>
      f.name.toLowerCase().endsWith(".json") || f.type === "application/json"
  );
  if (files.length === 0) {
    alert("Tidak ada file JSON terdeteksi.");
    return;
  }
  if (files.length > 1) {
    alert("Hanya satu file JSON diperbolehkan. Menggunakan file pertama.");
  }
  // selalu replace selection sebelumnya sehingga hanya ada maksimal 1 file
  selectedFilesJSON = [files[0]];
  renderPreviewsJSON();
}

// REPLACED: renderPreviewsJSON -> async, reads file content and shows pretty-printed, scrollable preview
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
    }</span>
                                  <span>${(file.size / 1024).toFixed(
                                    1
                                  )} KB</span>`;

    const actions = document.createElement("div");
    actions.style.marginTop = "8px";
    actions.innerHTML = `<button class="btn btn-outline" onclick="removeFileJSON(${idx})"><i class="fas fa-trash"></i> Hapus</button>`;

    item.appendChild(meta);

    // read file content and attempt pretty print
    try {
      const txt = await file.text();
      let pretty = txt;
      try {
        const parsed = JSON.parse(txt);
        pretty = JSON.stringify(parsed, null, 2);
      } catch (e) {
        // not valid JSON -> show raw text as-is
      }

      // create a scrollable <pre> for pretty output
      const pre = document.createElement("pre");
      pre.className = "json-preview";
      pre.textContent = pretty;
      // inline styles to ensure scrollable and readable even without external CSS
      pre.style.maxHeight = "240px";
      pre.style.overflow = "auto";
      pre.style.background = "#f7f7f7";
      pre.style.padding = "8px";
      pre.style.borderRadius = "4px";
      pre.style.border = "1px solid #e6e6e6";
      pre.style.fontFamily =
        "Menlo, Monaco, Consolas, 'Liberation Mono', monospace";
      pre.style.fontSize = "12px";
      pre.style.whiteSpace = "pre-wrap"; // wrap long lines but preserve indentation
      pre.style.marginTop = "8px";

      item.appendChild(pre);
    } catch (err) {
      console.error("Gagal membaca file untuk preview", file.name, err);
      const errNote = document.createElement("div");
      errNote.className = "muted";
      errNote.textContent = "Preview gagal dibuat.";
      item.appendChild(errNote);
    }

    item.appendChild(actions);
    previewGridJSON.appendChild(item);
  }

  if (statusAreaJSON) {
    resetStatus(
      statusAreaJSON,
      `${selectedFilesJSON.length} file JSON siap diproses.`
    );
  }

  // update clear button visibility after rendering
  updateClearButtonsVisibility();
}

function removeFileJSON(index) {
  selectedFilesJSON.splice(index, 1);
  renderPreviewsJSON();
  // update clear button visibility
  updateClearButtonsVisibility();
}

function clearAllJSON() {
  selectedFilesJSON = [];
  previewGridJSON.innerHTML = "";
  resetStatus(statusAreaJSON, "Belum ada file JSON dipilih.");

  // update clear button visibility
  updateClearButtonsVisibility();
}

async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsText(file, "utf-8");
  });
}

async function buildPayloadFromJSONFiles() {
  if (!selectedFilesJSON.length) {
    alert("Pilih minimal satu file JSON.");
    throw new Error("No json");
  }
  resetStatus(statusAreaJSON, "Membaca file JSON...");
  const filesPayload = [];
  for (let i = 0; i < selectedFilesJSON.length; i++) {
    const f = selectedFilesJSON[i];
    try {
      const txt = await readFileAsText(f);
      let parsed = null;
      try {
        parsed = JSON.parse(txt);
      } catch (e) {
        parsed = txt;
      }
      filesPayload.push({ name: f.name, size: f.size, content: parsed });
      resetStatus(
        statusAreaJSON,
        `Mempersiapkan file ${i + 1} / ${selectedFilesJSON.length} ...`
      );
    } catch (err) {
      console.error("Error membaca JSON", f.name, err);
      alert("Gagal membaca file: " + f.name);
    }
  }
  const payload = {
    meta: {
      createdAt: new Date().toISOString(),
      fileCount: filesPayload.length,
      uploader: "web-client",
    },
    files: filesPayload,
  };
  setStatusReady(statusAreaJSON, "Payload JSON siap.");
  return payload;
}

// NEW: flag to prevent duplicate JSON submission
let isSubmittingJSON = false;
let lastSubmitTimeJSON = 0;

async function buildAndSendJSONFiles() {
  // Prevent duplicate submission
  const now = Date.now();
  if (isSubmittingJSON) {
    console.warn(
      "⚠️ JSON submission already in progress. Ignoring duplicate call."
    );
    return;
  }

  // Debounce
  if (now - lastSubmitTimeJSON < SUBMIT_DEBOUNCE_MS) {
    console.warn(
      `⚠️ Please wait ${SUBMIT_DEBOUNCE_MS / 1000} seconds between submissions.`
    );
    alert(
      `Mohon tunggu ${SUBMIT_DEBOUNCE_MS / 1000} detik sebelum mengirim ulang.`
    );
    return;
  }

  isSubmittingJSON = true;
  lastSubmitTimeJSON = now;

  try {
    const payload = await buildPayloadFromJSONFiles();
    const endpoint = JSON_ENDPOINT; // gunakan konstanta
    statusAreaJSON.textContent = "Mengirim payload JSON ke server...";

    console.log(
      "🔵 Sending JSON payload to server:",
      JSON.stringify(payload, null, 2)
    );

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      statusAreaJSON.textContent = `Gagal mengirim: ${res.status} ${text}`;
      alert("Gagal mengirim data ke server. Periksa endpoint dan CORS.");
      isSubmittingJSON = false;
      return;
    }

    const result = await res.json().catch(() => null);

    console.log("✅ Server response:", result);

    statusAreaJSON.textContent = "Berhasil dikirim.";
    // jika server meminta redirect, ikuti (support relatif)
    if (result && result.redirect) {
      let redirectUrl = result.redirect;
      if (redirectUrl && redirectUrl.startsWith("/")) {
        const origin =
          window.location && window.location.origin
            ? window.location.origin
            : getBaseUrl();
        redirectUrl = origin.replace(/\/+$/, "") + redirectUrl;
      }

      // Reset flag sebelum redirect
      isSubmittingJSON = false;

      window.location.href = redirectUrl;
      return;
    }
    alert(
      "Payload JSON berhasil dikirim. Server merespon: " +
        (result ? JSON.stringify(result) : "OK")
    );
    isSubmittingJSON = false;
  } catch (err) {
    console.error(err);
    if (err.message !== "No json")
      statusAreaJSON.textContent = "Terjadi kesalahan saat mengirim.";
    isSubmittingJSON = false;
  }
}

// NEW: file system handles (global)
let uploadDirHandle = null; // root folder chosen by user (optional)
let uploadsDirHandle = null; // 'uploads' subfolder handle (optional)

// Save a file into uploadsDirHandle with given filename; returns saved name
async function saveFileToUploads(file, filename) {
  // If uploadsDirHandle not set, ask user to pick a folder and create 'uploads'
  if (!uploadsDirHandle) {
    try {
      uploadDirHandle = await window.showDirectoryPicker();
      uploadsDirHandle = await uploadDirHandle.getDirectoryHandle("uploads", {
        create: true,
      });
      statusAreaVideo.textContent =
        "Folder upload dipilih: " + (uploadsDirHandle.name || "uploads");
    } catch (err) {
      console.error("Directory pick cancelled or not supported", err);
      throw new Error(
        "Upload folder belum dipilih. Akses direktori dibutuhkan."
      );
    }
  }

  // sanitize filename minimally
  const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
  const fh = await uploadsDirHandle.getFileHandle(safeName, {
    create: true,
  });
  const writable = await fh.createWritable();
  await writable.write(file);
  await writable.close();
  return safeName;
}

// NEW: clear buttons refs
let clearAllVideoBtn;
let clearAllJSONBtn;

// NEW: show/hide clear buttons based on whether there are selected files
function updateClearButtonsVisibility() {
  try {
    if (clearAllVideoBtn) {
      clearAllVideoBtn.style.display =
        selectedFilesVideo && selectedFilesVideo.length ? "" : "none";
    }
    if (clearAllJSONBtn) {
      clearAllJSONBtn.style.display =
        selectedFilesJSON && selectedFilesJSON.length ? "" : "none";
    }
  } catch (e) {
    // silent
  }
}
