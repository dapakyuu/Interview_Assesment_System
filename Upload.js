/* ============================
   Video upload variables
   ============================ */
const fileInputVideo = document.getElementById("fileInputVideo");
const previewGridVideo = document.getElementById("previewGridVideo");
const statusAreaVideo = document.getElementById("statusAreaVideo");
const participantNameInput = document.getElementById("participantName");

let selectedFilesVideo = [];

fileInputVideo.addEventListener("change", (e) =>
  handleFilesVideo(e.target.files)
);
const uploadAreaVideo = document.getElementById("uploadAreaVideo");
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

// Replace previous SERVER_MODE/FS fallback logic with simple server upload endpoints
const FILE_UPLOAD_ENDPOINT = "http://127.0.0.1:8888/upload_file";
const DELETE_ENDPOINT = "http://127.0.0.1:8888/delete_file";
const VIDEO_ENDPOINT = "http://127.0.0.1:8888/upload";
const JSON_ENDPOINT = VIDEO_ENDPOINT;

// state
let uploadedUrls = [];
let uploadedNames = [];

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

// Upload single file: always POST to server endpoint
async function uploadOne(index) {
  const file = selectedFilesVideo[index];
  if (!file) return null;
  statusAreaVideo.textContent = `Mengunggah file ${index + 1} ke server...`;
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(FILE_UPLOAD_ENDPOINT, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) throw new Error("Upload failed: " + res.status);
    const j = await res.json();
    const url = j?.url || null;
    const name = j?.name || null;
    if (url) uploadedUrls[index] = url;
    if (name) uploadedNames[index] = name;
    renderPreviewsVideo();
    statusAreaVideo.textContent = `File ${index + 1} terunggah ke server.`;
    return url;
  } catch (err) {
    console.error("Upload to server error", err);
    statusAreaVideo.textContent = `Gagal unggah file ${index + 1} ke server.`;
    return null;
  }
}

// Saat user drop / pilih file: langsung tambah dan upload tiap file
function handleFilesVideo(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("video/"));
  if (files.length === 0) {
    alert("Tidak ada file video terdeteksi.");
    return;
  }

  // append files and ensure arrays alignment
  const startIndex = selectedFilesVideo.length;
  selectedFilesVideo = selectedFilesVideo.concat(files);
  // extend storage arrays
  for (let i = 0; i < files.length; i++) {
    uploadedUrls[startIndex + i] = null;
    uploadedNames[startIndex + i] = null;
  }
  renderPreviewsVideo();

  // Upload each new file automatically (do not block UI)
  files.forEach((_, idx) => {
    const globalIndex = startIndex + idx;
    uploadOne(globalIndex);
  });
}

// Perbarui renderPreviewsVideo: hanya preview + tombol hapus
function renderPreviewsVideo() {
  previewGridVideo.innerHTML = "";
  selectedFilesVideo.forEach((file, idx) => {
    const item = document.createElement("div");
    item.className = "preview-item";

    const video = document.createElement("video");
    video.className = "preview-video";
    video.controls = true;
    video.preload = "metadata";
    video.src = URL.createObjectURL(file);

    const meta = document.createElement("div");
    meta.className = "file-meta";
    meta.innerHTML = `<span title="${file.name}">${
      file.name.length > 30 ? file.name.slice(0, 27) + "..." : file.name
    }</span>
							  <span>${(file.size / 1024 / 1024).toFixed(2)} MB</span>`;

    const actions = document.createElement("div");
    actions.style.marginTop = "8px";

    // Hanya tombol Hapus — tidak menampilkan tombol Upload atau link hasil upload
    actions.innerHTML = `<button class="btn btn-outline" onclick="removeFileVideo(${idx})"><i class="fas fa-trash"></i> Hapus</button>`;

    item.appendChild(video);
    item.appendChild(meta);
    item.appendChild(actions);
    previewGridVideo.appendChild(item);
  });

  statusAreaVideo.textContent = `${selectedFilesVideo.length} file siap diproses.`;
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
  selectedFilesVideo.splice(index, 1);
  uploadedUrls.splice(index, 1);
  uploadedNames.splice(index, 1);
  renderPreviewsVideo();
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
  selectedFilesVideo = [];
  uploadedUrls = [];
  uploadedNames = [];
  previewGridVideo.innerHTML = "";
  statusAreaVideo.textContent = "Tidak ada file dipilih.";
  participantNameInput.value = "";
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
  statusAreaVideo.textContent =
    "Mengunggah file ke server (jika perlu) dan membangun payload...";

  const interviews = [];
  for (let i = 0; i < selectedFilesVideo.length; i++) {
    const f = selectedFilesVideo[i];
    statusAreaVideo.textContent = `Memproses file ${i + 1} / ${
      selectedFilesVideo.length
    } ...`;

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

  statusAreaVideo.textContent = "Payload video siap.";
  return payload;
}

// Kirim Video: build payload, show loading, POST to /upload, wait for processing result
async function buildAndSendVideo() {
  try {
    const payload = await buildPayloadVideo();
    showLoading("Sending payload and processing on server...");
    const res = await fetch(VIDEO_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      alert("Gagal mengirim data ke server: " + text);
      hideLoading();
      return;
    }
    const result = await res.json().catch(() => null);
    hideLoading();
    statusAreaVideo.textContent = "Selesai diproses.";
    // jika server menyertakan properti redirect, lakukan navigasi
    if (result && result.redirect) {
      // support relatif / absolut; jika relatif, ubah ke origin
      let redirectUrl = result.redirect;
      if (redirectUrl && redirectUrl.startsWith("/")) {
        const origin =
          window.location && window.location.origin
            ? window.location.origin
            : getBaseUrl();
        redirectUrl = origin.replace(/\/+$/, "") + redirectUrl;
      }
      window.location.href = redirectUrl;
      return;
    }
    alert("Server processed payload. Result: " + JSON.stringify(result));
  } catch (err) {
    console.error(err);
    hideLoading();
    statusAreaVideo.textContent = "Terjadi kesalahan saat mengirim.";
    alert("Error saat mengirim/pemrosesan: " + err.message);
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
const fileInputJSON = document.getElementById("fileInputJSON");
const previewGridJSON = document.getElementById("previewGridJSON");
const statusAreaJSON = document.getElementById("statusAreaJSON");
let selectedFilesJSON = [];

fileInputJSON.addEventListener("change", (e) =>
  handleFilesJSON(e.target.files)
);
const uploadAreaJSON = document.getElementById("uploadAreaJSON");
uploadAreaJSON.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadAreaJSON.style.opacity = "0.8";
});
uploadAreaJSON.addEventListener("dragleave", () => {
  uploadAreaJSON.style.opacity = "1";
});
uploadAreaJSON.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadAreaJSON.style.opacity = "1";
  if (e.dataTransfer?.files) handleFilesJSON(e.dataTransfer.files);
});

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

function renderPreviewsJSON() {
  previewGridJSON.innerHTML = "";
  selectedFilesJSON.forEach((file, idx) => {
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
    item.appendChild(actions);
    previewGridJSON.appendChild(item);
  });

  statusAreaJSON.textContent = `${selectedFilesJSON.length} file JSON siap diproses.`;
}

function removeFileJSON(index) {
  selectedFilesJSON.splice(index, 1);
  renderPreviewsJSON();
}

function clearAllJSON() {
  selectedFilesJSON = [];
  previewGridJSON.innerHTML = "";
  statusAreaJSON.textContent = "Belum ada file JSON dipilih.";
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
  statusAreaJSON.textContent = "Membaca file JSON...";
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
      statusAreaJSON.textContent = `Mempersiapkan file ${i + 1} / ${
        selectedFilesJSON.length
      } ...`;
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
  statusAreaJSON.textContent = "Payload JSON siap.";
  return payload;
}

async function buildAndSendJSONFiles() {
  try {
    const payload = await buildPayloadFromJSONFiles();
    const endpoint = JSON_ENDPOINT; // gunakan konstanta
    statusAreaJSON.textContent = "Mengirim payload JSON ke server...";
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      statusAreaJSON.textContent = `Gagal mengirim: ${res.status} ${text}`;
      alert("Gagal mengirim data ke server. Periksa endpoint dan CORS.");
      return;
    }

    const result = await res.json().catch(() => null);
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
      window.location.href = redirectUrl;
      return;
    }
    alert(
      "Payload JSON berhasil dikirim. Server merespon: " +
        (result ? JSON.stringify(result) : "OK")
    );
  } catch (err) {
    console.error(err);
    if (err.message !== "No json")
      statusAreaJSON.textContent = "Terjadi kesalahan saat mengirim.";
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
