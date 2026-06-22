const LEGACY_AUTH_KEYS = [
  "ai_interview_registered_users",
  "ai_interview_auth_session",
];
const NEXT_URL_KEY = "auth_next_url";

let supabaseClientInstance = null;

function clearLegacyAuthStorage() {
  LEGACY_AUTH_KEYS.forEach((key) => localStorage.removeItem(key));
}

function getSupabaseClient() {
  if (supabaseClientInstance) return supabaseClientInstance;

  const config = window.SUPABASE_CONFIG || {};
  const createClient = window.supabase?.createClient;

  if (!config.url || !config.anonKey || typeof createClient !== "function") {
    return null;
  }

  supabaseClientInstance = createClient(config.url, config.anonKey, {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: true,
    },
  });

  return supabaseClientInstance;
}

function getClientOrThrow() {
  const client = getSupabaseClient();
  if (!client) {
    throw new Error(
      "Supabase client belum siap. Pastikan supabase-js dan supabase-config.js sudah dimuat.",
    );
  }
  return client;
}

function setMessage(element, text, type) {
  if (!element) return;
  element.textContent = text;
  element.className = `auth-message show ${type}`;
}

function clearMessage(element) {
  if (!element) return;
  element.textContent = "";
  element.className = "auth-message";
}

function normalizeEmail(value) {
  return String(value || "")
    .trim()
    .toLowerCase();
}

function getNextUrl(defaultUrl = "Upload.html") {
  const params = new URLSearchParams(window.location.search);
  const next = params.get("next") || localStorage.getItem(NEXT_URL_KEY);
  if (!next) return defaultUrl;

  const safeNext = next.trim();
  if (
    !safeNext ||
    safeNext.includes("://") ||
    safeNext.startsWith("javascript:")
  ) {
    return defaultUrl;
  }

  return safeNext.replace(/^\/+/, "");
}

function setNextUrl(nextUrl) {
  if (nextUrl) {
    localStorage.setItem(NEXT_URL_KEY, nextUrl);
  }
}

function clearNextUrl() {
  localStorage.removeItem(NEXT_URL_KEY);
}

async function getSession() {
  const client = getSupabaseClient();
  if (!client) return null;

  const { data, error } = await client.auth.getSession();
  if (error) {
    console.error("Failed to load session:", error);
    return null;
  }

  return data?.session || null;
}

function getDisplayName(user) {
  return user?.user_metadata?.full_name || user?.user_metadata?.name || "User";
}

function getUserInitials(user) {
  const displayName = getDisplayName(user);
  const parts = String(displayName).trim().split(/\s+/).filter(Boolean);

  if (!parts.length) return "U";

  return parts
    .slice(0, 2)
    .map((part) => part.charAt(0).toUpperCase())
    .join("");
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function syncProfile(user, fallbackName = "") {
  const client = getSupabaseClient();
  if (!client || !user) return;

  const profile = {
    id: user.id,
    full_name:
      fallbackName ||
      user.user_metadata?.full_name ||
      user.user_metadata?.name ||
      user.email?.split("@")[0] ||
      "User",
    email: user.email || "",
    role: user.user_metadata?.role || "candidate",
    updated_at: new Date().toISOString(),
  };

  const { data, error } = await client.from("profiles").upsert(profile, {
    onConflict: "id",
  });

  console.log("syncProfile upsert result:", { data, error });

  if (error) {
    console.warn("Profile sync skipped:", error.message);
  }
}

async function redirectIfAuthenticated() {
  const session = await getSession();
  if (!session?.user) return false;
  window.location.href = getNextUrl();
  return true;
}

function redirectToLogin() {
  const next = encodeURIComponent(getNextUrl());
  window.location.href = `login.html?next=${next}`;
}

async function handleLoginSubmit(event) {
  event.preventDefault();

  const messageEl = document.getElementById("authMessage");
  clearMessage(messageEl);

  const email = normalizeEmail(document.getElementById("loginEmail")?.value);
  const password = document.getElementById("loginPassword")?.value || "";

  if (!email || !password) {
    setMessage(messageEl, "Email dan password wajib diisi.", "error");
    return;
  }

  try {
    const client = getClientOrThrow();
    const { data, error } = await client.auth.signInWithPassword({
      email,
      password,
    });

    console.log("signInWithPassword result:", { data, error });

    if (error) throw error;

    if (!data?.session) {
      setMessage(
        messageEl,
        "Login berhasil, tetapi sesi belum tersedia. Silakan refresh halaman.",
        "error",
      );
      return;
    }

    await syncProfile(data.user);
    clearNextUrl();
    window.location.href = getNextUrl();
  } catch (error) {
    setMessage(messageEl, error.message || "Login gagal.", "error");
  }
}

async function handleSignupSubmit(event) {
  event.preventDefault();

  const messageEl = document.getElementById("authMessage");
  clearMessage(messageEl);

  const name = document.getElementById("signupName")?.value.trim() || "";
  const email = normalizeEmail(document.getElementById("signupEmail")?.value);
  const password = document.getElementById("signupPassword")?.value || "";
  const confirmPassword =
    document.getElementById("signupConfirmPassword")?.value || "";

  if (!name || !email || !password || !confirmPassword) {
    setMessage(messageEl, "Semua field wajib diisi.", "error");
    return;
  }

  if (password.length < 6) {
    setMessage(messageEl, "Password minimal terdiri dari 6 karakter.", "error");
    return;
  }

  if (password !== confirmPassword) {
    setMessage(messageEl, "Password dan konfirmasi tidak sama.", "error");
    return;
  }

  try {
    const client = getClientOrThrow();
    const { data, error } = await client.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: name,
          role: "candidate",
        },
      },
    });

    console.log("signUp result:", { data, error });

    if (error) throw error;

    if (data?.user) {
      await syncProfile(data.user, name);
    }

    if (data?.session) {
      clearNextUrl();
      setMessage(
        messageEl,
        "Pendaftaran berhasil. Anda akan diarahkan ke halaman upload.",
        "success",
      );
      window.location.href = getNextUrl();
      return;
    }

    window.alert(
      "Akun berhasil dibuat. Silakan cek email untuk verifikasi. Setelah menekan OK, Anda akan diarahkan ke halaman login.",
    );
    window.location.href = `login.html?registered=1&email=${encodeURIComponent(email)}&next=${encodeURIComponent(getNextUrl())}`;
  } catch (error) {
    setMessage(messageEl, error.message || "Pendaftaran gagal.", "error");
  }
}

async function renderAuthState() {
  const authArea = document.getElementById("authArea");
  if (!authArea) return;

  const pageType = document.body?.dataset?.authPage || "";
  const session = await getSession();

  if (!session?.user) {
    const loginNext = pageType === "history" ? "history.html" : "Upload.html";
    authArea.innerHTML = `
      <a href="login.html?next=${encodeURIComponent(loginNext)}" class="btn btn-primary" style="text-decoration: none">
        <i class="fas fa-right-to-bracket"></i> Login
      </a>
    `;
    return;
  }

  const displayName = escapeHtml(getDisplayName(session.user));
  const initials = escapeHtml(getUserInitials(session.user));
  const navButton =
    pageType === "history"
      ? `<a href="Upload.html" class="btn btn-outline"><i class="fas fa-arrow-left"></i> Upload</a>`
      : `<a href="history.html" class="btn btn-outline"><i class="fas fa-clock-rotate-left"></i> History</a>`;

  const userChip =
    pageType === "dashboard"
      ? ""
      : `
      <div class="auth-user-chip">
        <div class="auth-user-avatar">${initials}</div>
        <div class="auth-user-meta">
          <strong>${displayName}</strong>
        </div>
      </div>
    `;

  authArea.innerHTML = `
    ${userChip}
    ${navButton}
    <button type="button" class="btn btn-outline" id="logoutButton">
      <i class="fas fa-right-from-bracket"></i> Logout
    </button>
  `;

  document
    .getElementById("logoutButton")
    ?.addEventListener("click", async () => {
      try {
        const client = getClientOrThrow();
        await client.auth.signOut();
      } catch (error) {
        console.error("Logout error:", error);
      } finally {
        localStorage.removeItem("video_processing_session");
        redirectToLogin();
      }
    });
}

async function guardAuthPage() {
  const pageType = document.body?.dataset?.authPage || "";

  if (pageType === "login") {
    if (await redirectIfAuthenticated()) return;

    const params = new URLSearchParams(window.location.search);
    const registered = params.get("registered");
    const email = params.get("email");
    const loginEmail = document.getElementById("loginEmail");
    const messageEl = document.getElementById("authMessage");

    if (registered && messageEl) {
      setMessage(
        messageEl,
        "Akun berhasil dibuat. Silakan masuk menggunakan email dan kata sandi Anda.",
        "success",
      );
    }

    if (email && loginEmail) {
      loginEmail.value = email;
    }

    document
      .getElementById("loginForm")
      ?.addEventListener("submit", handleLoginSubmit);
    return;
  }

  if (pageType === "signup") {
    if (await redirectIfAuthenticated()) return;
    document
      .getElementById("signupForm")
      ?.addEventListener("submit", handleSignupSubmit);
    return;
  }

  if (document.body?.dataset?.requireAuth === "true") {
    const session = await getSession();
    if (!session?.user) {
      redirectToLogin();
      return;
    }
  }

  const authArea = document.getElementById("authArea");
  if (authArea) {
    await renderAuthState();

    const client = getSupabaseClient();
    client?.auth.onAuthStateChange((event) => {
      if (event === "SIGNED_OUT") {
        localStorage.removeItem("video_processing_session");
        redirectToLogin();
      }
    });
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  clearLegacyAuthStorage();

  console.log("Auth init - supabase client ready:", !!getSupabaseClient());

  const pageType = document.body?.dataset?.authPage || "";
  const messageEl = document.getElementById("authMessage");

  if (!getSupabaseClient()) {
    if (messageEl && (pageType === "login" || pageType === "signup")) {
      setMessage(
        messageEl,
        "Supabase belum terkonfigurasi. Periksa supabase-config.js dan supabase-js CDN.",
        "error",
      );
    }

    if (document.body?.dataset?.requireAuth === "true") {
      redirectToLogin();
    }

    return;
  }

  await guardAuthPage();
});
