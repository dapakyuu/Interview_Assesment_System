let historySupabaseClient = null;

function getHistoryClient() {
  if (historySupabaseClient) return historySupabaseClient;

  const config = window.SUPABASE_CONFIG || {};
  if (!window.supabase?.createClient || !config.url || !config.anonKey) {
    return null;
  }

  historySupabaseClient = window.supabase.createClient(
    config.url,
    config.anonKey,
    {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    },
  );

  return historySupabaseClient;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("id-ID", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

async function renderHistory() {
  const statusEl = document.getElementById("historyStatus");
  const bodyEl = document.getElementById("historyTableBody");
  const emptyEl = document.getElementById("historyEmpty");

  if (!statusEl || !bodyEl || !emptyEl) return;

  const client = getHistoryClient();
  if (!client) {
    statusEl.textContent = "Supabase client belum siap.";
    return;
  }

  const { data: sessionData, error: sessionError } =
    await client.auth.getSession();
  if (sessionError || !sessionData?.session?.user) {
    window.location.href = "login.html?next=history.html";
    return;
  }

  const userId = sessionData.session.user.id;
  const { data, error } = await client
    .from("history")
    .select("session_id, created_at")
    .eq("user_id", userId)
    .order("created_at", { ascending: false });

  if (error) {
    statusEl.textContent = `Gagal memuat history: ${error.message}`;
    bodyEl.innerHTML = "";
    emptyEl.style.display = "none";
    return;
  }

  statusEl.textContent = `Ditemukan ${data.length} session.`;
  bodyEl.innerHTML = "";

  if (!data.length) {
    emptyEl.style.display = "block";
    return;
  }

  emptyEl.style.display = "none";

  data.forEach((row, index) => {
    const dashboardUrl = `Halaman_dasboard.html?session=${encodeURIComponent(row.session_id || "")}`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>${row.session_id || "-"}</td>
      <td>
        <a class="history-link-btn" href="${dashboardUrl}">
          <i class="fas fa-arrow-up-right-from-square"></i>
          Dashboard
        </a>
      </td>
      <td>${formatDate(row.created_at)}</td>
    `;
    bodyEl.appendChild(tr);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderHistory().catch((error) => {
    const statusEl = document.getElementById("historyStatus");
    if (statusEl) {
      statusEl.textContent = `Gagal memuat history: ${error.message}`;
    }
  });
});
