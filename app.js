/* ═══════════════════════════════════════════════════════════════════════════
   app.js — Library AI Agent frontend
   ─────────────────────────────────────────────────────────────────────────
   Handles:
     • Dark/light theme toggle
     • Sidebar panel navigation
     • Chat with streaming-style display
     • Book search + pagination
     • Personalised recommendations
     • Trending books panel
     • Reservation / waitlist actions
     • Student profile persistence
     • Book detail modal
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ─── State ────────────────────────────────────────────────────────────────────
const STATE = {
  studentProfile: {
    name: "Student",
    student_id: "GUEST",
    branch: "Computer Science",
    semester: "5th",
    courses: ["Machine Learning", "Data Structures"],
    language: "English",
  },
  currentPage: 1,
  lastQuery: "",
  selectedBookId: null,
};

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// ─── Bootstrap instances ──────────────────────────────────────────────────────
let profileModalInst, bookModalInst, toastInst;

// ═══════════════════════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  profileModalInst = new bootstrap.Modal($("profileModal"));
  bookModalInst    = new bootstrap.Modal($("bookModal"));
  toastInst        = new bootstrap.Toast($("appToast"), { delay: 3500 });

  _loadTheme();
  _loadProfile();
  _bindNav();
  _bindChat();
  _bindSearch();
  _bindProfile();
  _bindReservationTabs();

  // Auto-load trending on first render
  _loadTrending();
  _loadRecommendations();
});

// ═══════════════════════════════════════════════════════════════════════════════
//  THEME
// ═══════════════════════════════════════════════════════════════════════════════
function _loadTheme() {
  const saved = localStorage.getItem("lib-theme") || "light";
  _applyTheme(saved);
}

function _applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const icon = $("themeIcon");
  if (icon) {
    icon.className = theme === "dark" ? "bi bi-sun-fill" : "bi bi-moon-fill";
  }
  localStorage.setItem("lib-theme", theme);
}

$("themeToggle")?.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  _applyTheme(current === "dark" ? "light" : "dark");
});

// ═══════════════════════════════════════════════════════════════════════════════
//  SIDEBAR NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════════
function _bindNav() {
  $$(".sidebar-item").forEach(btn => {
    btn.addEventListener("click", () => {
      const panel = btn.dataset.panel;
      _showPanel(panel);

      $$(".sidebar-item").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
}

function _showPanel(name) {
  $$(".panel").forEach(p => p.classList.remove("active"));
  const target = $(`panel-${name}`);
  if (target) {
    target.classList.add("active");
    if (name === "trending")       _loadTrending();
    if (name === "recommendations") _loadRecommendations();
    if (name === "reservations")   _loadMyBooks();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CHAT
// ═══════════════════════════════════════════════════════════════════════════════
function _bindChat() {
  const form  = $("chatForm");
  const input = $("chatInput");

  // Auto-grow textarea
  input?.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });

  // Enter to send (Shift+Enter = newline)
  input?.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form?.requestSubmit();
    }
  });

  form?.addEventListener("submit", async e => {
    e.preventDefault();
    const msg = input.value.trim();
    if (!msg) return;
    input.value = "";
    input.style.height = "auto";
    await _sendChat(msg);
  });

  // Quick chips
  $$(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const msg = chip.dataset.msg;
      if (msg) _sendChat(msg);
    });
  });

  // Clear chat
  $("clearChat")?.addEventListener("click", () => {
    const body = $("chatBody");
    // Keep the welcome bubble only
    while (body.children.length > 1) body.removeChild(body.lastChild);
  });
}

async function _sendChat(message) {
  // Remove welcome chips once a real message is sent
  const welcome = document.querySelector(".chat-welcome");
  if (welcome) welcome.remove();

  _appendMsg("user", message);
  const typingId = _appendTyping();

  const lang = $("chatLang")?.value || "English";
  const profile = { ...STATE.studentProfile, language: lang };

  try {
    const res = await fetch("/api/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message, student_profile: profile }),
    });
    const data = await res.json();
    _removeTyping(typingId);

    if (data.error) {
      _appendMsg("assistant", `⚠️ Error: ${data.error}`);
    } else {
      _appendMsg("assistant", data.response, data.sources || []);
    }
  } catch (err) {
    _removeTyping(typingId);
    _appendMsg("assistant", "⚠️ Network error. Please check your connection and try again.");
  }
}

function _appendMsg(role, text, sources = []) {
  const body = $("chatBody");
  const row  = document.createElement("div");
  row.className = `msg-row ${role}`;

  const initials = role === "user"
    ? (STATE.studentProfile.name[0] || "S").toUpperCase()
    : "G";

  const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  let sourcesHtml = "";
  if (sources.length) {
    const chips = sources.map(s => `<span class="source-chip">${_esc(s)}</span>`).join("");
    sourcesHtml = `<div class="source-chips">${chips}</div>`;
  }

  row.innerHTML = `
    <div class="msg-avatar">${_esc(initials)}</div>
    <div>
      <div class="msg-bubble">${_esc(text)}</div>
      ${sourcesHtml}
      <div class="msg-time">${now}</div>
    </div>
  `;

  body.appendChild(row);
  body.scrollTop = body.scrollHeight;
  return row;
}

function _appendTyping() {
  const body = $("chatBody");
  const id   = "typing-" + Date.now();
  const row  = document.createElement("div");
  row.id        = id;
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="msg-avatar">G</div>
    <div class="msg-bubble" style="padding:.3rem .7rem">
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;
  body.appendChild(row);
  body.scrollTop = body.scrollHeight;
  return id;
}

function _removeTyping(id) {
  const el = $(id);
  if (el) el.remove();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  BOOK SEARCH
// ═══════════════════════════════════════════════════════════════════════════════
function _bindSearch() {
  $("searchBtn")?.addEventListener("click", () => { STATE.currentPage = 1; _runSearch(); });
  $("searchInput")?.addEventListener("keydown", e => {
    if (e.key === "Enter") { STATE.currentPage = 1; _runSearch(); }
  });
  $("resetFilters")?.addEventListener("click", () => {
    $("searchInput").value = "";
    $("filterDept").value  = "";
    $("filterAvail").value = "false";
    $("searchResults").innerHTML = `<div class="empty-state"><i class="bi bi-search"></i><p>Search for books by title, author, subject, or keyword</p></div>`;
    $("searchPagination").classList.add("d-none");
  });
}

async function _runSearch(page = STATE.currentPage) {
  const q       = $("searchInput")?.value.trim() || "";
  const dept    = $("filterDept")?.value || "";
  const avail   = $("filterAvail")?.value || "false";

  STATE.lastQuery = q;
  STATE.currentPage = page;

  const params = new URLSearchParams({ q, department: dept, available_only: avail, page, per_page: 12 });

  $("searchResults").innerHTML = `<div class="skeleton-grid">${"<div class='skeleton-card'></div>".repeat(6)}</div>`;
  $("searchPagination").classList.add("d-none");

  try {
    const res  = await fetch(`/api/books/search?${params}`);
    const data = await res.json();

    if (!data.books || data.books.length === 0) {
      $("searchResults").innerHTML = `<div class="empty-state"><i class="bi bi-book"></i><p>No books found matching "${_esc(q)}"</p></div>`;
      return;
    }

    $("searchResults").innerHTML = "";
    data.books.forEach(b => {
      $("searchResults").appendChild(_buildBookCard(b));
    });

    _renderPagination(data.page, data.total_pages);
  } catch (err) {
    $("searchResults").innerHTML = `<div class="empty-state"><i class="bi bi-exclamation-triangle"></i><p>Failed to load results. Is the server running?</p></div>`;
  }
}

function _renderPagination(current, total) {
  if (total <= 1) { $("searchPagination").classList.add("d-none"); return; }

  $("searchPagination").classList.remove("d-none");
  $("searchPagination").innerHTML = "";

  const prev = _makeBtn("‹ Prev", current <= 1, () => _runSearch(current - 1));
  $("searchPagination").appendChild(prev);

  for (let i = 1; i <= total; i++) {
    const btn = document.createElement("button");
    btn.className = `btn btn-sm ${i === current ? "btn-primary" : "btn-outline-secondary"}`;
    btn.textContent = i;
    btn.addEventListener("click", () => _runSearch(i));
    $("searchPagination").appendChild(btn);
  }

  const next = _makeBtn("Next ›", current >= total, () => _runSearch(current + 1));
  $("searchPagination").appendChild(next);
}

function _makeBtn(label, disabled, onClick) {
  const b = document.createElement("button");
  b.className = "btn btn-sm btn-outline-secondary";
  b.textContent = label;
  b.disabled = disabled;
  b.addEventListener("click", onClick);
  return b;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RECOMMENDATIONS
// ═══════════════════════════════════════════════════════════════════════════════
async function _loadRecommendations() {
  const p = STATE.studentProfile;
  const params = new URLSearchParams({
    branch:   p.branch,
    semester: p.semester,
    courses:  (p.courses || []).join(","),
  });

  // Update profile bar badges
  $("recBranchBadge").textContent = p.branch || "—";
  $("recSemBadge").textContent    = p.semester ? `${p.semester} Sem` : "—";
  const coursesBadges = $("recCoursesBadges");
  coursesBadges.innerHTML = "";
  (p.courses || []).slice(0, 4).forEach(c => {
    const b = document.createElement("span");
    b.className = "badge bg-primary";
    b.textContent = c;
    coursesBadges.appendChild(b);
  });

  $("recsGrid").innerHTML = `<div class="skeleton-grid">${"<div class='skeleton-card'></div>".repeat(6)}</div>`;

  try {
    const res  = await fetch(`/api/recommendations?${params}`);
    const data = await res.json();

    if (!data.recommendations || data.recommendations.length === 0) {
      $("recsGrid").innerHTML = `<div class="empty-state"><i class="bi bi-stars"></i><p>No recommendations yet. Update your profile with courses and branch.</p></div>`;
      return;
    }

    $("recsGrid").innerHTML = "";
    data.recommendations.forEach(b => {
      $("recsGrid").appendChild(_buildBookCard(b, b.match_reason));
    });
  } catch {
    $("recsGrid").innerHTML = `<div class="empty-state"><i class="bi bi-exclamation-triangle"></i><p>Could not load recommendations.</p></div>`;
  }
}

$("refreshRecs")?.addEventListener("click", _loadRecommendations);

// ═══════════════════════════════════════════════════════════════════════════════
//  TRENDING
// ═══════════════════════════════════════════════════════════════════════════════
async function _loadTrending() {
  const dept = $("trendingDeptFilter")?.value || "";
  const params = new URLSearchParams({ department: dept, limit: 8 });

  $("trendingGrid").innerHTML = `<div class="skeleton-grid">${"<div class='skeleton-card'></div>".repeat(4)}</div>`;

  try {
    const res  = await fetch(`/api/trending?${params}`);
    const data = await res.json();

    if (!data.trending || data.trending.length === 0) {
      $("trendingGrid").innerHTML = `<div class="empty-state"><i class="bi bi-graph-up-arrow"></i><p>No trending data available.</p></div>`;
      return;
    }

    const maxBorrows = Math.max(...data.trending.map(t => t.borrow_count_semester || 1));
    $("trendingGrid").innerHTML = "";

    data.trending.forEach((item, idx) => {
      const card = _buildBookCard(item);
      card.classList.add("trending-card");

      // Rank badge
      const rank = document.createElement("div");
      rank.className = `trending-rank ${idx < 3 ? "top3" : ""}`;
      rank.textContent = idx + 1;
      card.appendChild(rank);

      // Borrow bar
      const pct = Math.round((item.borrow_count_semester / maxBorrows) * 100);
      const barWrap = document.createElement("div");
      barWrap.className = "borrow-bar";
      const bar = document.createElement("div");
      bar.className = "borrow-bar-fill";
      bar.style.width = "0%";
      setTimeout(() => { bar.style.width = pct + "%"; }, 100);
      barWrap.appendChild(bar);

      const stats = document.createElement("div");
      stats.className = "d-flex justify-content-between mt-1";
      stats.innerHTML = `
        <small class="text-muted">${item.borrow_count_semester} borrows/sem</small>
        ${item.waitlist_count > 0 ? `<small style="color:var(--clr-warning)"><i class="bi bi-clock"></i> ${item.waitlist_count} waiting</small>` : ""}
      `;

      card.appendChild(barWrap);
      card.appendChild(stats);
      $("trendingGrid").appendChild(card);
    });
  } catch {
    $("trendingGrid").innerHTML = `<div class="empty-state"><i class="bi bi-exclamation-triangle"></i><p>Could not load trending books.</p></div>`;
  }
}

$("trendingDeptFilter")?.addEventListener("change", _loadTrending);

// ═══════════════════════════════════════════════════════════════════════════════
//  MY BOOKS / RESERVATIONS
// ═══════════════════════════════════════════════════════════════════════════════
function _bindReservationTabs() {
  $$("[data-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("[data-tab]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const tab = btn.dataset.tab;
      $("borrowsList").classList.toggle("d-none", tab !== "borrows");
      $("holdsList").classList.toggle("d-none",   tab !== "holds");
    });
  });

  $("refreshMyBooks")?.addEventListener("click", _loadMyBooks);
}

async function _loadMyBooks() {
  const studentId = STATE.studentProfile.student_id || "GUEST";

  [$("borrowsList"), $("holdsList")].forEach(el => {
    el.innerHTML = `<div class="text-center py-3"><div class="spinner-border spinner-border-sm text-primary"></div></div>`;
  });

  try {
    const res  = await fetch(`/api/waitlist/${studentId}`);
    const data = await res.json();

    // Active borrows
    const borrows = data.active_borrows || [];
    if (borrows.length === 0) {
      $("borrowsList").innerHTML = `<div class="empty-state"><i class="bi bi-book"></i><p>No books currently borrowed</p></div>`;
    } else {
      $("borrowsList").innerHTML = "";
      borrows.forEach(r => {
        $("borrowsList").appendChild(_buildMyBookItem(r, "borrow"));
      });
    }

    // Holds
    const holds = data.reservations || [];
    if (holds.length === 0) {
      $("holdsList").innerHTML = `<div class="empty-state"><i class="bi bi-bookmark"></i><p>No active holds or waitlist entries</p></div>`;
    } else {
      $("holdsList").innerHTML = "";
      holds.forEach(r => {
        $("holdsList").appendChild(_buildMyBookItem(r, "hold"));
      });
    }
  } catch {
    [$("borrowsList"), $("holdsList")].forEach(el => {
      el.innerHTML = `<div class="empty-state"><i class="bi bi-exclamation-triangle"></i><p>Could not load your books</p></div>`;
    });
  }
}

function _buildMyBookItem(record, type) {
  const div = document.createElement("div");
  div.className = "my-book-item";

  if (type === "borrow") {
    const due = record.due_date ? new Date(record.due_date) : null;
    const today = new Date();
    const isOverdue = due && due < today;
    const dueStr = due ? due.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" }) : "—";
    div.innerHTML = `
      <div class="my-book-spine"><i class="bi bi-book"></i></div>
      <div class="my-book-info">
        <div class="my-book-title">${_esc(record.title)}</div>
        <div class="my-book-meta">Borrowed: ${_esc(record.borrowed_date || "—")} &nbsp;·&nbsp; Renewals: ${record.renewals || 0}</div>
      </div>
      <div><span class="due-badge ${isOverdue ? "overdue" : ""}">
        <i class="bi bi-calendar3 me-1"></i>Due ${_esc(dueStr)}
      </span></div>`;
  } else {
    div.innerHTML = `
      <div class="my-book-spine"><i class="bi bi-bookmark-fill"></i></div>
      <div class="my-book-info">
        <div class="my-book-title">${_esc(record.title)}</div>
        <div class="my-book-meta">Reserved: ${_esc(record.reserved_date || "—")} &nbsp;·&nbsp;
          Status: <strong>${record.status === "ready" ? "✅ Ready for pickup" : "⏳ Waiting"}</strong>
        </div>
      </div>
      <div><span class="queue-badge"><i class="bi bi-list-ol me-1"></i>Queue #${record.queue_position}</span></div>`;
  }
  return div;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  BOOK CARD (shared)
// ═══════════════════════════════════════════════════════════════════════════════
function _buildBookCard(book, matchReason = null) {
  const card = document.createElement("div");
  card.className = "book-card";
  card.dataset.bookId = book.book_id;

  const status     = book.status || (book.copies_available > 0 ? "available" : "issued");
  const statusText = { available: "Available", issued: "Issued", reserved: "Reserved" }[status] || status;
  const statusCls  = `status-${status}`;
  const avail      = book.copies_available !== undefined ? book.copies_available : "?";
  const total      = book.copies_total !== undefined ? book.copies_total : "?";

  const colors = [
    "135deg, #3b6ef5, #7c5cd8",
    "135deg, #06b6d4, #3b6ef5",
    "135deg, #7c5cd8, #ec4899",
    "135deg, #22c55e, #3b6ef5",
    "135deg, #f59e0b, #ef4444",
  ];
  const colorIdx = book.book_id ? book.book_id.charCodeAt(1) % colors.length : 0;

  card.innerHTML = `
    <div class="book-card-header">
      <div class="book-card-spine" style="background: linear-gradient(${colors[colorIdx]})">
        <i class="bi bi-book-fill"></i>
      </div>
      <div class="book-card-info">
        <div class="book-title" title="${_esc(book.title)}">${_esc(book.title)}</div>
        <div class="book-author">${_esc(book.author || "—")}</div>
      </div>
    </div>
    <div class="book-meta">
      <span class="badge-avail ${statusCls}">${statusText} (${avail}/${total})</span>
      <span class="badge-shelf"><i class="bi bi-geo-alt me-1"></i>${_esc(book.shelf_location || "—")}</span>
    </div>
    <div class="book-footer">
      <span class="book-dept">${_esc(book.department || "")} ${book.year ? "· " + book.year : ""}</span>
      ${matchReason ? `<span class="match-reason" title="${_esc(matchReason)}">${_esc(matchReason)}</span>` : ""}
    </div>
  `;

  card.addEventListener("click", () => _openBookModal(book.book_id));
  return card;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  BOOK DETAIL MODAL
// ═══════════════════════════════════════════════════════════════════════════════
async function _openBookModal(bookId) {
  STATE.selectedBookId = bookId;
  $("bookModalTitle").textContent = "Loading…";
  $("bookModalBody").innerHTML = `<div class="text-center py-4"><div class="spinner-border text-primary"></div></div>`;
  $("modalReserveBtn").style.display = "none";
  bookModalInst.show();

  try {
    const res  = await fetch(`/api/books/${bookId}`);
    const book = await res.json();

    if (book.error) {
      $("bookModalBody").innerHTML = `<p class="text-danger">Book not found.</p>`;
      return;
    }

    $("bookModalTitle").textContent = book.title;

    const status     = book.status || "issued";
    const statusText = { available: "✅ Available", issued: "❌ Issued", reserved: "⏳ Reserved" }[status] || status;
    const statusCls  = `status-${status}`;

    $("bookModalBody").innerHTML = `
      <div class="d-flex gap-3 mb-3">
        <div style="width:52px;height:72px;border-radius:6px;background:linear-gradient(135deg,#3b6ef5,#7c5cd8);display:flex;align-items:center;justify-content:center;color:#fff;font-size:1.5rem;flex-shrink:0">
          <i class="bi bi-book-fill"></i>
        </div>
        <div>
          <h6 class="mb-0">${_esc(book.title)}</h6>
          <div class="text-muted" style="font-size:.85rem">${_esc(book.author)} · ${_esc(book.year || "")} · ${_esc(book.edition || "")} Ed.</div>
          <div class="mt-1"><span class="badge-avail ${statusCls}">${statusText}</span></div>
        </div>
      </div>
      <p style="font-size:.875rem;color:var(--text-secondary)">${_esc(book.description || "")}</p>
      <hr style="border-color:var(--border-color)">
      <div class="book-detail-grid">
        <div class="book-detail-row"><strong>Publisher</strong>${_esc(book.publisher || "—")}</div>
        <div class="book-detail-row"><strong>ISBN</strong>${_esc(book.isbn || "—")}</div>
        <div class="book-detail-row"><strong>Department</strong>${_esc(book.department || "—")}</div>
        <div class="book-detail-row"><strong>Subject</strong>${_esc(book.subject || "—")}</div>
        <div class="book-detail-row"><strong>Shelf</strong>${_esc(book.shelf_location || "—")}</div>
        <div class="book-detail-row"><strong>Language</strong>${_esc(book.language || "—")}</div>
        <div class="book-detail-row"><strong>Copies</strong>${book.copies_available} available / ${book.copies_total} total</div>
        ${book.due_back ? `<div class="book-detail-row"><strong>Due back</strong>${_esc(book.due_back)} <small class="text-muted">(verify at desk)</small></div>` : ""}
        ${book.waitlist_count ? `<div class="book-detail-row"><strong>Waitlist</strong>${book.waitlist_count} student(s) waiting</div>` : ""}
      </div>
      ${book.tags && book.tags.length ? `<div class="mt-3 d-flex flex-wrap gap-1">${book.tags.map(t => `<span class="source-chip">${_esc(t)}</span>`).join("")}</div>` : ""}
    `;

    $("modalReserveBtn").style.display = "inline-flex";
    $("modalReserveBtn").dataset.bookId = bookId;
  } catch {
    $("bookModalBody").innerHTML = `<p class="text-danger">Failed to load book details.</p>`;
  }
}

$("modalReserveBtn")?.addEventListener("click", async () => {
  const bookId = $("modalReserveBtn").dataset.bookId;
  if (!bookId) return;

  $("modalReserveBtn").disabled = true;
  $("modalReserveBtn").innerHTML = `<div class="spinner-border spinner-border-sm me-1"></div>Processing…`;

  try {
    const res  = await fetch("/api/reserve", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ book_id: bookId, student_id: STATE.studentProfile.student_id }),
    });
    const data = await res.json();

    bookModalInst.hide();
    _showToast(data.message || (data.success ? "Reservation placed!" : "Could not place reservation."),
               data.success ? "success" : "warning");
  } catch {
    _showToast("Network error placing reservation.", "danger");
  } finally {
    $("modalReserveBtn").disabled = false;
    $("modalReserveBtn").innerHTML = `<i class="bi bi-bookmark-plus me-1"></i>Reserve / Join Waitlist`;
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
//  STUDENT PROFILE
// ═══════════════════════════════════════════════════════════════════════════════
function _loadProfile() {
  const saved = localStorage.getItem("lib-profile");
  if (saved) {
    try {
      STATE.studentProfile = JSON.parse(saved);
      _populateProfileForm();
      _updateNavName();
    } catch {}
  }

  // Also sync to server session
  fetch("/api/student/profile", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(STATE.studentProfile),
  }).catch(() => {});
}

function _populateProfileForm() {
  const p = STATE.studentProfile;
  $("pName").value      = p.name || "";
  $("pStudentId").value = p.student_id || "";
  $("pBranch").value    = p.branch || "Computer Science";
  $("pSemester").value  = p.semester || "";
  $("pCourses").value   = (p.courses || []).join(", ");
  $("pLang").value      = p.language || "English";
}

function _updateNavName() {
  const nav = $("navStudentName");
  if (nav) nav.textContent = STATE.studentProfile.name || "My Profile";
}

function _bindProfile() {
  $("saveProfile")?.addEventListener("click", async () => {
    const profile = {
      name:       $("pName").value.trim() || "Student",
      student_id: $("pStudentId").value.trim() || "GUEST",
      branch:     $("pBranch").value,
      semester:   $("pSemester").value,
      courses:    $("pCourses").value.split(",").map(c => c.trim()).filter(Boolean),
      language:   $("pLang").value,
    };

    STATE.studentProfile = profile;
    localStorage.setItem("lib-profile", JSON.stringify(profile));
    _updateNavName();
    $("chatLang").value = profile.language;

    try {
      await fetch("/api/student/profile", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(profile),
      });
    } catch {}

    profileModalInst.hide();
    _showToast("Profile saved! Refreshing recommendations…", "success");
    setTimeout(_loadRecommendations, 300);
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TOAST
// ═══════════════════════════════════════════════════════════════════════════════
function _showToast(msg, type = "info") {
  const toast = $("appToast");
  toast.className = `toast align-items-center text-bg-${type} border-0`;
  $("toastMsg").textContent = msg;
  toastInst.show();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════
function _esc(str) {
  if (str === undefined || str === null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}
