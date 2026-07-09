"""
app.py — Flask backend for the Library AI Agent
─────────────────────────────────────────────────
Endpoints:
  POST /api/chat              — RAG + Granite LLM conversational query
  GET  /api/books/search      — full-text book search
  GET  /api/books/<book_id>   — book detail + availability
  POST /api/reserve           — place a reservation / join waitlist
  GET  /api/trending          — high-demand & trending titles
  GET  /api/recommendations   — personalised book recommendations
  GET  /api/student/profile   — get student profile
  POST /api/student/profile   — save/update student profile
  POST /api/index/rebuild     — (admin) rebuild the vector store
"""

import os
import csv
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Flask app ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
KB_DIR     = BASE_DIR / "knowledge_base"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["JSON_AS_ASCII"] = False   # allow UTF-8 (Hindi, Marathi, box chars) in responses
CORS(app, supports_credentials=True)

# ─── Config ──────────────────────────────────────────────────────────────────
IBM_API_KEY         = os.getenv("IBM_API_KEY", "")
WATSONX_PROJECT_ID  = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL         = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
LLM_MODEL_ID        = os.getenv("LLM_MODEL_ID", "ibm/granite-13b-chat-v2")
LIBRARY_NAME        = os.getenv("LIBRARY_NAME", "University Central Library")

# ─── Lazy imports (expensive) ────────────────────────────────────────────────
_rag_ready  = False
_watsonx_llm = None

def _ensure_rag():
    global _rag_ready
    if not _rag_ready:
        from rag_pipeline import initialize_vector_store
        initialize_vector_store()
        _rag_ready = True


# Sentinel so a failed init doesn't retry on every request
_WATSONX_FAILED = object()

def _get_watsonx_llm():
    """
    Returns an ibm_watsonx_ai ModelInference instance (lazy-loaded).
    Returns None if credentials are missing or invalid — app falls back to
    demo mode. Uses a sentinel so a bad-credential error only logs once.
    """
    global _watsonx_llm
    # Already tried and failed — don't retry
    if _watsonx_llm is _WATSONX_FAILED:
        return None
    if _watsonx_llm is not None:
        return _watsonx_llm

    if not IBM_API_KEY or IBM_API_KEY.startswith("your_") \
            or not WATSONX_PROJECT_ID or WATSONX_PROJECT_ID.startswith("your_"):
        logger.warning("IBM credentials not configured — running in DEMO mode")
        _watsonx_llm = _WATSONX_FAILED
        return None

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        credentials = Credentials(url=WATSONX_URL, api_key=IBM_API_KEY)
        _watsonx_llm = ModelInference(
            model_id=LLM_MODEL_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={
                GenParams.MAX_NEW_TOKENS: 1024,
                GenParams.TEMPERATURE:    0.3,
                GenParams.TOP_P:          0.9,
                GenParams.REPETITION_PENALTY: 1.1,
            },
        )
        logger.info(f"watsonx.ai LLM ready: {LLM_MODEL_ID}")
    except Exception as e:
        logger.error(f"Failed to initialize watsonx LLM: {e}")
        _watsonx_llm = _WATSONX_FAILED
        return None

    return _watsonx_llm


# ══════════════════════════════════════════════════════════════════════════════
#  CATALOG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_catalog_cache: Optional[list] = None
_circulation_cache: Optional[dict] = None


def _load_catalog() -> list:
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    catalog = []
    csv_path = KB_DIR / "book_catalog" / "catalog.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["copies_total"]     = int(row.get("copies_total", 0))
                row["copies_available"] = int(row.get("copies_available", 0))
                row["tags"] = row.get("tags", "").split(";")
                catalog.append(row)
    _catalog_cache = catalog
    return catalog


def _load_circulation() -> dict:
    global _circulation_cache
    if _circulation_cache is not None:
        return _circulation_cache

    circ_path = KB_DIR / "book_catalog" / "circulation_data.json"
    if circ_path.exists():
        with open(circ_path, encoding="utf-8") as f:
            _circulation_cache = json.load(f)
    else:
        _circulation_cache = {"borrow_records": [], "reservations": [], "trending_books": []}
    return _circulation_cache


def _get_book_status(book_id: str) -> dict:
    """Returns availability detail for a given book_id."""
    catalog   = _load_catalog()
    circ_data = _load_circulation()

    book = next((b for b in catalog if b["book_id"] == book_id), None)
    if not book:
        return {"status": "not_found"}

    avail = book["copies_available"]
    total = book["copies_total"]

    # Find active borrows for this book
    active_borrows = [
        r for r in circ_data.get("borrow_records", [])
        if r["book_id"] == book_id and not r.get("returned", False)
    ]

    # Find waitlist
    waitlist = [
        r for r in circ_data.get("reservations", [])
        if r["book_id"] == book_id
    ]

    earliest_return = None
    if active_borrows:
        due_dates = [r["due_date"] for r in active_borrows]
        earliest_return = min(due_dates)

    if avail > 0:
        status = "available"
    elif len(waitlist) > 0:
        status = "reserved"
    else:
        status = "issued"

    return {
        "book_id":        book_id,
        "title":          book["title"],
        "author":         book["author"],
        "shelf_location": book["shelf_location"],
        "copies_total":   total,
        "copies_available": avail,
        "status":         status,
        "due_back":       earliest_return,
        "waitlist_count": len(waitlist),
        "department":     book["department"],
        "subject":        book["subject"],
        "isbn":           book["isbn"],
        "year":           book["year"],
        "edition":        book["edition"],
        "publisher":      book["publisher"],
        "language":       book["language"],
        "description":    book["description"],
        "tags":           book["tags"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Serve frontend
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR / "templates"), "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "library": LIBRARY_NAME, "timestamp": datetime.utcnow().isoformat()})


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _safe_str(text: str) -> str:
    """
    Sanitise a string so it is safe to log and serialise on Windows cp1252.
    1. Replace known problematic Unicode chars with readable ASCII equivalents.
    2. Nuclear fallback: encode to cp1252 ignoring errors, then decode back —
       any character still not representable is simply dropped rather than
       raising UnicodeEncodeError.
    The JSON HTTP response body is always UTF-8 (JSON_AS_ASCII=False), so
    Hindi / Marathi Devanagari characters are preserved end-to-end; only the
    Windows console logging path needs the cp1252 safety net.
    """
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2550": "=",   # ═  box double horizontal
        "\u2551": "|",   # ║  box double vertical
        "\u2554": "+",   # ╔
        "\u2557": "+",   # ╗
        "\u255a": "+",   # ╚
        "\u255d": "+",   # ╝
        "\u2560": "+",   # ╠
        "\u2563": "+",   # ╣
        "\u2566": "+",   # ╦
        "\u2569": "+",   # ╩
        "\u256c": "+",   # ╬
        "\u2500": "-",   # ─  box light horizontal
        "\u2502": "|",   # │  box light vertical
        "\u250c": "+",   # ┌
        "\u2510": "+",   # ┐
        "\u2514": "+",   # └
        "\u2518": "+",   # ┘
        "\u251c": "+",   # ├
        "\u2524": "+",   # ┤
        "\u252c": "+",   # ┬
        "\u2534": "+",   # ┴
        "\u253c": "+",   # ┼
        "\u2014": "--",  # — em dash
        "\u2013": "-",   # – en dash
        "\u2019": "'",   # ' right single quote
        "\u2018": "'",   # ' left single quote
        "\u201c": '"',   # " left double quote
        "\u201d": '"',   # " right double quote
        "\u2022": "*",   # • bullet (round)
        "\u00b7": ".",   # · middle dot (used in catalog CSV)
        "\u00a0": " ",   # non-breaking space
        "\u00e9": "e",   # é  (e.g. Géron)
        "\u00e8": "e",   # è
        "\u00ea": "e",   # ê
        "\u00eb": "e",   # ë
        "\u00e0": "a",   # à
        "\u00e2": "a",   # â
        "\u00e4": "a",   # ä
        "\u00f6": "o",   # ö
        "\u00fc": "u",   # ü
        "\u00df": "ss",  # ß
        "\u00e7": "c",   # ç
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Nuclear fallback: strip any remaining non-cp1252 chars from the
    # string used in logging/tracebacks (the HTTP response is UTF-8)
    text = text.encode("cp1252", errors="replace").decode("cp1252")
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Chat (RAG + LLM)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data    = request.get_json(force=True)
        query   = data.get("message", "").strip()
        profile = data.get("student_profile", {})

        if not query:
            return jsonify({"error": "Empty query"}), 400

        _ensure_rag()
        from rag_pipeline import retrieve
        from agent_instructions import build_system_prompt

        # RAG retrieval — sanitize context to avoid cp1252 encode errors on Windows
        context       = _safe_str(retrieve(query))
        system_prompt = build_system_prompt(context, profile)

        # Build messages array for Granite chat model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ]

        llm = _get_watsonx_llm()
        if llm is not None:
            try:
                # Real watsonx.ai call
                prompt_text = _build_granite_prompt(messages)
                response    = llm.generate_text(prompt=prompt_text)
                answer      = _safe_str(response.strip() if isinstance(response, str) else str(response))
            except Exception as llm_err:
                # Auth failure / quota / network — log once, fall back to demo
                logger.warning(f"watsonx LLM call failed ({llm_err}), falling back to demo mode")
                global _watsonx_llm
                _watsonx_llm = _WATSONX_FAILED   # stop retrying
                answer = _demo_response(query, context, profile)
        else:
            # Demo/dev fallback — generate a useful response from RAG context
            answer = _demo_response(query, context, profile)

        return jsonify({
            "response": answer,
            "sources":  _extract_sources(context),
        })

    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "detail": _safe_str(str(e))}), 500


def _build_granite_prompt(messages: list) -> str:
    """
    Formats messages for IBM Granite chat models using the
    <|system|> / <|user|> / <|assistant|> turn format.
    """
    prompt = ""
    for m in messages:
        role    = m["role"]
        content = m["content"]
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"
    return prompt


# ─── Intent detection helpers ────────────────────────────────────────────────

_GREET_KW = {
    "hello", "hi", "hey", "namaste", "namaskar", "hlo", "hii",
    "good morning", "good afternoon", "good evening", "good night",
    "greetings", "howdy", "sup", "what's up", "whats up",
    # Hindi / Marathi
    "\u0928\u092e\u0938\u094d\u0924\u0947",   # नमस्ते
    "\u0928\u092e\u0938\u094d\u0915\u093e\u0930",  # नमस्कार
    "\u0939\u0947\u0932\u094b",               # हेलो
}
_THANKS_KW  = {"thank", "thanks", "thankyou", "thank you", "dhanyavad",
               "\u0927\u0928\u094d\u092f\u0935\u093e\u0926"}  # धन्यवाद
_HELP_KW    = {"help", "what can you do", "how do you work", "capabilities",
               "kya kar sakte", "madad", "use karna"}
_ENOUGH_KW  = {"enough", "sufficient", "is this enough", "more books",
               "required any other", "need more", "aur books", "aur chahiye",
               "kuch aur", "bas itna", "sufficient hai"}
_SUGGEST_KW = {"suggest", "recommend", "which book", "books for", "book for",
               "best book", "top book", "good book", "useful book",
               "for my course", "for exam", "gate exam", "placement", "interview",
               "kaunsi kitab", "kaun si", "sujhao", "batao", "bata do",
               "pustak", "kitab",
               "\u0938\u0941\u091d\u093e\u0935", "\u0938\u0941\u091a\u0935\u093e",  # सुझाव / सुचवा
               "\u0915\u093f\u0924\u093e\u092c"}                                     # किताब
_AVAIL_KW   = {"available", "availability", "in stock", "copy", "copies",
               "is there", "do you have", "kya available", "milegi", "milega",
               "hai kya", "\u0909\u092a\u0932\u092c\u094d\u0927"}                   # उपलब्ध
_SHELF_KW   = {"shelf", "location", "where is", "kahan hai", "kahan milega",
               "rack", "section", "floor"}
_SYLLABUS_KW= {"syllabus", "course", "semester", "subject", "curriculum",
               "unit", "topics", "what to study", "cs501", "cs301", "cs401",
               "cs601", "me101", "ee201"}
_TRENDING_KW= {"trending", "popular", "most issued", "high demand", "waitlist",
               "frequently", "top books", "best sellers", "most borrowed"}
_AUTHOR_KW  = {"author", "written by", "who wrote", "kisne likha"}
_DUE_KW     = {"due", "due date", "return date", "when to return", "kab return"}


def _intent(q: str) -> str:
    """Return the intent label for the query."""
    q = q.lower().strip()
    # single-word or very short — likely greeting or noise
    words = q.split()
    if len(words) <= 2:
        if any(kw in q for kw in _GREET_KW):
            return "greeting"
        if any(kw in q for kw in _THANKS_KW):
            return "thanks"
        if any(kw in q for kw in _HELP_KW):
            return "help"
    # longer queries
    if any(kw in q for kw in _GREET_KW) and len(words) <= 4:
        return "greeting"
    if any(kw in q for kw in _THANKS_KW):
        return "thanks"
    if any(kw in q for kw in _HELP_KW):
        return "help"
    if any(kw in q for kw in _ENOUGH_KW):
        return "followup_enough"
    if any(kw in q for kw in _AVAIL_KW):
        return "availability"
    if any(kw in q for kw in _SHELF_KW):
        return "shelf"
    if any(kw in q for kw in _SUGGEST_KW):
        return "suggest"
    if any(kw in q for kw in _TRENDING_KW):
        return "trending"
    if any(kw in q for kw in _SYLLABUS_KW):
        return "syllabus"
    if any(kw in q for kw in _AUTHOR_KW):
        return "author"
    if any(kw in q for kw in _DUE_KW):
        return "due_date"
    return "general"


def _books_from_context(context: str, limit: int = 5) -> list[dict]:
    """Parse structured book fields out of RAG context chunks."""
    books = []
    for chunk in context.split("---"):
        book: dict = {}
        for line in chunk.split("\n"):
            line = line.strip()
            if line.startswith("Book Title:"):
                book["title"] = line.replace("Book Title:", "").strip()
            elif line.startswith("Author:"):
                book["author"] = line.replace("Author:", "").strip()
            elif line.startswith("Shelf Location:"):
                book["shelf"] = line.replace("Shelf Location:", "").strip()
            elif line.startswith("Availability:"):
                book["availability"] = line.replace("Availability:", "").strip()
            elif line.startswith("Subject:"):
                book["subject"] = line.replace("Subject:", "").strip()
        if book.get("title"):
            books.append(book)
        if len(books) >= limit:
            break
    return books


def _format_book_list(books: list[dict]) -> str:
    """Format a list of book dicts into readable lines."""
    lines = []
    for i, b in enumerate(books, 1):
        avail = b.get("availability", "")
        shelf = b.get("shelf", "")
        parts = [f"{i}. {b['title']}"]
        if b.get("author"):
            parts[0] += f"  by {b['author']}"
        if avail:
            parts.append(f"   Availability: {avail}")
        if shelf:
            parts.append(f"   Shelf: {shelf}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _demo_response(query: str, context: str, profile: dict) -> str:
    """
    Intent-aware conversational response engine used when IBM credentials
    are not configured. Detects 12 intent types and formats structured,
    relevant answers from RAG context. Falls back to a polite out-of-scope
    message for truly unrelated queries (greetings, thanks, etc.).
    """
    intent   = _intent(query)
    name     = profile.get("name", "").strip() or "there"
    books    = _books_from_context(context)
    q_lower  = query.lower()

    # ── Greeting ──────────────────────────────────────────────────────────────
    if intent == "greeting":
        branch = profile.get("branch", "")
        sem    = profile.get("semester", "")
        greeting_part = f"Hello, {name}!" if name not in ("there", "Student", "") else "Hello!"
        profile_hint  = ""
        if branch and sem:
            profile_hint = (f" I see you are a {sem} semester {branch} student."
                            f" I can help you with book recommendations, availability checks,"
                            f" syllabus queries, and more.")
        elif branch:
            profile_hint = f" I can help you find books for {branch} and more."
        return _safe_str(
            f"{greeting_part} Welcome to Grantha, your AI Library Assistant.{profile_hint}\n\n"
            f"You can ask me things like:\n"
            f"- 'Suggest books for my Data Structures course'\n"
            f"- 'Is Introduction to Algorithms available?'\n"
            f"- 'What are trending books in Computer Science?'\n"
            f"- 'What books are on the CS501 Machine Learning syllabus?'\n\n"
            f"How can I help you today?"
        )

    # ── Thanks ────────────────────────────────────────────────────────────────
    if intent == "thanks":
        return _safe_str(
            f"You're welcome{', ' + name if name not in ('there','Student','') else ''}! "
            f"Feel free to ask if you need more help with books or the library."
        )

    # ── Help ──────────────────────────────────────────────────────────────────
    if intent == "help":
        return _safe_str(
            "Here is what I can help you with:\n\n"
            "1. Book recommendations  - 'Suggest books for Machine Learning'\n"
            "2. Availability check    - 'Is CLRS available right now?'\n"
            "3. Shelf location        - 'Where is the Operating Systems book?'\n"
            "4. Syllabus queries      - 'What books are on the CS501 syllabus?'\n"
            "5. Trending books        - 'Most issued books this semester'\n"
            "6. Author search         - 'Books by Cormen'\n"
            "7. Hindi / Marathi       - You can ask in Hindi or Marathi too!\n\n"
            "Just type your question naturally and I will do my best to help."
        )

    # ── Follow-up: is this enough / need more? ────────────────────────────────
    if intent == "followup_enough":
        if books:
            extra = [b for b in books]
            followup_list = _format_book_list(extra[:3])
            return _safe_str(
                "These core books should cover the fundamentals well. "
                "If you want to go deeper, here are some additional titles from our catalog:\n\n"
                + followup_list
                + "\n\nYou can also check the department reading list at the library desk "
                  "for the full recommended list for your semester."
            )
        return _safe_str(
            "The books mentioned should be a good starting point. "
            "Visit the library desk or use the Search panel to explore more titles in your area."
        )

    # ── Availability check ────────────────────────────────────────────────────
    if intent == "availability":
        if books:
            result_lines = []
            for b in books[:4]:
                avail = b.get("availability", "unknown")
                shelf = b.get("shelf", "N/A")
                status = "AVAILABLE" if "available" in avail.lower() and not avail.startswith("0") else "ISSUED/RESERVED"
                result_lines.append(
                    f"- {b['title']}\n"
                    f"  Status: {status}  |  {avail}  |  Shelf: {shelf}"
                )
            return _safe_str(
                "Here is the current availability for relevant books:\n\n"
                + "\n\n".join(result_lines)
                + "\n\nNote: Please confirm exact availability at the circulation desk, "
                  "as copies may have been issued or returned since last update."
            )
        return _safe_str(
            "I could not find that specific book in the catalog. "
            "Please check the title spelling or ask at the library desk.\n"
            "You can also use the 'Search' panel on the left to browse the full catalog."
        )

    # ── Shelf / location ──────────────────────────────────────────────────────
    if intent == "shelf":
        if books:
            location_lines = [
                f"- {b['title']}  -->  Shelf: {b.get('shelf', 'Ask at desk')}"
                for b in books[:4]
            ]
            return _safe_str(
                "Shelf locations for the relevant books:\n\n"
                + "\n".join(location_lines)
                + "\n\nShelf codes: CS = Computer Science wing, ME = Mechanical, "
                  "EE = Electrical, MATH = Mathematics, PHY = Physics."
            )
        return _safe_str("I could not find that book's location. Please ask at the library help desk.")

    # ── Suggest / recommend ────────────────────────────────────────────────────
    if intent == "suggest":
        is_exam = any(kw in q_lower for kw in ["gate", "exam", "placement", "interview", "competitive"])
        intro = (
            "For GATE and competitive exam preparation, these foundational books from our catalog are highly recommended:\n\n"
            if is_exam else
            "Based on your query, here are the recommended books from our library catalog:\n\n"
        )
        # If RAG didn't return books, fall back to profile branch catalog lookup
        if not books:
            branch  = profile.get("branch", "Computer Science")
            catalog = _load_catalog()
            fallback = [
                b for b in catalog
                if b.get("department", "").lower() == branch.lower()
                and int(b.get("copies_available", 0)) > 0
            ][:5]
            if fallback:
                books = [
                    {"title": b["title"], "author": b.get("author",""),
                     "shelf": b.get("shelf_location",""),
                     "availability": f"{b['copies_available']} of {b['copies_total']} copies available"}
                    for b in fallback
                ]
        if books:
            return _safe_str(
                intro
                + _format_book_list(books[:5])
                + "\n\nTip: Available copies are ready for immediate borrowing. "
                  "Use the 'Search' panel to check live availability and reserve copies."
            )
        return _safe_str(
            "I could not find specific book recommendations for that query in the current catalog. "
            "Try the 'Search' panel with keywords, or ask the librarian for guidance."
        )

    # ── Trending ──────────────────────────────────────────────────────────────
    if intent == "trending":
        # Load real trending data
        try:
            circ = _load_circulation()
            trending = circ.get("trending_books", [])
            catalog_map = {b["book_id"]: b for b in _load_catalog()}
            q_dept = ""
            for dept in ["computer science", "mathematics", "physics", "electrical"]:
                if dept in q_lower:
                    q_dept = dept
                    break
            if q_dept:
                trending = [t for t in trending if q_dept in t.get("department", "").lower()]
            if trending:
                lines_out = []
                for i, t in enumerate(trending[:5], 1):
                    book = catalog_map.get(t["book_id"], {})
                    shelf = book.get("shelf_location", "N/A")
                    avail = book.get("copies_available", "?")
                    wait  = t.get("waitlist_count", 0)
                    borrows = t.get("borrow_count_semester", "?")
                    wl_note = f"  [Waitlist: {wait} students]" if wait > 0 else ""
                    lines_out.append(
                        f"{i}. {t['title']}\n"
                        f"   Borrows this semester: {borrows}  |  Available: {avail}  |  Shelf: {shelf}{wl_note}"
                    )
                dept_label = f" in {q_dept.title()}" if q_dept else ""
                return _safe_str(
                    f"Most issued and trending books{dept_label} this semester:\n\n"
                    + "\n\n".join(lines_out)
                    + "\n\nVisit the 'Trending' panel for a live chart with borrow statistics."
                )
        except Exception:
            pass
        if books:
            return _safe_str(
                "Frequently borrowed books related to your query:\n\n"
                + _format_book_list(books[:5])
            )
        return _safe_str("No trending data available for that query. Check the 'Trending' panel for live stats.")

    # ── Syllabus ──────────────────────────────────────────────────────────────
    if intent == "syllabus":
        # Try to extract course code or semester from query
        import re
        course_match = re.search(r'CS\d{3}|ME\d{3}|EE\d{3}|MA\d{3}', query.upper())
        sem_match    = re.search(r'(\d+)(st|nd|rd|th)\s*sem', q_lower)

        # Pull relevant syllabus lines from context
        syllabus_lines = []
        unit_lines = []
        book_lines_raw = []
        for line in [l.strip() for l in context.split("\n") if l.strip()]:
            if line.startswith("UNIT") or line.startswith("Recommended Books"):
                unit_lines.append(line)
            elif line.startswith("Book Title:"):
                book_lines_raw.append(line.replace("Book Title:", "").strip())

        header = ""
        if course_match:
            header = f"Syllabus information for {course_match.group()}:\n\n"
        elif sem_match:
            header = f"Books and syllabus for {sem_match.group(1)}{sem_match.group(2)} semester:\n\n"
        else:
            header = "Syllabus and reading information:\n\n"

        if unit_lines:
            body = "\n".join(unit_lines[:12])
        elif book_lines_raw:
            body = "Recommended books:\n" + "\n".join(f"- {t}" for t in book_lines_raw[:6])
        elif books:
            body = "Relevant books found:\n" + _format_book_list(books[:4])
        else:
            body = "No specific syllabus data found. Please check with your department."

        return _safe_str(header + body + "\n\nFor the full syllabus, contact your department office "
                         "or visit the reading lists section at the library desk.")

    # ── Author search ─────────────────────────────────────────────────────────
    if intent == "author":
        author_lines = []
        for chunk in context.split("---"):
            title = author = ""
            for line in chunk.split("\n"):
                line = line.strip()
                if line.startswith("Book Title:"):
                    title = line.replace("Book Title:", "").strip()
                elif line.startswith("Author:"):
                    author = line.replace("Author:", "").strip()
            if title and author:
                author_lines.append(f"- {title}  by  {author}")
        if author_lines:
            return _safe_str(
                "Books matching your author query:\n\n"
                + "\n".join(author_lines[:5])
                + "\n\nUse the 'Search' panel to filter by author name for full results."
            )
        return _safe_str("I could not find that author in the catalog. Try the 'Search' panel with the author's name.")

    # ── Due date ──────────────────────────────────────────────────────────────
    if intent == "due_date":
        return _safe_str(
            "Due date information is specific to your account.\n\n"
            "Please check:\n"
            "- The 'My Books' panel (left sidebar) to see your currently borrowed books and due dates.\n"
            "- The library circulation desk for the most accurate due date, especially after renewals.\n\n"
            "Note: Books can usually be renewed once if no waitlist exists."
        )

    # ── General / fallback — use context intelligently ────────────────────────
    if context and context != "No relevant information found in the knowledge base.":
        if books:
            return _safe_str(
                f"Here are the most relevant books I found for: \"{query}\"\n\n"
                + _format_book_list(books[:4])
                + "\n\nWant more details? Ask me to check availability, "
                  "recommend course books, or search by topic."
            )
        # Extract meaningful non-metadata lines from context
        meaningful = [
            l.strip() for l in context.split("\n")
            if l.strip()
            and not l.strip().startswith(("ISBN:", "Publisher:", "Language:", "Tags:"))
            and len(l.strip()) > 20
        ]
        if meaningful:
            return _safe_str(
                f"Here is what I found related to your question:\n\n"
                + "\n".join(meaningful[:8])
                + "\n\nFor more details, try asking a specific question like "
                  "'Is [book title] available?' or 'Suggest books for [course name]'."
            )

    # ── True fallback: out-of-scope ───────────────────────────────────────────
    return _safe_str(
        "I am Grantha, your library assistant. I can help with:\n\n"
        "- Finding and recommending books for your courses\n"
        "- Checking book availability and shelf locations\n"
        "- Browsing syllabus reading lists\n"
        "- Discovering trending and high-demand titles\n\n"
        "Please ask a library-related question and I will be happy to help!"
    )


def _extract_sources(context: str) -> list:
    sources = []
    for chunk in context.split("---"):
        for line in chunk.split("\n"):
            if line.startswith("Book Title:"):
                title = line.replace("Book Title:", "").strip()
                if title not in sources:
                    sources.append(title)
    return sources[:5]


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Book Search & Detail
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/books/search")
def search_books():
    q          = request.args.get("q", "").lower().strip()
    dept       = request.args.get("department", "").lower().strip()
    subject    = request.args.get("subject", "").lower().strip()
    available  = request.args.get("available_only", "false").lower() == "true"
    page       = int(request.args.get("page", 1))
    per_page   = int(request.args.get("per_page", 12))

    catalog = _load_catalog()
    results = []

    for book in catalog:
        # Filter by search query
        if q:
            searchable = " ".join([
                book.get("title", ""),
                book.get("author", ""),
                book.get("subject", ""),
                book.get("department", ""),
                book.get("description", ""),
                " ".join(book.get("tags", [])),
            ]).lower()
            if q not in searchable:
                continue

        if dept and dept not in book.get("department", "").lower():
            continue
        if subject and subject not in book.get("subject", "").lower():
            continue
        if available and book.get("copies_available", 0) == 0:
            continue

        status = "available" if book["copies_available"] > 0 else "issued"
        results.append({
            **book,
            "status": status,
        })

    total   = len(results)
    start   = (page - 1) * per_page
    end     = start + per_page
    paged   = results[start:end]

    return jsonify({
        "books":       paged,
        "total":       total,
        "page":        page,
        "per_page":    per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    })


@app.route("/api/books/<book_id>")
def book_detail(book_id: str):
    detail = _get_book_status(book_id)
    if detail.get("status") == "not_found":
        return jsonify({"error": "Book not found"}), 404
    return jsonify(detail)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Reservations & Waitlist
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/reserve", methods=["POST"])
def reserve_book():
    data       = request.get_json(force=True)
    book_id    = data.get("book_id", "")
    student_id = data.get("student_id", "GUEST")

    if not book_id:
        return jsonify({"error": "book_id required"}), 400

    detail = _get_book_status(book_id)
    if detail.get("status") == "not_found":
        return jsonify({"error": "Book not found"}), 404

    circ = _load_circulation()

    # Check if already reserved by this student
    existing = [
        r for r in circ.get("reservations", [])
        if r["book_id"] == book_id and r["student_id"] == student_id
    ]
    if existing:
        return jsonify({
            "success":      False,
            "message":      "You already have a reservation for this book.",
            "queue_position": existing[0]["queue_position"],
        })

    current_waitlist = [r for r in circ.get("reservations", []) if r["book_id"] == book_id]
    queue_pos        = len(current_waitlist) + 1

    new_reservation = {
        "student_id":     student_id,
        "book_id":        book_id,
        "title":          detail["title"],
        "reserved_date":  datetime.utcnow().strftime("%Y-%m-%d"),
        "queue_position": queue_pos,
        "status":         "ready" if detail["copies_available"] > 0 else "waiting",
    }
    circ.setdefault("reservations", []).append(new_reservation)

    # Persist updated circulation
    circ_path = KB_DIR / "book_catalog" / "circulation_data.json"
    with open(circ_path, "w", encoding="utf-8") as f:
        json.dump(circ, f, indent=2)
    global _circulation_cache
    _circulation_cache = circ

    return jsonify({
        "success":        True,
        "message":        (
            f"Reservation placed! Queue position: {queue_pos}. "
            f"The book is {'ready for pickup' if new_reservation['status'] == 'ready' else 'currently issued; you will be notified when available'}."
        ),
        "queue_position": queue_pos,
        "status":         new_reservation["status"],
        "book_title":     detail["title"],
    })


@app.route("/api/waitlist/<student_id>")
def get_waitlist(student_id: str):
    circ = _load_circulation()
    reservations = [
        r for r in circ.get("reservations", [])
        if r["student_id"] == student_id
    ]
    active_borrows = [
        r for r in circ.get("borrow_records", [])
        if r["student_id"] == student_id and not r.get("returned", False)
    ]
    return jsonify({
        "reservations":  reservations,
        "active_borrows": active_borrows,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Trending Books
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/trending")
def trending():
    dept  = request.args.get("department", "").lower().strip()
    limit = int(request.args.get("limit", 8))

    circ    = _load_circulation()
    catalog = _load_catalog()

    catalog_map = {b["book_id"]: b for b in catalog}
    trending    = circ.get("trending_books", [])

    if dept:
        trending = [t for t in trending if dept in t.get("department", "").lower()]

    # Enrich with shelf location and availability from catalog
    enriched = []
    for item in trending[:limit]:
        book = catalog_map.get(item["book_id"], {})
        enriched.append({
            **item,
            "shelf_location":   book.get("shelf_location", ""),
            "copies_available": book.get("copies_available", 0),
            "copies_total":     book.get("copies_total", 0),
            "author":           book.get("author", ""),
            "status": "available" if book.get("copies_available", 0) > 0 else "issued",
        })

    # Sort by borrow_count_semester descending
    enriched.sort(key=lambda x: x.get("borrow_count_semester", 0), reverse=True)
    return jsonify({"trending": enriched})


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Personalised Recommendations
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/recommendations")
def recommendations():
    branch   = request.args.get("branch", "Computer Science")
    semester = request.args.get("semester", "")
    courses  = request.args.get("courses", "")

    catalog = _load_catalog()

    course_list = [c.strip() for c in courses.split(",") if c.strip()]

    # Keyword mapping for common courses
    course_keywords = {
        "machine learning":         ["machine learning", "deep learning", "neural", "data science"],
        "data structures":          ["algorithms", "data structures", "discrete"],
        "artificial intelligence":  ["artificial intelligence", "AI", "search", "reasoning"],
        "computer networks":        ["networking", "network", "TCP", "IP"],
        "operating systems":        ["operating system", "OS", "processes", "memory"],
        "database":                 ["database", "SQL", "relational"],
        "software engineering":     ["software engineering", "clean code", "agile"],
        "computer vision":          ["computer vision", "image processing"],
        "nlp":                      ["NLP", "natural language", "text"],
    }

    matched_books = []
    seen_ids      = set()

    for course in course_list:
        c_lower = course.lower()
        keywords = []
        for key, kws in course_keywords.items():
            if key in c_lower or any(k.lower() in c_lower for k in kws):
                keywords.extend(kws)

        if not keywords:
            keywords = [c_lower]

        for book in catalog:
            if book["book_id"] in seen_ids:
                continue
            book_text = " ".join([
                book.get("subject", ""),
                book.get("description", ""),
                " ".join(book.get("tags", [])),
            ]).lower()
            if any(kw.lower() in book_text for kw in keywords):
                seen_ids.add(book["book_id"])
                matched_books.append({
                    **book,
                    "status": "available" if book["copies_available"] > 0 else "issued",
                    "match_reason": f"Relevant to {course}",
                })

    # If no matches, fall back to department books
    if not matched_books:
        dept_lower = branch.lower()
        for book in catalog:
            if book["book_id"] not in seen_ids and dept_lower in book.get("department", "").lower():
                seen_ids.add(book["book_id"])
                matched_books.append({
                    **book,
                    "status": "available" if book["copies_available"] > 0 else "issued",
                    "match_reason": f"Recommended for {branch} students",
                })

    # Sort: available first, then by course match
    matched_books.sort(key=lambda x: (0 if x["status"] == "available" else 1))

    return jsonify({
        "recommendations": matched_books[:12],
        "profile":         {"branch": branch, "semester": semester, "courses": course_list},
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Student Profile (session-based)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/student/profile", methods=["GET", "POST"])
def student_profile():
    if request.method == "POST":
        data = request.get_json(force=True)
        session["student_profile"] = {
            "name":       data.get("name", "Student"),
            "student_id": data.get("student_id", "GUEST"),
            "branch":     data.get("branch", "Computer Science"),
            "semester":   data.get("semester", ""),
            "courses":    data.get("courses", []),
            "language":   data.get("language", "English"),
        }
        return jsonify({"success": True, "profile": session["student_profile"]})

    profile = session.get("student_profile", {
        "name": "Student", "student_id": "GUEST",
        "branch": "Computer Science", "semester": "",
        "courses": [], "language": "English",
    })
    return jsonify(profile)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Admin: Rebuild Index
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/index/rebuild", methods=["POST"])
def rebuild_index():
    try:
        from rag_pipeline import rebuild_index as _rebuild
        _rebuild()
        return jsonify({"success": True, "message": "Index rebuilt successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def warm_up():
    """Pre-load catalog and start vector store in background on startup."""
    _load_catalog()
    _load_circulation()
    logger.info(f"Catalog loaded: {len(_catalog_cache)} books")
    try:
        _ensure_rag()
    except Exception as e:
        logger.warning(f"RAG warm-up failed (will retry on first request): {e}")


def _find_free_port(preferred: int) -> int:
    """Return `preferred` if it is free, otherwise find the next free port."""
    import socket
    for port in range(preferred, preferred + 20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
            return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found near {preferred}")


if __name__ == "__main__":
    warm_up()
    preferred = int(os.getenv("PORT", 5001))
    port      = _find_free_port(preferred)
    debug     = os.getenv("FLASK_ENV", "development") == "development"
    if port != preferred:
        logger.warning(f"Port {preferred} is in use — using port {port} instead")

    print()
    print("=" * 55)
    print(f"  Grantha Library AI Agent is running!")
    print(f"  Open this URL in your browser:")
    print(f"  --> http://127.0.0.1:{port}")
    print("=" * 55)
    print()

    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
