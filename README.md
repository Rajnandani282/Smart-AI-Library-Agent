# Library AI Agent — Deployment & Setup Guide

> Library AI Agent is an AI-powered library assistant developed to help students find books, explore course syllabi, and get useful academic recommendations through a simple chat interface.

The project combines Retrieval-Augmented Generation (RAG) with IBM watsonx.ai to search information from the library knowledge base and provide relevant responses. It also allows users to search books, check availability, view recommended reading lists, and access department resources.

The application is built using Python, Flask, IBM watsonx.ai, ChromaDB/FAISS, and Bootstrap, making it easy to run locally as well as deploy on cloud platforms.

This project is designed to make library services faster, smarter, and more convenient for students by providing information through natural language conversations.

---

## Project Structure

```
library_agent/
│
├── app.py                        ← Flask backend (all API endpoints)
├── rag_pipeline.py               ← RAG pipeline: embed + store + retrieve
├── agent_instructions.py         ← ★ Edit here to customise agent behaviour
│
├── .env.example                  ← Template — copy to .env
├── requirements.txt
│
├── knowledge_base/
│   ├── book_catalog/
│   │   ├── catalog.csv           ← 40-book sample catalog (extend as needed)
│   │   └── circulation_data.json ← Borrows, reservations, trending data
│   ├── syllabi/
│   │   ├── CS301_Data_Structures_Algorithms.txt
│   │   ├── CS401_Computer_Networks.txt
│   │   ├── CS501_Machine_Learning.txt
│   │   └── CS601_Artificial_Intelligence.txt
│   ├── reading_lists/
│   │   └── semester_reading_lists.txt
│   └── department_guides/
│       ├── CS_Department_Resource_Guide.txt
│       └── EE_Department_Resource_Guide.txt
│
├── templates/
│   └── index.html                ← Single-page frontend
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
└── vector_store/                 ← Auto-created on first run
    ├── chroma_db/                ← ChromaDB persisted index
    └── faiss_index.*             ← FAISS index files
```

---

## 1 · Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.11 |
| pip | 23+ |
| IBM Cloud Account | Free tier works |
| watsonx.ai Project | Required for real LLM |

---

## 2 · IBM Cloud Setup

### 2.1 Get an API Key
1. Log in to **[cloud.ibm.com](https://cloud.ibm.com)**
2. Go to **Manage → Access (IAM) → API keys**
3. Click **Create an IBM Cloud API key** → copy the key

### 2.2 Create a watsonx.ai Project
1. Go to **[watsonx.ai](https://dataplatform.cloud.ibm.com/wx/home)**
2. Click **New project → Create an empty project**
3. Open the project → **Manage tab → General** → copy the **Project ID**

### 2.3 Find your Region URL
| IBM Cloud Region | watsonx URL |
|---|---|
| Dallas (us-south) | `https://us-south.ml.cloud.ibm.com` |
| Frankfurt (eu-de) | `https://eu-de.ml.cloud.ibm.com` |
| Tokyo (jp-tok) | `https://jp-tok.ml.cloud.ibm.com` |
| London (eu-gb) | `https://eu-gb.ml.cloud.ibm.com` |

---

## 3 · Local Development

### 3.1 Clone / navigate to project
```bash
cd library_agent
```

### 3.2 Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3.3 Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First install downloads ~400 MB for sentence-transformers model weights.
> Subsequent runs use the cached model.

### 3.4 Configure environment
```bash
cp .env.example .env
```

Edit `.env` and fill in:
```env
IBM_API_KEY=your_actual_api_key
WATSONX_PROJECT_ID=your_actual_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=<run: python -c "import secrets;print(secrets.token_hex(32))">
```

### 3.5 Run the application
```bash
python app.py
```

On first run the app will:
1. Load all catalog CSV and text files
2. Split documents into chunks
3. Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`
4. Persist the ChromaDB index to `./vector_store/chroma_db/`

This takes **30–90 seconds** on first run. Subsequent starts are instant.

Visit **[http://localhost:5000](http://localhost:5000)**

---

## 4 · Demo Mode (no IBM credentials)

If `IBM_API_KEY` is not set, the agent runs in **demo mode**:
- RAG retrieval still works fully
- Responses are generated from catalog context (rule-based, not LLM)
- A `*(Demo mode)*` note is appended to every response

This lets you evaluate the UI and RAG pipeline without an IBM account.

---

## 5 · Customising the Agent

Open **`agent_instructions.py`** — every section is clearly labelled:

| Section | What to change |
|---|---|
| `AGENT_NAME` | Change the bot's name (default: *Grantha*) |
| `COMMUNICATION_STYLE` | `formal` / `friendly-academic` / `casual` |
| `PRIMARY_DOMAIN` | `engineering` / `sciences` / `humanities` / `all` |
| `TONE_INSTRUCTIONS` | Free-text tone rules |
| `SPECIALIZATION_INSTRUCTIONS` | Domain-specific richness rules |
| `SAFETY_INSTRUCTIONS` | Disclaimers, safety guardrails |
| `LANGUAGE_INSTRUCTIONS` | Supported regional languages & behavior |
| `RAG_INSTRUCTIONS` | How the agent uses retrieved context |

No other file needs to be touched for behaviour changes.

---

## 6 · Adding Books to the Catalog

### Option A — Edit catalog.csv directly
Add rows to `knowledge_base/book_catalog/catalog.csv` following the header:
```
book_id,title,author,isbn,publisher,year,edition,department,subject,
copies_total,copies_available,shelf_location,language,description,tags
```

### Option B — Add syllabus/guide text files
Drop `.txt` files into:
- `knowledge_base/syllabi/`
- `knowledge_base/department_guides/`
- `knowledge_base/reading_lists/`

### Rebuild the index after adding documents
```bash
# Via API (POST request)
curl -X POST http://localhost:5000/api/index/rebuild

# Or restart the app and delete the old index
rm -rf vector_store/
python app.py
```

---

## 7 · Switching to FAISS

```env
VECTOR_STORE=faiss
```

FAISS is faster for similarity search at scale but does not persist metadata.
ChromaDB (default) is recommended for most deployments.

---

## 8 · Changing the LLM Model

Edit `.env`:
```env
# Granite models available on watsonx.ai
LLM_MODEL_ID=ibm/granite-13b-chat-v2      # default — balanced
LLM_MODEL_ID=ibm/granite-20b-multilingual # better for Hindi/Marathi
LLM_MODEL_ID=ibm/granite-3-8b-instruct    # faster, smaller
```

For multilingual queries in Hindi/Marathi, `granite-20b-multilingual` gives
better results.

---

## 9 · Production Deployment

### 9.1 Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app --timeout 120
```

### 9.2 Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app", "--timeout", "120"]
```

```bash
docker build -t library-agent .
docker run -p 5000:5000 --env-file .env library-agent
```

### 9.3 IBM Code Engine (Serverless)
```bash
ibmcloud ce project create --name library-agent-project
ibmcloud ce app create \
  --name library-agent \
  --image icr.io/yournamespace/library-agent:latest \
  --port 5000 \
  --env-from-secret lib-agent-secrets
```

---

## 10 · API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | RAG + LLM chat query |
| `GET` | `/api/books/search?q=&department=&available_only=` | Full-text book search |
| `GET` | `/api/books/<book_id>` | Book detail + live availability |
| `POST` | `/api/reserve` | Place reservation / join waitlist |
| `GET` | `/api/waitlist/<student_id>` | Student's borrows & holds |
| `GET` | `/api/trending?department=` | High-demand books |
| `GET` | `/api/recommendations?branch=&semester=&courses=` | Personalised recommendations |
| `GET/POST` | `/api/student/profile` | Get / save student profile |
| `POST` | `/api/index/rebuild` | Rebuild vector store (admin) |

### Chat request body
```json
{
  "message": "Suggest books for my Machine Learning course",
  "student_profile": {
    "name": "Priya Sharma",
    "branch": "Computer Science",
    "semester": "5th",
    "courses": ["Machine Learning", "Data Science"],
    "language": "English"
  }
}
```

---

## 11 · Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: chromadb` | Run `pip install -r requirements.txt` in venv |
| `Authentication error` from watsonx | Verify `IBM_API_KEY` and `WATSONX_PROJECT_ID` in `.env` |
| First run very slow | Normal — embedding model (~90 MB) downloads and indexes on first run |
| Chat returns "Demo mode" | Add valid `IBM_API_KEY` and `WATSONX_PROJECT_ID` to `.env` |
| `Address already in use` | Run `fuser -k 5000/tcp` (Linux) or change `PORT=5001` in `.env` |
| FAISS index out of date | Delete `vector_store/` and restart, or call `/api/index/rebuild` |
| Hindi/Marathi responses in English | Switch to `ibm/granite-20b-multilingual` in `.env` |

---

## 12 · Security Notes

- **Never commit `.env`** — it is in `.gitignore` by default
- Rotate `FLASK_SECRET_KEY` before production deployment
- The `/api/index/rebuild` endpoint should be protected by an admin token in production
- IBM API keys can be scoped to specific services in IAM for least-privilege access

---

*Made with IBM watsonx.ai + Granite · Library AI Agent*
