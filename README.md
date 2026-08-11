# ⚖️ Pakistan Law Assistant

> **RAG-based legal information assistant** — answers questions using real
> Pakistani legal documents (Constitution, Penal Code, Contract Act, PECA).
> Educational use only. **Not a substitute for professional legal advice.**

---

## 🏗️ Architecture

```
User Question
     │
     ▼
Streamlit Frontend (port 8501)
     │  HTTP POST /api/v1/ask
     ▼
FastAPI Backend (port 8000)
     │
     ├─► LawRetriever (LangChain Retriever over ChromaDB)
     │       │
     │       ├─► HuggingFace sentence-transformers (embed query)
     │       └─► ChromaDB (cosine similarity search)
     │
     ├─► Format retrieved chunks with citations
     │
     └─► Groq LLM (llama-3.3-70b-versatile, grounded generation)
              │
              ▼
         Answer + Citations + Legal Disclaimer
```

**Pipeline exactly as required:** LangChain `PyPDFLoader` → LangChain
`RecursiveCharacterTextSplitter` → HuggingFace embeddings → ChromaDB →
LangChain Retriever → Groq LLM → grounded, cited answer.

---

## 🚀 Quick Start

### 1. Get the source documents (free, official, direct downloads)

| Document | Source |
|---|---|
| Constitution of Pakistan | https://www.na.gov.pk/uploads/documents/1549886415_632.pdf |
| Pakistan Penal Code | https://www.unodc.org/cld/uploads/res/document/pak/1860/pakistan_penal_code_1860_html/Pakistan_Penal_Code_1860_incorporating_amendments_to_16_February_2017.pdf |
| Contract Act 1872 | https://pakistancode.gov.pk/english/ (search "Contract Act") |
| PECA 2016 | https://pakistancode.gov.pk/english/ (search "PECA" / "Prevention of Electronic Crimes Act") |

Download each PDF and rename/place it into `data/raw/` as:
```
data/raw/constitution.pdf
data/raw/ppc.pdf
data/raw/contract_act.pdf
data/raw/peca.pdf
```
(You don't need all four to start — ingestion skips any missing file with a warning.)

### 2. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```
Edit `.env` and set your `GROQ_API_KEY` (get one free at https://console.groq.com/keys).

### 4. Ingest the documents

```bash
python ingestion/ingest.py
# python ingestion/ingest.py --force     # to re-ingest from scratch
```

### 5. Run the app

**Terminal 1 — Backend:**
```bash
python start_backend.py
# API docs: http://localhost:8000/docs
```

**Terminal 2 — Frontend:**
```bash
python start_frontend.py
# App: http://localhost:8501
```

---

## 📁 Project Structure

```
pakistan-law-rag/
├── api/                     # FastAPI backend
│   ├── main.py               # App factory, CORS, lifespan
│   ├── routes.py             # /ask, /search, /health endpoints
│   └── models.py             # Pydantic request/response models
│
├── ingestion/               # Data ingestion pipeline
│   ├── loader.py              # LangChain PyPDFLoader wrapper
│   ├── splitter.py             # LangChain RecursiveCharacterTextSplitter wrapper
│   └── ingest.py                # CLI entry point (load -> split -> embed -> store)
│
├── retrieval/                # RAG pipeline
│   ├── retriever.py            # LangChain Retriever + scored chunk wrapper
│   └── rag_chain.py             # LCEL chain: retrieval + Groq generation
│
├── frontend/                # Streamlit UI
│   └── app.py                  # Chat + Search tabs, calls the FastAPI backend
│
├── prompts/
│   └── system_prompt.txt        # Grounded-answer + disclaimer instructions
│
├── utils/
│   ├── config.py                # Pydantic settings
│   └── logger.py                 # Loguru structured logging
│
├── data/raw/                # Place downloaded PDFs here (gitignored)
├── chroma_db/                # ChromaDB persistence (auto-created, gitignored)
├── .env.example
├── requirements.txt
├── runtime.txt               # Pins Python 3.11 for cloud deploys
├── start_backend.py
└── start_frontend.py
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | System health, chunk count, loaded documents |
| POST | `/api/v1/ask` | Ask a legal question — returns grounded answer + citations |
| POST | `/api/v1/search` | Raw semantic search — no LLM, just ranked source chunks |

Full interactive docs: **http://localhost:8000/docs**

**Example `/ask` request:**
```json
{
  "question": "What does the Constitution say about freedom of speech?",
  "top_k": 5,
  "source_filter": null
}
```

**Example response:**
```json
{
  "answer": "According to the Constitution... (Source: Constitution of Pakistan, p.12) ... ⚠️ This response is for educational and informational purposes only...",
  "sources": [
    {"citation": "Constitution of Pakistan, p.12", "confidence": 82, "confidence_label": "Very High", ...}
  ],
  "disclaimer": "This response is for educational and informational purposes only..."
}
```

---

## 🛡️ Grounding & Safety Design

The system prompt (`prompts/system_prompt.txt`) enforces:
1. Answer **only** from retrieved context — never from the model's general knowledge.
2. Explicitly state when the documents don't contain enough information.
3. Cite the exact document + page for every substantive claim.
4. Never fabricate section/article/page numbers.
5. **Always** append the legal disclaimer, verbatim, to every response.

---

## ☁️ Deployment

### Backend (Render, Railway, Koyeb, or similar)
- Build command: `pip install -r requirements.txt`
- Start command: `python start_backend.py`
- Environment variables: `GROQ_API_KEY`, `GROQ_MODEL`, `EMBEDDING_MODEL`, `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION_NAME`, `TOP_K_RESULTS`, `SIMILARITY_THRESHOLD`
- **Important:** ChromaDB storage is ephemeral on most free tiers — either commit a pre-built `chroma_db/` (not recommended, large) or re-run `python ingestion/ingest.py` as part of your start command / a startup hook so the app self-seeds on each cold start.

### Frontend (Streamlit Community Cloud)
- Main file path: `frontend/app.py`
- Secrets: set `API_BASE_URL` to your deployed backend's URL, e.g. `https://your-backend.onrender.com/api/v1`
- No GPU/heavy embedding model runs on the frontend here — it only calls the backend over HTTP, so this deploys light and fast.

---

## 📜 Data Sources

- **Constitution of Pakistan (1973)** — National Assembly of Pakistan, official publication.
- **Pakistan Penal Code (1860)** — UNODC mirror, amended to Feb 2017.
- **Contract Act (1872)** and **PECA (2016)** — The Pakistan Code, Ministry of Law and Justice.

All sources are publicly available official government legislation.

---

*Educational project — not legal advice. Always consult a licensed lawyer for real legal matters.*
