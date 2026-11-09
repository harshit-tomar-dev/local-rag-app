# 🎓 StudyAI • ChatGPT & Claude RAG Assistant
### *A Production-Grade, Multi-Provider AI & Student Study Platform with Grounded Citations*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.162+-009688.svg)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/Tests-32%20Passing-brightgreen.svg)](https://pytest.org)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI%20%7C%20Claude%20%7C%20Gemini%20%7C%20Groq%20%7C%20Ollama-orange.svg)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Created by **Harsh & Bhavishya**, **StudyAI** is an advanced, privacy-first Retrieval-Augmented Generation (RAG) web application and student study assistant. It brings the sleek conversational experience of **ChatGPT (OpenAI)** together with support for **Anthropic Claude**, **Google Gemini**, **Groq**, **Ollama**, and a **100% offline Extractive QA engine**.

Students and researchers can upload textbooks, lecture slides, research papers, notes, or datasets (`.pdf`, `.docx`, `.xlsx`, `.md`, `.csv`, `.txt`) and instantly get **executive study summaries**, **interactive 3D flashcards**, **auto-graded practice quizzes**, and **chat with exact page-level citations**.

---

## 🌟 Key Features

### 1. 🤖 Universal Multi-Provider AI (BYOK - Bring Your Own Key)
Switch models on the fly directly from the ChatGPT-style header dropdown:
- **OpenAI (ChatGPT)**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`, `o1-mini`
- **Anthropic Claude**: `claude-3-7-sonnet`, `claude-3-5-sonnet`, `claude-3-5-haiku`, `claude-3-opus`
- **Google Gemini**: `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`
- **Groq**: Ultra-fast inference with `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b`
- **Ollama (Local AI)**: 100% private offline models (`llama3.2`, `mistral`, `deepseek-r1`, `phi3`)
- **Extractive Engine**: Offline direct semantic extraction that runs out-of-the-box with **zero API keys and zero server requirements**.

> 💡 **In-Browser Key Management:** Users can paste their API keys directly into the built-in **Settings Modal**. Keys are securely kept in browser storage and validated on-the-spot with the "Test Key" button!

---

### 2. 🎓 Student Study Suite
Built specifically to accelerate exam preparation and deep document comprehension:
- 📑 **Comprehensive Study Summary**: Automatically extracts executive summaries, core concepts, bulleted takeaways, and a glossary of key definitions.
- 💡 **Interactive 3D Flashcards**: Generates high-yield active-recall question/answer card decks with smooth click-to-flip animations.
- 🎯 **Interactive Practice Mock Test**: Multiple-Choice Questions (MCQs) with instant answer verification (green/red feedback), score counter, and grounded document explanations.
- 📝 **Predicted Exam Questions**: Generates top exam questions with model answers, grading checklists, and common pitfalls to avoid.
- ✨ **Explain Like I'm 5 (ELI5)**: Breaks down the hardest formulas or theories into intuitive everyday analogies.

---

### 3. 🎨 Authentic OpenAI / ChatGPT Experience
- **Real-Time Token Streaming**: Real-time Server-Sent Events (SSE) streaming token-by-token.
- **Rich Markdown & Code**: Headers, tables, bulleted lists, and syntax-highlighted code blocks with a one-click "Copy Code" button.
- **Clickable Page Citations**: Every answer links directly to its source chunk `[📄 Biology.pdf • Page 14]`. Clicking a citation opens a slide-over drawer showing the exact source text and similarity score.
- **Direct Composer Attachments**: Upload PDFs or notes directly from the input bar with the 📎 paperclip button.
- **Dark & Light Mode**: Clean, distraction-free aesthetic matching OpenAI.

---

### 4. ⚡ Core RAG & Ingestion Engine
- **Multi-Format Extraction**:
  - `.pdf`: PyMuPDF with exact page preservation, handling encrypted or multi-page documents.
  - `.docx`: Paragraphs and table extraction with structure.
  - `.xlsx` & `.csv`: Sheet and tabular data parsing.
  - `.txt` & `.md`: UTF-8 text and markdown parsing.
- **Sliding-Window Chunking**: Context preservation across page boundaries.
- **Okapi BM25 & Hybrid Vector Retrieval**: Reciprocal Rank Fusion (`RRF`) for optimal precision and recall.
- **SQLite Database Persistence**: Full WAL mode persistence for knowledge bases, documents, conversations, and messages.

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/harshit-tomar-dev/local-rag-app.git
cd local-rag-app
```

### 2. Create Virtual Environment
#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🏃 Running the Application

Start the local server with hot-reload enabled:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

1. Open your browser at **`http://127.0.0.1:8000`** to access the StudyAI Web App.
2. Open interactive Swagger API documentation at **`http://127.0.0.1:8000/docs`**.

---

## 🔑 Configuring AI Keys

You can configure your AI keys in either of two ways:

### Method A: Directly in the Web UI (Recommended)
1. Open the web app at `http://127.0.0.1:8000`.
2. Click the **⚙️ Settings** icon in the sidebar or bottom left.
3. Paste your OpenAI, Claude, Gemini, or Groq API key.
4. Click **Test** to verify connection, then **Save Keys**.

### Method B: Via `.env` File
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your desired keys:
```env
# Active provider: openai | claude | gemini | groq | ollama | extractive
LLM_PROVIDER=extractive
LLM_MODEL=extractive

OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Local Ollama endpoint (optional for local LLMs)
LLM_BASE_URL=http://localhost:11434
```

---

## 🧪 Running the Test Suite

Run all 32 unit and integration tests covering extraction, hybrid search, multi-provider LLMs, and student study tools:

```bash
python run_tests.py
```
Or directly with pytest:
```bash
pytest -v
```

---

## 📁 Project Architecture

```
local-rag-app/
├── app/
│   ├── config.py             # Settings, environment variables & key resolution
│   ├── database.py           # SQLite persistence, WAL mode, transactional commit
│   ├── llm.py                # Multi-provider client (OpenAI, Claude, Gemini, Groq, Ollama)
│   ├── main.py               # FastAPI endpoints, SSE streaming, CORS & static files
│   ├── schemas.py            # Pydantic validation schemas
│   ├── services.py           # Knowledge Base management & RAG query orchestration
│   ├── study.py              # Student Study Suite: Summaries, Flashcards, Quizzes, ELI5
│   └── static/
│       └── index.html        # ChatGPT-style Web UI (Dark/Light, Flashcards, Quiz, Citations)
├── data/
│   ├── rag.db                # SQLite database (auto-created)
│   └── uploads/              # Ingested source files
├── eval/
│   ├── benchmark.py          # Recall@k, MRR, citation evaluation suite
│   └── run.py                # Benchmark runner
├── sample_docs/              # Reference test files (.txt, .docx, .pdf)
├── tests/
│   ├── test_api.py           # REST endpoints, isolation, and error handling
│   ├── test_core.py          # Extraction, chunking, and lexical search
│   ├── test_hybrid.py        # Vector similarity & Reciprocal Rank Fusion (RRF)
│   ├── test_pdf.py           # Multi-page PDF extraction and edge cases
│   ├── test_persistence.py   # SQLite persistence & filename sanitization
│   └── test_study_and_providers.py # Provider listing, key testing, study generation
├── ragcore.py                # Core engine: BM25, sliding-window chunking, extraction
├── run_tests.py              # Test runner script
├── verify_app_live.py        # End-to-end verification script
├── requirements.txt          # Pinned project dependencies
├── .env.example              # Configuration template
└── README.md                 # Project documentation
```

---

## 🛠️ API Reference Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check & active model status |
| `GET` | `/api/settings/providers` | List available providers and model options |
| `POST` | `/api/settings/verify-key` | Test connection for an API key |
| `POST` | `/api/study/generate` | Generate Summary, Flashcards, Mock Quiz, or ELI5 |
| `POST` | `/conversations/{id}/stream` | SSE real-time streaming chat endpoint |
| `POST` | `/conversations/{id}/messages` | Standard non-streaming chat endpoint |
| `POST` | `/kbs/{kb_id}/documents` | Multi-file document upload (`.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.xlsx`) |
| `GET` | `/kbs` & `/kbs/{id}` | Manage knowledge bases |
| `GET` | `/documents/{id}/chunks/{chunk_id}` | Source citation inspection |

---

## 👥 Authors & Acknowledgments

- **Harsh** & **Bhavishya** — Concept, Architecture, Full-Stack Development & Study Tool Design.
- Inspired by OpenAI ChatGPT, Claude by Anthropic, and modern local RAG technologies.

---
⭐ **Star this repository on GitHub if you find it helpful!**
