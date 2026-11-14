# Grounded RAG

A temporary, session-isolated document Q&A app built with React, Tailwind CSS, FastAPI, LangChain, Gemini embeddings, and Groq.

## Requirements

- Python 3.12
- Node.js 18 or newer
- Google Gemini API key and Groq API key

## Configure API keys

From the project folder, create a local environment file without overwriting an existing one:

```powershell
Set-Location 'C:\Users\rajma\Desktop\GENERATIVE AI\RAG PROJECT'
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set `GOOGLE_API_KEY` and `GROQ_API_KEY` in `.env`. The existing Gemini embedding model is `gemini-embedding-001`; the Groq model defaults to `openai/gpt-oss-20b`. To override the model, set `GROQ_MODEL` in `.env`. Keep `.env` local; it is ignored by Git.

## Install and run the backend

In a PowerShell terminal:

```powershell
Set-Location 'C:\Users\rajma\Desktop\GENERATIVE AI\RAG PROJECT'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

If PowerShell blocks virtual-environment activation, run this once in that terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The API and interactive docs are at `http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`.

## Install and run the frontend

In a second PowerShell terminal:

```powershell
Set-Location 'C:\Users\rajma\Desktop\GENERATIVE AI\RAG PROJECT\frontend'
npm install
npm run dev
```

Open the local URL Vite prints (normally `http://127.0.0.1:5173`). Vite forwards `/api` requests to the backend on port 8000.

## Use

Start a session, then add one or more PDF/TXT files or paste text. Ask questions in the chat panel. The retriever searches only the current session's Chroma collection; a new session starts with no documents. There is no login and the UI does not save chat history.

Uploads are limited to 10 MB per file and 25 MB per batch. TXT files must be UTF-8; PDFs must contain extractable text. Pasted text is limited to 500,000 characters and extracted text per document to 1,000,000 characters.

Each session's uploads, extracted text, and Chroma database are stored under `.rag_sessions\session-<id>`. A background cleanup removes them after 30 minutes without session activity (checked every 60 seconds); explicitly starting a new session removes the previous one immediately. The data is also removed when the backend shuts down, and abandoned session directories are purged when it starts. Keep one backend process/worker: session state is intentionally in memory and is not shared between workers.

Gemini and Groq calls require valid server-side environment variables. API keys are never sent to the browser. The pre-existing `chroma-db` directory is not used or modified by this web app.
