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

Session uploads are processed in memory. Session metadata, document chunks, Redis vectors, cached answers, and cache statistics are stored in Upstash Redis and expire after 30 minutes of inactivity. A new session or any document addition invalidates that session's cached answers; deleting a session removes its cache and statistics. The cache is scoped by session and document version, so answers cannot cross between documents. Redis-backed entries persist across backend restarts while the session remains active.

Gemini and Groq calls require valid server-side environment variables. API keys are never sent to the browser. The pre-existing `chroma-db` directory is not used or modified by this web app.

## Answer caching and performance

Session uploads are processed in memory. Session metadata, document chunks, Redis vectors, cached answers, and cache statistics are stored in Upstash Redis and expire after 30 minutes of inactivity. A new session or any document addition invalidates that session's cached answers; deleting a session removes its cache and statistics. The cache is scoped by session and document version, so answers cannot cross between documents. Redis-backed entries persist across backend restarts while the session remains active.

Exact normalized-question caching is enabled by default. Exact cache hits return the saved answer and page citations before calling Gemini embeddings, querying the Redis vector index, or calling Groq. Optional semantic caching can be enabled with `RAG_SEMANTIC_CACHE_ENABLED=true`; it uses a strict embedding threshold plus term, number, and negation checks. Semantic lookup still needs a Gemini query embedding before it can avoid vector retrieval and Groq.

The UI reports per-request `CACHE HIT` / `CACHE MISS`, elapsed time, session hit rate, avoided vector searches and LLM calls, estimated cost saved, and average hit/miss response times. Set `RAG_ESTIMATED_LLM_COST_PER_CALL_USD` to your own estimated Groq cost per answer call to calculate dollar savings; it defaults to `0` because provider pricing and token usage vary.

Each answer includes retrieved passage excerpts alongside the source and page number. Words in those excerpts that also appear in the answer are underlined and highlighted. The full retrieved excerpt remains visible for context; the highlighting is a text-overlap aid, not a separate model claim about exact word-level attribution.

On a cache miss, the existing RAG flow remains in use: query embedding, one vector search, and an LLM call when relevant chunks are found. An exact cache hit avoids those operations. A semantic hit avoids vector retrieval and LLM generation, but makes one query-embedding call. Actual response-time averages are collected by the running app; benchmark them with your documents and deployment.

Run backend cache tests from the project root with:

```powershell
py -3.12 -m unittest backend.test_cache -v
```
