import { useCallback, useEffect, useRef, useState } from "react";
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";
const CITATION_STOP_WORDS = new Set(
  "about after again all also am an and are as at be been before being but by can could did do does for from had has have he her here him his how i if in into is it its may me might more most must my no nor not of on or our out over she should so some than that the their them then there these they this those through to too under up us was we were what when where which who why will with would you your".split(" ")
);

function renderCitationExcerpt(excerpt, answer) {
  const answerTerms = new Set(
    (answer.match(/[\p{L}\p{N}]{4,}/gu) || [])
      .map((word) => word.toLowerCase())
      .filter((word) => !CITATION_STOP_WORDS.has(word))
  );
  const parts = [];
  const wordPattern = /[\p{L}\p{N}]+/gu;
  let previousIndex = 0;
  let match;

  while ((match = wordPattern.exec(excerpt)) !== null) {
    parts.push(excerpt.slice(previousIndex, match.index));
    const word = match[0];
    parts.push(
      answerTerms.has(word.toLowerCase())
        ? <mark key={match.index} className="bg-[#f7e7a8] text-inherit underline decoration-[#9a6b00] decoration-2 underline-offset-2">{word}</mark>
        : word
    );
    previousIndex = wordPattern.lastIndex;
  }

  parts.push(excerpt.slice(previousIndex));
  return parts;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error("Can't reach the server. Make sure the FastAPI backend is running.");
  }
  const payload = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.detail || `Request failed (${response.status}).`);
  }
  return payload;
}

function Icon({ name, size = 18, className = "" }) {
  const paths = {
    leaf: <><path d="M20.8 3.2C12.5 3.1 6.2 5.4 4.1 10.1c-1.5 3.3.5 6.2 3.5 6.2 5.5 0 9.5-6 11.6-11.1-1.5 5.4-4.9 10.1-10.4 13.6" /><path d="M4 20c2.1-3.7 5-6.4 8.8-8.1" /></>,
    upload: <><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M20 16.5v3a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 19.5v-3" /></>,
    file: <><path d="M13 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10z" /><path d="M13 3v7h7M8 15h8M8 18h5" /></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" /><path d="m9 12 2 2 4-4" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    close: <><path d="m18 6-12 12M6 6l12 12" /></>,
    arrow: <><path d="M7 17 17 7M7 7h10v10" /></>,
    spark: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-2-5.8L4 11l6-2.2L12 3Z" /><path d="m19 14 1.2 2.8L23 18l-2.8 1.1L19 22l-1.1-2.9L15 18l2.9-1.2L19 14Z" /></>,
  };
  return (
    <svg aria-hidden="true" className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      {paths[name]}
    </svg>
  );
}

function App() {
  const [sessionId, setSessionId] = useState("");
  const [documents, setDocuments] = useState(null);
  const [files, setFiles] = useState([]);
  const [pastedText, setPastedText] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [cacheStats, setCacheStats] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);
  const questionInput = useRef(null);

  const startSession = useCallback(async () => {
    const created = await request("/api/sessions", { method: "POST" });
    setSessionId(created.session_id);
  }, []);

  useEffect(() => {
    startSession().catch((reason) => setError(reason.message));
  }, [startSession]);

  const addFiles = async () => {
    if (!files.length || !sessionId) return;
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      files.forEach((file) => form.append("files", file));
      const result = await request(`/api/sessions/${sessionId}/documents`, { method: "POST", body: form });
      setDocuments(result);
      setFiles([]);
      if (fileInput.current) fileInput.current.value = "";
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
    }
  };

  const addText = async (event) => {
    event.preventDefault();
    if (!pastedText.trim() || !sessionId) return;
    setBusy(true);
    setError("");
    try {
      const result = await request(`/api/sessions/${sessionId}/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: pastedText }),
      });
      setDocuments(result);
      setPastedText("");
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
    }
  };

  const ask = async (event) => {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt || !sessionId || !documents) return;
    setQuestion("");
    setMessages((current) => [...current, { role: "user", text: prompt }]);
    setBusy(true);
    setError("");
    try {
      const result = await request(`/api/sessions/${sessionId}/questions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: prompt }),
      });
      setCacheStats(result.cache_stats);
      setMessages((current) => [...current, {
        role: "assistant",
        text: result.answer,
        sources: result.sources,
        citations: result.citations,
        cacheStatus: result.cache_status,
        responseTimeMs: result.response_time_ms,
      }]);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
      questionInput.current?.focus();
    }
  };

  const newSession = async () => {
    setBusy(true);
    setError("");
    try {
      if (sessionId) {
        await request(`/api/sessions/${sessionId}`, { method: "DELETE" });
      }
      setDocuments(null);
      setMessages([]);
      setCacheStats(null);
      setFiles([]);
      setPastedText("");
      await startSession();
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
    }
  };

  const ready = Boolean(documents?.document_count);

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <header className="border-b border-[#e5eae4] bg-white/80">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between px-5 py-4 sm:px-8">
          <a className="flex items-center gap-2.5" href="/" aria-label="Grounded home">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-forest text-white"><Icon name="leaf" size={21} /></span>
            <span className="font-display text-lg font-extrabold tracking-tight">grounded<span className="text-forest">.</span></span>
          </a>
          <div className="hidden items-center gap-2 rounded-full border border-[#e6ebe6] bg-white px-3 py-2 text-xs font-semibold text-muted sm:flex">
            <span className="h-2 w-2 rounded-full bg-[#56a987]" /> Private, temporary workspace
          </div>
          <button className="button-secondary !px-3 !py-2 text-sm" onClick={newSession} disabled={busy || !sessionId}>
            <Icon name="plus" size={16} /> New session
          </button>
        </div>
      </header>

      <main className="mx-auto grid min-h-[calc(100vh-73px)] max-w-[1440px] grid-cols-1 gap-6 px-4 py-5 sm:px-8 sm:py-8 lg:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="space-y-5">
          <section className="rounded-2xl border border-[#e4eae4] bg-white p-5 shadow-card">
            <div className="mb-5 flex items-start justify-between">
              <div>
                <p className="section-label">YOUR WORKSPACE</p>
                <h1 className="mt-1 font-display text-xl font-bold">Add your sources</h1>
              </div>
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#edf5f0] text-forest"><Icon name="file" /></span>
            </div>
            <div
              className="upload-zone cursor-pointer rounded-xl p-5 text-center transition hover:border-forest hover:bg-[#f7fbf8]"
              onClick={() => fileInput.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                setFiles(Array.from(event.dataTransfer.files).filter((file) => /\.(pdf|txt)$/i.test(file.name)));
              }}
            >
              <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-full bg-[#eaf3ed] text-forest"><Icon name="upload" size={20} /></div>
              <p className="mt-3 text-sm font-semibold">Drop files here or <span className="text-forest">browse</span></p>
              <p className="mt-1 text-xs text-muted">PDF or TXT · up to 10 MB each</p>
              <input
                ref={fileInput}
                className="sr-only"
                type="file"
                accept=".pdf,.txt,application/pdf,text/plain"
                multiple
                onChange={(event) => setFiles(Array.from(event.target.files || []))}
              />
            </div>
            {files.length > 0 && (
              <div className="mt-3 space-y-2">
                {files.map((file) => <div className="flex items-center gap-2 rounded-lg bg-[#f5f7f4] px-3 py-2 text-xs" key={`${file.name}-${file.lastModified}`}><Icon name="file" size={15} className="shrink-0 text-forest" /><span className="truncate">{file.name}</span></div>)}
                <button className="button-primary w-full" disabled={busy || !sessionId} onClick={addFiles}>{busy ? "Processing…" : "Add selected files"}</button>
              </div>
            )}

            <div className="my-5 flex items-center gap-3 text-[10px] font-bold uppercase tracking-[0.16em] text-[#9aa5a0]"><span className="h-px flex-1 bg-[#e8ece8]" />or paste text<span className="h-px flex-1 bg-[#e8ece8]" /></div>
            <form onSubmit={addText}>
              <label className="sr-only" htmlFor="pasted-text">Paste your text</label>
              <textarea id="pasted-text" className="text-area min-h-[112px] w-full resize-y" maxLength={500000} placeholder="Paste notes, an article, or any text you want to explore…" value={pastedText} onChange={(event) => setPastedText(event.target.value)} />
              <button className="button-secondary mt-2 w-full justify-center" type="submit" disabled={busy || !sessionId || !pastedText.trim()}>{busy ? "Processing…" : "Add pasted text"}</button>
            </form>
          </section>

          <section className="rounded-2xl border border-[#e4eae4] bg-white p-5 shadow-card">
            <div className="flex items-center justify-between">
              <div><p className="section-label">IN THIS SESSION</p><h2 className="mt-1 font-display text-base font-bold">Your documents</h2></div>
              <span className="rounded-full bg-[#f1f4f0] px-2.5 py-1 text-xs font-bold text-muted">{documents?.document_count || 0}</span>
            </div>
            {documents?.sources?.length ? (
              <ul className="mt-4 space-y-2">
                {documents.sources.map((source) => <li className="flex min-w-0 items-center gap-2 rounded-lg bg-[#f7f8f6] px-3 py-2.5 text-xs font-medium" key={source}><Icon name="file" size={15} className="shrink-0 text-forest" /><span className="truncate">{source}</span></li>)}
              </ul>
            ) : <p className="mt-3 rounded-lg bg-[#f7f8f6] px-3 py-3 text-xs leading-5 text-muted">Your added files will appear here. Nothing is saved between sessions.</p>}
            {documents?.chunk_count > 0 && <p className="mt-3 text-[11px] text-muted">{documents.chunk_count} searchable passages</p>}
          </section>

          <div className="flex gap-3 rounded-xl border border-[#e0e9e1] bg-[#eff5ef] p-4">
            <Icon name="shield" size={19} className="mt-0.5 shrink-0 text-forest" />
            <p className="text-xs leading-[1.65] text-[#52615a]"><strong className="font-semibold text-ink">Private by design.</strong> Your documents stay isolated to this session and are permanently deleted after 30 minutes of inactivity.</p>
          </div>
        </aside>

        <section className="flex min-h-[560px] flex-col overflow-hidden rounded-2xl border border-[#e4eae4] bg-white shadow-card lg:min-h-0">
          <div className="flex items-center justify-between border-b border-[#edf0ec] px-5 py-4 sm:px-7">
            <div><p className="section-label">DOCUMENT ASSISTANT</p><h2 className="mt-1 font-display text-lg font-bold">Ask your documents</h2></div>
            <div className="flex items-center gap-1.5 rounded-full bg-[#f4f6f3] px-3 py-1.5 text-[11px] font-semibold text-muted"><Icon name="clock" size={14} /> Auto-delete · 30 min idle</div>
          </div>
          {cacheStats && (
            <div className="border-b border-[#edf0ec] px-5 py-2 text-[11px] leading-5 text-muted sm:px-7" aria-live="polite">
              Cache: {cacheStats.total_cache_hits} hits · {cacheStats.total_cache_misses} misses ·
              {` ${cacheStats.avoided_llm_calls} LLM calls / ${cacheStats.avoided_vector_searches} searches avoided · `}
              {cacheStats.cache_hit_rate.toFixed(1)}% hit rate ·
              {` est. $${cacheStats.estimated_api_cost_saved_usd.toFixed(6)} saved · `}
              avg {cacheStats.average_cache_hit_response_ms.toFixed(0)} ms hit / {cacheStats.average_cache_miss_response_ms.toFixed(0)} ms miss
            </div>
          )}

          <div className="flex flex-1 flex-col overflow-y-auto p-5 sm:p-8">
            {messages.length === 0 ? (
              <div className="m-auto max-w-md py-10 text-center">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-[#edf5f0] text-forest"><Icon name="spark" size={25} /></div>
                <p className="section-label mt-5">A little less searching</p>
                <h3 className="mt-2 font-display text-2xl font-bold tracking-tight">{ready ? "Your sources are ready." : "Bring your knowledge together."}</h3>
                <p className="mx-auto mt-3 max-w-sm text-sm leading-6 text-muted">{ready ? "Ask a question and get an answer grounded only in the documents you added." : "Upload a PDF, add a text file, or paste text. Then ask anything about what you shared."}</p>
                {ready && <button className="suggestion-chip mt-6" onClick={() => { setQuestion("What are the main ideas?"); questionInput.current?.focus(); }}>What are the main ideas? <Icon name="arrow" size={14} /></button>}
              </div>
            ) : (
              <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
                {messages.map((message, index) => (
                  <article className={message.role === "user" ? "ml-auto max-w-[88%]" : "max-w-[94%]"} key={`${message.role}-${index}`}>
                    <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.14em] text-[#8b9791]">{message.role === "user" ? "You" : "Grounded"}</p>
                    {message.cacheStatus && <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.14em] text-forest">{message.cacheStatus} · {message.responseTimeMs} ms</p>}
                    <div className={message.role === "user" ? "rounded-2xl rounded-tr-sm bg-forest px-4 py-3 text-sm leading-6 text-white" : "rounded-2xl rounded-tl-sm bg-[#f4f6f3] px-4 py-3 text-sm leading-6 text-ink"}>
                      <p className="whitespace-pre-wrap">{message.text}</p>
                      {message.citations?.length > 0 ? (
                        <div className="mt-3 space-y-3 border-t border-[#dfe5df] pt-3">
                          <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted">Retrieved passages · matching answer words underlined</p>
                          {message.citations.map((citation, citationIndex) => (
                            <figure key={`${citation.source}-${citation.page}-${citationIndex}`}>
                              <figcaption className="mb-1 text-[11px] font-semibold text-forest">
                                {citation.source}{citation.page != null ? ` · page ${citation.page}` : ""}
                              </figcaption>
                              <blockquote className="whitespace-pre-wrap border-l-2 border-[#d9c99a] pl-3 text-xs leading-5 text-[#52615a]">
                                {renderCitationExcerpt(citation.excerpt, message.text)}
                              </blockquote>
                            </figure>
                          ))}
                        </div>
                      ) : message.sources?.length > 0 && (
                        <p className="mt-3 border-t border-[#dfe5df] pt-2 text-[11px] text-muted">
                          Sources: {message.sources.join(", ")}
                        </p>
                      )}
                    </div>
                  </article>
                ))}
                {busy && <div className="flex items-center gap-2 text-xs text-muted"><span className="typing-dot" /><span className="typing-dot animation-delay-150" /><span className="typing-dot animation-delay-300" /> Checking your sources…</div>}
              </div>
            )}
          </div>

          {error && <div className="mx-5 mb-3 rounded-lg border border-[#f1d5ce] bg-[#fff6f3] px-4 py-3 text-sm text-[#9a4737] sm:mx-8" role="alert">{error}</div>}
          <form className="border-t border-[#edf0ec] bg-white p-4 sm:px-7 sm:py-5" onSubmit={ask}>
            <div className="mx-auto flex max-w-3xl items-end gap-3 rounded-xl border border-[#e1e7e1] bg-[#fafbf9] p-2 pl-4 focus-within:border-[#8ab5a0] focus-within:ring-2 focus-within:ring-[#e2f0e7]">
              <label className="sr-only" htmlFor="question">Ask a question about your documents</label>
              <textarea
                ref={questionInput}
                id="question"
                className="max-h-32 min-h-10 flex-1 resize-none bg-transparent py-2 text-sm leading-5 outline-none placeholder:text-[#a1aaa5] disabled:cursor-not-allowed"
                placeholder={ready ? "Ask a question about your sources…" : "Add a document to start asking"}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form.requestSubmit();
                  }
                }}
                disabled={!ready || busy}
              />
              <button className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-forest text-white transition hover:bg-[#105b46] disabled:cursor-not-allowed disabled:bg-[#ccd5ce]" type="submit" aria-label="Send question" disabled={!ready || busy || !question.trim()}><Icon name="send" size={17} /></button>
            </div>
            <p className="mt-2 text-center text-[10px] text-[#9aa39e]">Answers are generated from your sources only. Anything missing? Just ask.</p>
          </form>
        </section>
      </main>
    </div>
  );
}

export default App;
