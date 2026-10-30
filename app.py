import os
import re
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Internal Runbook Search Portal")
PDF_PATH = "runbook.pdf"

def load_and_chunk_pdf(path: str):
    """Reads the PDF and breaks it down. Falls back to demo text if PDF is corrupt/missing."""
    demo_fallback_text = """
    [RUN-01] Database Connection Pool Exhausted
    Symptoms: Applications throwing FATAL: remaining connection slots are reserved. HTTP 500 errors.
    Resolution: Run pg_activity. Run query: SELECT pid, query, state FROM pg_stat_activity WHERE state != 'idle'; Kill zombie sessions using pg_terminate_backend(pid).

    [RUN-02] API Gateway 504 Bad Gateway Error
    Symptoms: Load balancer health checks failing. Users receiving HTTP 504 Gateway Timeout. Nginx upstream timed out logs.
    Resolution: Check logs using tail -n 100 /var/log/nginx/access.log | grep "504". Restart service deployment with kubectl rollout restart deployment/billing-service.

    [RUN-03] User Authentication & JWT Failures
    Symptoms: Session dropouts, 401 Unauthorized errors, Key fetch failed from JWKS endpoint.
    Resolution: Open Redis cache console. Clear token database via redis-cli -h auth-cache.internal flushdb. Force auth gateway sync.

    [RUN-04] Disk Space Alert - Log Rotation Fix
    Symptoms: Disk Space Critical alerts on /var/log usage > 95%. Files cannot be written.
    Resolution: SSH into host. Check space using df -h && du -sh /var/log/*. Truncate old active logs via truncate -s 0 /var/log/nginx/error.log.1. Restart logrotate daemon.
    """
    
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print("⚠️ runbook.pdf not found or empty. Loading built-in demo text data instead.")
        return [c.strip() for c in demo_fallback_text.strip().split('\n\n') if c.strip()]

    try:
        reader = PdfReader(path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        chunks = re.split(r'\n\s*\n', full_text)
        return [c.strip() for c in chunks if len(c.strip()) > 30]
    except Exception as e:
        print(f"⚠️ PDF Read Error ({str(e)}). Switching safely to built-in demo text data.")
        return [c.strip() for c in demo_fallback_text.strip().split('\n\n') if c.strip()]

# Initialize data and index the PDF on server startup
RUNBOOK_CHUNKS = load_and_chunk_pdf(PDF_PATH)

# Build the vectorizer index safely
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(RUNBOOK_CHUNKS)

class SearchQuery(BaseModel):
    query: str

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Internal Runbook Search Portal</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; padding: 40px; margin: 0; }
        .box { max-width: 750px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        h2 { margin-top: 0; color: #1e293b; font-size: 24px; }
        .search-container { display: flex; gap: 10px; margin-bottom: 20px; }
        input { flex: 1; padding: 12px; border: 2px solid #cbd5e1; border-radius: 6px; font-size: 16px; outline: none; }
        input:focus { border-color: #2563eb; }
        button { background: #2563eb; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 500; }
        button:hover { background: #1d4ed8; }
        .result-card { background: #f1f5f9; padding: 16px; border-left: 4px solid #2563eb; border-radius: 4px; margin-bottom: 12px; font-size: 15px; line-height: 1.6; color: #334155; white-space: pre-wrap; }
        .no-results { color: #64748b; text-align: center; margin-top: 20px; font-style: italic; }
        .match-score { font-size: 11px; color: #94a3b8; font-weight: bold; text-transform: uppercase; margin-bottom: 4px; }
    </style>
</head>
<body>
    <div class="box">
        <h2>📖 Team Runbook Engine (FastAPI)</h2>
        <div class="search-container">
            <input type="text" id="query" placeholder="Type keywords (e.g., 504 gateway, database timeout, user login)..." onkeyup="if(event.key==='Enter') searchRunbook()">
            <button onclick="searchRunbook()">Search</button>
        </div>
        <div id="results"></div>
    </div>

    <script>
        async function searchRunbook() {
            const query = document.getElementById('query').value;
            const resultsDiv = document.getElementById('results');
            if(!query.trim()) return;
            
            resultsDiv.innerHTML = "<p style='color:#64748b;'>Searching runbook indexed chunks...</p>";
            
            try {
                const response = await fetch('./search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                if (!response.ok) {
                    throw new Error("Server returned an error response");
                }
                
                const data = await response.json();
                resultsDiv.innerHTML = "";
                
                if(!data.results || data.results.length === 0) {
                    resultsDiv.innerHTML = "<div class='no-results'>No matching resolution steps found for those keywords.</div>";
                    return;
                }
                
                data.results.forEach(item => {
                    const card = document.createElement('div');
                    card.className = 'result-card';
                    card.innerHTML = `<div class='match-score'>Match Relevance: ${(item.score * 100).toFixed(0)}%</div><div>${item.text}</div>`;
                    resultsDiv.appendChild(card);
                });
            } catch (err) {
                console.error(err);
                resultsDiv.innerHTML = "<div class='no-results' style='color:#ef4444;'>Error communicating with local search app. Check your console log.</div>";
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.post("/search")
async def search(payload: SearchQuery):
    user_query = payload.query.strip()
    if not user_query or tfidf_matrix is None:
        return {"results": []}
    
    query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    top_matches = []
    for idx, score in enumerate(similarity_scores):
        if score > 0.01:  # Slightly lowered match threshold for broader detection
            top_matches.append({"text": RUNBOOK_CHUNKS[idx], "score": float(score)})
            
    top_matches = sorted(top_matches, key=lambda x: x['score'], reverse=True)[:3]
    return {"results": top_matches}
