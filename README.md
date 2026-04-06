# 🏥 ClaimIQ — Automated Insurance Claims Adjudicator

An AI-powered system that reads your insurance policy PDF and hospital bill, then automatically adjudicates each line item — telling you exactly what's covered, what you owe, and which clause in the policy backs up the decision.

---

## Architecture

```
Policy PDF  →  Azure OCR (chunked)  →  TextNodes
                                           ↓
                               TF-IDF filter (remove noise)
                                           ↓
                          Gemini Embeddings + BM25 → Hybrid Index
                                                           ↓
Hospital Bill  →  Azure OCR (invoice)  →  Line items
                                                    ↓
                                    QueryFusion Retriever (vector + BM25)
                                                    ↓
                                        Gemini LLM Adjudicator
                                                    ↓
                              Results table | PDF highlight | CSV report
```

---

## Features

- **Full policy ingestion** — splits PDF into 2-page chunks to work within Azure free tier limits, processes all pages in parallel
- **Smart filtering** — TF-IDF cosine similarity removes definitions, glossary, headers, footers, and company boilerplate before embedding — reduces Gemini API calls by ~60%
- **Hybrid retrieval** — combines vector similarity (Gemini embeddings) + BM25 keyword search via QueryFusionRetriever for best clause recall
- **Structured verdicts** — each bill item gets: `Approved` / `Partially Approved` / `Rejected`, with exact math breakdown and policy citation
- **PDF clause highlighting** — highlights the exact polygon region on the policy page using PyMuPDF
- **CSV export** — download full adjudication report

---

## Project Structure

```
insurance-agent/
├── app.py                  # Streamlit UI
├── ocr.py                  # Azure OCR + PyMuPDF bill parser
├── core/
│   ├── retriver.py         # Hybrid index + retrieval
│   └── adjuvicator.py      # Gemini LLM adjudication
├── utils.py                # PDF highlighting (PyMuPDF)
├── data/                   # ChromaDB / temp storage
├── .env                    # API keys (see below)
└── requirements.txt
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo>
cd insurance-agent
python -m venv ai
ai\Scripts\activate        # Windows
# source ai/bin/activate   # Mac/Linux
```

### 2. Install dependencies

```bash
pip install streamlit pandas python-dotenv pypdf pymupdf scikit-learn
pip install llama-index llama-index-embeddings-google llama-index-llms-google-genai
pip install llama-index-retrievers-bm25 llama-index-vector-stores-chroma
pip install azure-ai-documentintelligence azure-core
pip install chromadb
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```dotenv
AZURE_ENDPOINT=your_azure_endpoint_here
AZURE_KEY="your_azure_key_here"
GOOGLE_API_KEY=your_google_api_key_here
```

> **Get your Google API key** at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — free tier includes 1000 embedding requests/day and LLM calls.

> **Azure note** — the endpoint above is for Document Intelligence layout analysis (`prebuilt-layout`). Bill parsing uses PyMuPDF + Gemini locally and does not require Azure.

### 4. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Step 1 — Upload your insurance policy PDF
- Supports any standard health insurance policy document (up to 50 pages recommended)
- Click **Index Policy** — this will:
  - Extract all text via Azure OCR (parallel chunking, ~30s for 25 pages)
  - Filter out noise nodes using TF-IDF semantic similarity
  - Build a hybrid vector + BM25 index using Gemini embeddings (~1–2 min)

### Step 2 — Upload your hospital bill PDF
- Click **Adjudicate Bill** — for each line item the system will:
  - Retrieve the top 3 most relevant policy clauses
  - Run the Gemini LLM adjudicator to determine coverage
  - Display verdict, math breakdown, and policy citation

### Step 3 — Review results
- Expand any item to see full reasoning and the exact policy quote
- Click **Show Clause in PDF** to see the highlighted clause in the policy document
- Download the full report as CSV

---

## API Rate Limits (Free Tier)

| Service | Limit | Notes |
|---|---|---|
| Gemini embeddings | 1000 req/day, 100 req/min | Use `batch_size=20` to stay under per-min limit |
| Gemini LLM | 1500 req/day | ~5–10 calls per bill item adjudication |
| Azure OCR | 500 pages/month (F0) | 2-page chunks, parallel processing |

For a 25-page policy + 10-item bill: ~30 embedding calls + ~10 LLM calls — well within daily limits.

---

## Known Issues

- `resource module not available on Windows` — harmless warning from Streamlit on Windows, does not affect functionality
- Azure free tier (F0) processes max 2 pages per call — the chunked parallel pipeline handles this automatically
- If Gemini LLM times out (`ConnectTimeout`), disable VPN if active and retry

---

## Models Used

| Component | Model |
|---|---|
| Policy OCR | Azure `prebuilt-layout` |
| Bill parsing | Azure `prebuilt-invoice` |
| Embeddings | `models/gemini-embedding-001` |
| Adjudication LLM | `gemini-3.1-flash-lite-preview` |

---

## Built With

- [Streamlit](https://streamlit.io) — UI
- [LlamaIndex](https://www.llamaindex.ai) — RAG framework
- [Azure Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence) — OCR
- [Google Gemini](https://ai.google.dev) — embeddings + LLM
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF manipulation
- [scikit-learn](https://scikit-learn.org) — TF-IDF filtering
