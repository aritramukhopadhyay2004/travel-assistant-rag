# Tourism Guide Assistant (RAG) ✈️

A production-grade, enterprise-compatible Retrieval-Augmented Generation (RAG) system designed to deliver consistent, sourced, and accurate travel answers for travelers, tourists, and visitors. Built with **Streamlit**, **Groq API** (`llama-3.3-70b-versatile`), **SentenceTransformers** (`all-MiniLM-L6-v2`), and **Supabase** (`pgvector`).

---

## 🌟 Key Features

1. **Grounded Travel Question Answering**:
   - Accurately answers tourist questions regarding **top attractions**, **opening hours**, **ticket prices/fees**, **best time to visit & weather**, and **airport-to-city transport**.
   - Includes precise inline citations (`[Document Name - Page X]`) for every response.

2. **Strict Security & Privacy Guardrails**:
   - Rejects out-of-scope requests (e.g. unrelated destinations/visas, personal financial/medical advice, real-time news updates, private/sensitive data).
   - Enforces a **Zero-Hallucination** clause: if information is missing from documents, the chatbot clearly states: *"I cannot find this information in the official travel knowledge sources."*

3. **Supabase Vector Database + Local Fallback**:
   - Indexes PDF, TXT, and Markdown document chunks into Supabase `document_chunks` table using 384-dimensional embeddings and cosine similarity search (`match_documents` RPC).
   - Features a high-performance local vector store fallback for offline testing or development without database credentials.

4. **Automated Benchmark Evaluator**:
   - Built-in test suite evaluating performance across the 5 core benchmark questions and out-of-scope refusal scenarios.

5. **Interactive Streamlit Web UI**:
   - Modern dark-mode interface with chat history, raw document context inspector, file manager/uploader, and privacy governance viewer.

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌──────────────────────────────────────┐
│  Security & Out-of-Scope Guardrails  │ ──► Refusal Response (if out-of-scope)
└──────────────────────────────────────┘
    │ (Allowed)
    ▼
┌──────────────────────────────────────┐
│ Embedding Engine (SentenceTransform)│ ──► 384-dim Dense Vector
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Vector Store (Supabase pgvector)    │ ──► Cosine Similarity Match (Top-K Chunks)
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│    Groq LLM Engine (Llama-3.3-70b)   │ ──► Grounded Answer + Source Citations
└──────────────────────────────────────┘
```

---

## 🛠️ Project Structure

```
rag-internship/
├── app.py                      # Streamlit Web Application
├── requirements.txt            # Python Dependencies
├── .env.example                # Environment Variable Template
├── .gitignore                  # Git Ignore Rules
├── README.md                   # Project Documentation
├── config/
│   ├── domain_config.yaml      # Tourism Domain Rules & Refusal Messages
│   ├── retrieval_config.yaml   # Chunking, Vector Top-K, and LLM Settings
│   └── security_policy.yaml    # Security, Privacy & Grounding Directives
├── data/
│   ├── raw_documents/          # Tourism PDFs, TXT, MD Documents
│   └── notes/                  # Summaries and domain notes
├── scripts/
│   └── supabase_schema.sql     # Supabase pgvector table & search function SQL
└── src/
    ├── __init__.py
    ├── config_loader.py        # YAML Configuration Loader
    ├── document_processor.py   # PDF, Text, MD Extractor & Chunker
    ├── embeddings.py           # SentenceTransformers Embedding Engine
    ├── vector_store.py         # Supabase pgvector & Local Fallback Vector Store
    ├── guardrails.py           # Security & Out-of-Scope Query Validator
    ├── llm.py                  # Groq API Client & Prompt Builder
    ├── rag_engine.py           # End-to-End RAG Orchestration Pipeline
    └── evaluator.py            # Automated Benchmark Test Suite
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/your-username/tourism-rag-assistant.git
cd tourism-rag-assistant
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your API keys:
- `GROQ_API_KEY`: Get a free key from [Groq Console](https://console.groq.com).
- `SUPABASE_URL`: Your Supabase Project URL (`https://xyz.supabase.co`).
- `SUPABASE_KEY`: Your Supabase Anon or Service Role key.

*(Note: If Supabase credentials are not supplied, the system automatically runs in Local Fallback Vector Mode!)*

### 3. Setup Supabase Vector Database (Optional for Cloud Mode)
1. Open your Supabase SQL Editor.
2. Run the SQL script located in `scripts/supabase_schema.sql`.
3. This creates the `document_chunks` table and `match_documents` vector search function.

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Benchmark & Quality Verification

Run the automated evaluation suite from the terminal:
```bash
python -m src.evaluator
```

This will test all **5 expected questions**:
1. *Top attractions and recommended duration.*
2. *Opening hours of specific attractions on weekdays/weekends.*
3. *Ticket prices / entry fees and available discounts.*
4. *Best time to visit and expected weather.*
5. *Airport to city transport options.*

...along with negative out-of-scope queries (unrelated destinations, personal advice, real-time news) to verify 100% refusal accuracy.

---

## 📄 License & Attribution

Developed for the RAG Internship Capstone Project under the DomainLens Enterprise RAG Framework.
