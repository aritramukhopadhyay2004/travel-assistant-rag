import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Tourism Guide Assistant (RAG)",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode / Glassmorphism aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-green { background-color: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-blue { background-color: rgba(96, 165, 250, 0.15); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3); }
    .badge-yellow { background-color: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    
    .card-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .citation-tag {
        background: rgba(96, 165, 250, 0.2);
        color: #93c5fd;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Import RAG Pipeline
from src.rag_engine import TourismRAGPipeline
from src.evaluator import Evaluator, BENCHMARK_QUESTIONS

# Initialize session state pipeline
@st.cache_resource
def load_rag_pipeline():
    pipeline = TourismRAGPipeline()
    # Auto-ingest raw documents if available in data/raw_documents
    raw_dir = Path("data/raw_documents")
    if raw_dir.exists():
        for file_path in raw_dir.glob("*.*"):
            if file_path.suffix.lower() in [".pdf", ".txt", ".md"]:
                try:
                    pipeline.ingest_document(str(file_path))
                except Exception as e:
                    pass
    return pipeline

pipeline = load_rag_pipeline()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/illustrations/100/compass.png", width=70)
    st.markdown("### ✈️ Tourism Assistant")
    st.markdown("**Domain:** Singapore Travel Guide")
    st.markdown("---")
    
    st.markdown("#### ⚡ System Status")
    groq_key_set = bool(os.getenv("GROQ_API_KEY"))
    supa_set = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_KEY"))

    if groq_key_set:
        st.markdown('<span class="status-badge badge-green">Groq LLM: Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-yellow">Groq LLM: Extractive Fallback</span>', unsafe_allow_html=True)

    if supa_set:
        st.markdown('<span class="status-badge badge-green">Supabase DB: Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-blue">Vector Store: Local Mode</span>', unsafe_allow_html=True)

    st.markdown('<span class="status-badge badge-green">Guardrails: Active</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📜 Core Knowledge Docs")
    st.caption("• Official Visitor Guide (PDF)")
    st.caption("• Transit & Accessibility Guide")
    st.caption("• Official Tourism FAQ (TXT)")
    st.caption("• Cultural & Safety Guide")
    st.caption("• Ticketing & Timings Brochure")

    st.markdown("---")
    st.caption("Powered by Groq API + SentenceTransformers + Supabase pgvector")

# Main Header
st.markdown('<div class="main-title">Tourism Guide Assistant (RAG)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Consistent, Sourced, and Grounded Answers for Travelers & Tourists</div>', unsafe_allow_html=True)

# App Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Travel Assistant Chat",
    "📂 Knowledge Base Manager",
    "🧪 Benchmark Evaluator",
    "🛡️ Security & Governance"
])

# ==========================================
# TAB 1: TRAVEL ASSISTANT CHAT
# ==========================================
with tab1:
    st.markdown("### 💬 Ask a Travel Question")
    st.markdown("Choose one of the 5 expected benchmark questions or ask your custom query below:")

    # Quick prompt buttons for 5 benchmark questions
    col1, col2, col3, col4, col5 = st.columns(5)
    selected_prompt = None

    if col1.button("📌 1. Top Attractions", help="What are top attractions and time allocation?"):
        selected_prompt = "What are the top attractions in Singapore and how much time should I allocate for each?"
    if col2.button("⏰ 2. Opening Hours", help="Gardens by the Bay weekday/weekend hours"):
        selected_prompt = "What are the opening hours of Gardens by the Bay on weekdays and weekends?"
    if col3.button("🎟️ 3. Ticket Prices", help="Singapore Zoo ticket prices & discounts"):
        selected_prompt = "What is the ticket price / entry fee for Singapore Zoo, and are there any discounts?"
    if col4.button("🌤️ 4. Best Time to Visit", help="Best time to visit & expected weather"):
        selected_prompt = "When is the best time to visit Singapore, and what weather should I expect?"
    if col5.button("🚇 5. Airport Transport", help="Getting from Changi Airport to the city"):
        selected_prompt = "How do I get from Changi airport to the city, and what public transport options are available?"

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Welcome! I am your official Tourism Guide Assistant. Ask me about top attractions, opening hours, ticket prices, best travel times, transit options, or local visiting etiquette."
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your travel question here...")
    prompt_to_process = selected_prompt or user_input

    if prompt_to_process:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Process with RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching official travel documents..."):
                response = pipeline.query(prompt_to_process)
                answer_text = response["answer"]

                st.markdown(answer_text)

                # Render Citations & Source Chunks
                if response.get("citations"):
                    st.markdown("**Citations & Sources:**")
                    cite_html = "".join([f'<span class="citation-tag">{c}</span>' for c in response["citations"]])
                    st.markdown(cite_html, unsafe_allow_html=True)

                if response.get("retrieved_chunks"):
                    with st.expander("🔍 Inspect Retrieved Raw Document Context"):
                        for idx, chunk in enumerate(response["retrieved_chunks"], 1):
                            st.markdown(f"**Chunk {idx}** - `{chunk['document_name']}` (Page {chunk['page_number']}) | Similarity: `{round(chunk['similarity'], 3)}`")
                            st.info(chunk["content"])

        st.session_state.messages.append({"role": "assistant", "content": answer_text})

# ==========================================
# TAB 2: KNOWLEDGE BASE MANAGER
# ==========================================
with tab2:
    st.markdown("### 📂 Knowledge Base Document Manager")
    st.markdown("Upload travel documents (PDF, TXT, MD) to automatically chunk, embed, and index them into Supabase Vector Store.")

    uploaded_files = st.file_uploader(
        "Upload Official Tourism Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process & Ingest Files into Vector Database"):
            ingest_dir = Path("data/raw_documents")
            ingest_dir.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                save_path = ingest_dir / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                res = pipeline.ingest_document(str(save_path))
                if res["status"] in ["success", "partial_success"]:
                    st.success(f"✅ Ingested **{uploaded_file.name}**: {res['chunks_count']} chunks created and stored.")
                else:
                    st.error(f"❌ Failed to ingest {uploaded_file.name}: {res.get('message')}")

    st.markdown("---")
    st.markdown("#### 📊 Currently Ingested Knowledge Sources")
    raw_dir = Path("data/raw_documents")
    if raw_dir.exists():
        files = list(raw_dir.glob("*.*"))
        if files:
            for f in files:
                size_kb = round(f.stat().st_size / 1024, 1)
                st.markdown(f"📄 **{f.name}** (`{size_kb} KB`) - Category: `{pipeline.doc_processor._classify_doc_type(f.name)}`")
        else:
            st.info("No raw documents uploaded yet.")

# ==========================================
# TAB 3: BENCHMARK EVALUATOR
# ==========================================
with tab3:
    st.markdown("### 🧪 RAG Evaluation & Quality Verification")
    st.markdown("Runs automated test suite for the **5 expected benchmark questions** and **3 negative out-of-scope test cases**.")

    if st.button("▶️ Run Automated Benchmark Test Suite"):
        evaluator = Evaluator(pipeline)
        with st.spinner("Executing benchmark queries..."):
            summary = evaluator.run_benchmark()

        c1, c2, c3 = st.columns(3)
        c1.metric("Positive 5 Questions Met", summary["positive_benchmark_passed"])
        c2.metric("Out-of-Scope Refusals Met", summary["negative_benchmark_passed"])
        c3.metric("Overall RAG Accuracy", f"{summary['overall_accuracy_percent']}%")

        if summary["success_criteria_met"]:
            st.success("🎉 **Success Criteria Passed**: Chatbot accurately answered expected questions and refused out-of-scope requests!")
        else:
            st.warning("⚠️ Success Criteria partially met.")

        st.markdown("#### 📋 Detailed Test Results")
        for detail in summary["test_details"]:
            status_icon = "✅ PASS" if detail["passed"] else "❌ FAIL"
            with st.expander(f"{status_icon} | [{detail['id']}] {detail['question']}"):
                st.write(f"**Expected Type**: `{detail['expected_type']}`")
                st.write(f"**Evaluation Notes**: {detail['notes']}")
                st.write(f"**Answer Snippet**: {detail['answer_snippet']}")

# ==========================================
# TAB 4: SECURITY & GOVERNANCE
# ==========================================
with tab4:
    st.markdown("### 🛡️ Security, Privacy & Travel Terms Governance")
    
    st.markdown("""
    <div class="card-box">
        <h4>🔒 Security & Privacy Directives</h4>
        <ul>
            <li><b>Zero Hallucination Enforcement:</b> Factual travel information (opening hours, entry fees, transport routes) must be directly sourced from official documents.</li>
            <li><b>PII & Sensitive Data Protection:</b> User queries are checked against strict privacy filters to prevent processing personal data or confidential records.</li>
            <li><b>Out-of-Scope Guardrails:</b> Requests involving non-destination cities, personal financial/medical advice, or real-time news are declined automatically.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📜 Active Domain Configuration (`domain_config.yaml`)")
    st.json(pipeline.domain_config)
