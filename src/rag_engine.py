import os
import logging
from typing import Dict, Any, List

from .config_loader import ConfigLoader
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingEngine
from .vector_store import VectorStore
from .guardrails import SecurityGuardrail
from .llm import GroqLLM

logger = logging.getLogger(__name__)

class TourismRAGPipeline:
    """
    End-to-End Tourism Assistant RAG Pipeline.
    """
    def __init__(self, config_dir: str = None):
        self.config_loader = ConfigLoader(config_dir)
        self.domain_config = self.config_loader.get_domain_config()
        self.retrieval_config = self.config_loader.get_retrieval_config()
        self.security_config = self.config_loader.get_security_config()

        # Components
        self.doc_processor = DocumentProcessor(
            chunk_size=self.retrieval_config.get("chunking", {}).get("chunk_size", 500),
            chunk_overlap=self.retrieval_config.get("chunking", {}).get("chunk_overlap", 50)
        )
        model_name = self.retrieval_config.get("embedding", {}).get("model_name", "all-MiniLM-L6-v2")
        self.embeddings_engine = EmbeddingEngine(model_name=model_name)
        self.vector_store = VectorStore()
        self.guardrail = SecurityGuardrail(self.domain_config, self.security_config)
        
        llm_model = self.retrieval_config.get("llm", {}).get("default_model", "llama-3.3-70b-versatile")
        self.llm = GroqLLM(model_name=llm_model)

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingests a PDF/TXT/MD document, chunks it, generates embeddings, and uploads to vector store.
        """
        chunks = self.doc_processor.process_file(file_path)
        if not chunks:
            return {"status": "error", "message": "No valid text content extracted.", "chunks_count": 0}

        texts = [c["content"] for c in chunks]
        embeddings = self.embeddings_engine.embed_documents(texts)
        success = self.vector_store.add_chunks(chunks, embeddings)

        return {
            "status": "success" if success else "partial_success",
            "file_name": os.path.basename(file_path),
            "chunks_count": len(chunks),
            "embeddings_generated": len(embeddings)
        }

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Executes user query through Guardrail -> Retrieval -> Groq LLM pipeline.
        """
        # Step 1: Security & Scope Guardrail Validation
        is_allowed, refusal_reason, category = self.guardrail.validate_query(user_query)
        if not is_allowed:
            refusal_msg = self.domain_config.get(
                "refusal_response",
                "I cannot find this information in the official travel knowledge sources or the request is outside the scope of the destination tourism guide."
            )
            return {
                "allowed": False,
                "refusal_reason": refusal_reason,
                "category": category,
                "answer": f"⚠️ **Guardrail Notice**: {refusal_reason}\n\n{refusal_msg}",
                "retrieved_chunks": [],
                "citations": []
            }

        # Step 2: Generate Query Embedding
        query_vec = self.embeddings_engine.embed_query(user_query)

        # Step 3: Similarity Search in Vector Store (Supabase / Local)
        top_k = self.retrieval_config.get("retrieval", {}).get("top_k", 4)
        threshold = self.retrieval_config.get("retrieval", {}).get("similarity_threshold", 0.35)
        retrieved_chunks = self.vector_store.search_similarity(query_vec, top_k=top_k, threshold=threshold)

        # Step 4: Synthesize Response with Groq LLM
        llm_response = self.llm.generate_response(user_query, retrieved_chunks)

        return {
            "allowed": True,
            "category": "Allowed Tourism Query",
            "answer": llm_response["answer"],
            "citations": llm_response["citations"],
            "grounded": llm_response.get("grounded", True),
            "model_used": llm_response.get("model", "groq"),
            "retrieved_chunks": retrieved_chunks
        }
