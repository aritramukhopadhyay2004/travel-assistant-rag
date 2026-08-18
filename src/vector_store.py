import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Supabase pgvector interface with local in-memory/NumPy fallback.
    """
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.client = None
        self.local_chunks: List[Dict[str, Any]] = []
        self.local_embeddings: Optional[np.ndarray] = None
        
        self._init_supabase()

    def _init_supabase(self):
        if self.supabase_url and self.supabase_key:
            try:
                from supabase import create_client
                self.client = create_client(self.supabase_url, self.supabase_key)
                logger.info("Successfully connected to Supabase Vector Store.")
            except Exception as e:
                logger.warning(f"Could not connect to Supabase: {e}. Operating in Local Fallback Mode.")
                self.client = None
        else:
            logger.info("Supabase credentials not set. Operating in Local Fallback Vector Mode.")

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch.")

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        # Store in local fallback array always for hybrid fast search
        self._add_to_local(chunks, embeddings)

        # Upload to Supabase if client is active
        if self.client:
            try:
                records = []
                for chunk, emb in zip(chunks, embeddings):
                    records.append({
                        "document_name": chunk["document_name"],
                        "document_type": chunk["document_type"],
                        "page_number": chunk.get("page_number", 1),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "content": chunk["content"],
                        "metadata": chunk.get("metadata", {}),
                        "embedding": emb
                    })
                # Batch upsert into document_chunks table
                response = self.client.table("document_chunks").insert(records).execute()
                logger.info(f"Uploaded {len(records)} chunks to Supabase document_chunks table.")
                return True
            except Exception as e:
                logger.error(f"Error inserting chunks to Supabase: {e}. Stored in local fallback vector store.")
                return False
        return True

    def search_similarity(self, query_embedding: List[float], top_k: int = 4, threshold: float = 0.35) -> List[Dict[str, Any]]:
        # 1. Try Supabase pgvector search first if available
        if self.client:
            try:
                rpc_response = self.client.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": threshold,
                        "match_count": top_k
                    }
                ).execute()

                if rpc_response.data:
                    results = []
                    for row in rpc_response.data:
                        results.append({
                            "document_name": row.get("document_name"),
                            "document_type": row.get("document_type"),
                            "page_number": row.get("page_number", 1),
                            "content": row.get("content"),
                            "metadata": row.get("metadata", {}),
                            "similarity": row.get("similarity", 0.0)
                        })
                    return results
            except Exception as e:
                logger.warning(f"Supabase RPC search failed: {e}. Falling back to local vector search.")

        # 2. Local Fallback Search using Cosine Similarity
        return self._search_local(query_embedding, top_k, threshold)

    def _add_to_local(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.local_chunks.extend(chunks)
        emb_np = np.array(embeddings, dtype=np.float32)
        if self.local_embeddings is None:
            self.local_embeddings = emb_np
        else:
            self.local_embeddings = np.vstack([self.local_embeddings, emb_np])

    def _search_local(self, query_embedding: List[float], top_k: int = 4, threshold: float = 0.35) -> List[Dict[str, Any]]:
        if self.local_embeddings is None or len(self.local_chunks) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-10
        q_vec_norm = q_vec / q_norm

        doc_norms = np.linalg.norm(self.local_embeddings, axis=1, keepdims=True) + 1e-10
        doc_vecs_norm = self.local_embeddings / doc_norms

        similarities = np.dot(doc_vecs_norm, q_vec_norm)

        # Get top indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                chunk = self.local_chunks[idx]
                results.append({
                    "document_name": chunk.get("document_name"),
                    "document_type": chunk.get("document_type"),
                    "page_number": chunk.get("page_number", 1),
                    "content": chunk.get("content"),
                    "metadata": chunk.get("metadata", {}),
                    "similarity": sim
                })
        return results

    def clear(self):
        self.local_chunks = []
        self.local_embeddings = None
