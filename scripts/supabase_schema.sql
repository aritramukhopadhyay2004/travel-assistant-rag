-- =========================================================
-- Supabase pgvector Schema for Tourism Guide Assistant RAG
-- =========================================================

-- 1. Enable the vector extension to work with embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the document_chunks table
CREATE TABLE IF NOT EXISTS public.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_name TEXT NOT NULL,
    document_type TEXT NOT NULL,
    page_number INT,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(384), -- 384 dimensions for all-MiniLM-L6-v2
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Create vector similarity search index (IVFFlat or HNSW)
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx 
ON public.document_chunks 
USING hnsw (embedding vector_cosine_ops);

-- 4. Create RPC function for Cosine Similarity Search
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding VECTOR(384),
  match_threshold FLOAT DEFAULT 0.35,
  match_count INT DEFAULT 5,
  filter_doc_name TEXT DEFAULT NULL
)
RETURNS TABLE (
  id UUID,
  document_name TEXT,
  document_type TEXT,
  page_number INT,
  chunk_index INT,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    dc.id,
    dc.document_name,
    dc.document_type,
    dc.page_number,
    dc.chunk_index,
    dc.content,
    dc.metadata,
    1 - (dc.embedding <=> query_embedding) AS similarity
  FROM public.document_chunks dc
  WHERE 1 - (dc.embedding <=> query_embedding) >= match_threshold
    AND (filter_doc_name IS NULL OR dc.document_name = filter_doc_name)
  ORDER BY dc.embedding <=> query_embedding ASC
  LIMIT match_count;
END;
$$;
