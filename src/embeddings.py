import logging
from typing import List, Union

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Generates text embeddings using SentenceTransformers (default: all-MiniLM-L6-v2, 384-dim).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer '{self.model_name}': {e}. Fallback to dummy/mock embedding for offline mode.")
            self.model = None

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 384
        if self.model:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            # Deterministic hash-based fallback embedding if offline
            import hashlib
            import numpy as np
            h = hashlib.md5(text.encode("utf-8")).digest()
            np.random.seed(int.from_bytes(h[:4], "big"))
            vec = np.random.randn(384).astype(np.float32)
            vec /= np.linalg.norm(vec)
            return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
            return embeddings.tolist()
        else:
            return [self.embed_query(t) for t in texts]
