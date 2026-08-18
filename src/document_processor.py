import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            pages = self._extract_pdf(path)
        elif ext in [".txt", ".md"]:
            pages = self._extract_text(path)
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return []

        chunks = []
        chunk_counter = 0
        for page in pages:
            text = page["content"]
            page_num = page["page_number"]
            doc_name = path.name
            doc_type = self._classify_doc_type(doc_name)

            page_chunks = self._chunk_text(text)
            for ch in page_chunks:
                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"{path.stem}_chunk_{chunk_counter}",
                    "document_name": doc_name,
                    "document_type": doc_type,
                    "page_number": page_num,
                    "chunk_index": chunk_counter,
                    "content": ch,
                    "metadata": {
                        "source": doc_name,
                        "doc_type": doc_type,
                        "page": page_num,
                        "char_count": len(ch)
                    }
                })

        logger.info(f"Processed {file_path}: created {len(chunks)} chunks across {len(pages)} pages.")
        return chunks

    def _extract_pdf(self, path: Path) -> List[Dict[str, Any]]:
        pages = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                if text.strip():
                    pages.append({"page_number": i + 1, "content": text})
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {e}")
        return pages

    def _extract_text(self, path: Path) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            cleaned = self._clean_text(text)
            return [{"page_number": 1, "content": cleaned}]
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        # Normalize whitespace while preserving paragraphs
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split(" ")
        if len(words) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_str = " ".join(chunk_words).strip()
            if chunk_str:
                chunks.append(chunk_str)
        return chunks

    def _classify_doc_type(self, doc_name: str) -> str:
        name = doc_name.lower()
        if "faq" in name:
            return "Official Tourism FAQ"
        elif "transit" in name or "accessibility" in name or "transport" in name:
            return "Transit & Accessibility Guide"
        elif "visitor" in name or "guide" in name:
            return "Official Visitor Guide"
        elif "cultural" in name or "safety" in name or "laws" in name:
            return "Cultural & Safety Guide"
        elif "ticket" in name or "timing" in name or "fee" in name:
            return "Ticketing & Timings Brochure"
        return "Travel Document"
