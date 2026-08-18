import os
import re
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class GroqLLM:
    """
    Groq API client for fast grounded LLM generation without any reasoning output.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "openai/gpt-oss-20b"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Initialized Groq LLM client with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")
                self.client = None
        else:
            logger.info("Groq API key not set in environment variables.")

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a direct, clean, to-the-point response using provided context chunks.
        """
        if not context_chunks:
            return {
                "answer": "I cannot find this information in the official travel knowledge sources.",
                "citations": [],
                "grounded": False
            }

        # Build formatted context block with citations
        context_str = ""
        citations = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("document_name", "Official Document")
            doc_type = chunk.get("document_type", "Travel Guide")
            page = chunk.get("page_number", 1)
            content = chunk.get("content", "")
            
            cite_tag = f"[{source} - Page {page}]"
            citations.append(cite_tag)
            context_str += f"--- Source {i}: {cite_tag} ({doc_type}) ---\n{content}\n\n"

        system_prompt = (
            "You are the Official Tourism Guide Assistant. Provide a direct, beautifully structured, "
            "and to-the-point answer for travelers and tourists.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the final answer directly. DO NOT include any thinking process, reasoning steps, or internal meta-commentary.\n"
            "2. Format your response clearly with bold headers, bullet points, and clean paragraphs.\n"
            "3. Answer using ONLY the provided official document context below. Never invent prices, opening hours, transport details, or rules.\n"
            "4. If a specific detail requested (e.g., recommended time allocation for each attraction) is missing from the context, include a clear 1-line note:\n"
            "   'ℹ️ *Note: Specific visit duration/time allocations are not stated in the official travel documents.*'\n"
            "5. Include precise inline source references (e.g. [Document Name - Page X]) for every factual detail."
        )

        user_message = f"DOCUMENT CONTEXT:\n{context_str}\nUSER QUESTION:\n{query}"

        if self.client:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model_name,
                    temperature=0.1,
                    max_tokens=1200
                )
                raw_answer = chat_completion.choices[0].message.content or ""
                clean_answer = self._clean_llm_output(raw_answer)

                return {
                    "answer": clean_answer,
                    "citations": list(set(citations)),
                    "grounded": True,
                    "model": self.model_name
                }
            except Exception as e:
                logger.error(f"Groq API call error: {e}")
                fallback_answer = self._format_local_extractive_answer(query, context_chunks)
                return {
                    "answer": fallback_answer,
                    "citations": list(set(citations)),
                    "grounded": True,
                    "model": "local_extractive_fallback"
                }

        fallback_answer = self._format_local_extractive_answer(query, context_chunks)
        return {
            "answer": fallback_answer,
            "citations": list(set(citations)),
            "grounded": True,
            "model": "local_extractive_fallback"
        }

    def _clean_llm_output(self, text: str) -> str:
        if not text:
            return ""
        # Handle <think> tags if any appear
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        elif "<think>" in text:
            # If think tag exists without closing, extract text after <think> if any non-think text exists
            text = re.sub(r"(?i)^<think>.*?</think>", "", text, flags=re.DOTALL)
            text = re.sub(r"(?i)^<think>.*?\n\n", "", text, flags=re.DOTALL)
            text = re.sub(r"(?i)<think>", "", text)
        
        text = re.sub(r"(?i)<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
        return text.strip()

    def _format_local_extractive_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        res = "Based on official travel documents:\n\n"
        for i, chunk in enumerate(chunks[:3], 1):
            src = chunk.get("document_name")
            pg = chunk.get("page_number")
            text = chunk.get("content", "").strip()
            res += f"📌 **[{src} - Page {pg}]**:\n{text[:400]}...\n\n"
        return res