import re
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class SecurityGuardrail:
    """
    Enforces security, privacy, and out-of-scope domain boundaries.
    """
    def __init__(self, domain_config: Dict[str, Any] = None, security_config: Dict[str, Any] = None):
        self.domain_config = domain_config or {}
        self.security_config = security_config or {}
        
        # Out-of-scope destination keywords
        self.unrelated_destinations = [
            "tokyo", "japan", "paris", "france", "london", "uk", "new york", "usa",
            "dubai", "uae", "rome", "italy", "sydney", "australia", "canada", "visa for",
            "passport renewal", "embassy application", "schengen visa"
        ]

        # Financial / Medical personal advice keywords
        self.personal_advice_keywords = [
            "should i invest", "buy stocks", "medical treatment", "prescribe", "diagnosis",
            "legal suit", "lawsuit", "my private password", "personal tax"
        ]

        # Real-time change questions keywords
        self.realtime_keywords = [
            "did opening hours change yesterday", "live traffic status right now",
            "is it raining right now at this minute", "real time stock price",
            "current flight status live"
        ]

        # Sensitive & PII keywords
        self.sensitive_keywords = [
            "social security number", "credit card number", "bank account",
            "employee salary", "internal password", "confidential document"
        ]

    def validate_query(self, query: str) -> Tuple[bool, str, str]:
        """
        Validates user query against security and domain guardrails.
        Returns: (is_allowed: bool, refusal_reason: str, category: str)
        """
        q_lower = query.lower().strip()

        # 1. Check Unrelated Destinations & Visas
        for dest in self.unrelated_destinations:
            if dest in q_lower and "singapore" not in q_lower:
                return False, f"Request pertains to unrelated destination or international visa processing ({dest}).", "Unrelated Destination"

        # 2. Check Personal Advice (Financial/Medical/Legal)
        for kw in self.personal_advice_keywords:
            if kw in q_lower:
                return False, "Request asks for personal financial, medical, or legal advice unsupported by travel documents.", "Personal Advice"

        # 3. Check Real-Time Updates
        for kw in self.realtime_keywords:
            if kw in q_lower:
                return False, "Request requires live/real-time updates not present in static travel documents.", "Real-Time Query"

        # 4. Check Private & Sensitive Info
        for kw in self.sensitive_keywords:
            if kw in q_lower:
                return False, "Request pertains to private/sensitive data or confidential records.", "Privacy & Sensitive Info"

        return True, "", "Allowed"
