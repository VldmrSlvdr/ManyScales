"""Main translator class for LLM-based survey translation."""

import os
import logging
from typing import Dict, Optional, List
import requests

from .models import ModelSpec


logger = logging.getLogger(__name__)


class Translator:
    """Handles translation using various LLM providers."""
    
    def __init__(
        self,
        model: str,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        temperature: float = 0.0,
        effort: str = "medium"
    ):
        """
        Initialize translator.
        
        Args:
            model: Model name (will be normalized)
            openai_api_key: OpenAI API key (defaults to env var)
            gemini_api_key: Gemini API key (defaults to env var)
            temperature: Temperature for sampling (ignored for reasoning models)
            effort: Effort level for reasoning models ("low", "medium", "high")
        """
        self.model_spec = ModelSpec.get_model_spec(model)
        self.model = self.model_spec.name
        self.temperature = temperature if self.model_spec.supports_temp else None
        self.effort = effort
        
        # Get API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        # Validate API keys
        if self.model_spec.provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        if self.model_spec.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("Gemini API key not provided")
    
    def translate(self, text: str, target_lang: str, context: Optional[str] = None) -> str:
        """
        Translate text to target language.
        
        Args:
            text: Source text to translate
            target_lang: Target language code (e.g., "ja", "zh", "es")
            context: Optional context about the survey/scale
        
        Returns:
            Translated text
        """
        prompt = self._build_translation_prompt(text, target_lang, context)
        response = self._call_llm(prompt)
        return self._clean_response(response)
    
    def back_translate(self, text: str, source_lang: str) -> str:
        """
        Back-translate text to source language.
        
        Args:
            text: Translated text
            source_lang: Original source language code
        
        Returns:
            Back-translated text
        """
        prompt = self._build_back_translation_prompt(text, source_lang)
        response = self._call_llm(prompt)
        return self._clean_response(response)
    
    def reconcile(self, original: str, translation_a: str, translation_b: str, 
                  target_lang: str) -> Dict[str, str]:
        """
        Reconcile two translations and provide explanation.
        
        Args:
            original: Original source text
            translation_a: First translation
            translation_b: Second translation
            target_lang: Target language code
        
        Returns:
            Dictionary with 'revised' translation and 'explanation'
        """
        prompt = self._build_reconciliation_prompt(
            original, translation_a, translation_b, target_lang
        )
        response = self._call_llm(prompt)
        return self._parse_reconciliation(response)
    
    def batch_translate(self, texts: List[str], target_lang: str, 
                       context: Optional[str] = None) -> List[str]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of source texts
            target_lang: Target language code
            context: Optional context
        
        Returns:
            List of translated texts
        """
        return [self.translate(text, target_lang, context) for text in texts]
    
    def _build_translation_prompt(self, text: str, target_lang: str, 
                                  context: Optional[str]) -> str:
        """Build prompt for forward translation (TRAPD/ISPOR-aligned)."""
        from_lang = "English"
        lines = [
            "You are a professional survey translator working within TRAPD/ISPOR best practices.",
            f"Goal: Produce a conceptually equivalent {target_lang} version of the item below.",
            "Constraints:",
            "- Preserve meaning, intent, item polarity (negations), numbers, quantifiers, modality, and time references. Do not introduce changes if not needed. A translation as directly as possible is preferable as long as it is culturally meaningful.",
            "- Avoid culture-bound idioms/metaphors unless an equivalent exists.",
            "- Keep reading level and tone comparable to the source.",
            f"- Output ONLY the final {target_lang} item text (no quotes, no explanations).",
        ]
        if context:
            lines.append("")
            lines.append(f"Context: {context}")
        lines.extend([
            "",
            f"ITEM ({from_lang}): {text}",
        ])
        return "\n".join(lines)
    
    def _build_back_translation_prompt(self, text: str, source_lang: str) -> str:
        """Build prompt for back translation (TRAPD/ISPOR-aligned)."""
        to_lang = "target"
        lines = [
            "You are performing a blind back-translation as part of a TRAPD/ISPOR quality check.",
            f"Goal: Render the {to_lang} item back into {source_lang} as literally as possible while remaining grammatical.",
            "Rules:",
            f"- Do NOT try to improve wording or guess the original; reflect exactly what the {to_lang} item says.",
            "- Preserve polarity, quantifiers, modality, and tense.",
            "- No added comments, brackets, or explanations.",
            f"- Output ONLY the {source_lang} text (no quotes).",
            "",
            f"ITEM ({to_lang}): {text}",
        ]
        return "\n".join(lines)
    
    def _build_reconciliation_prompt(self, original: str, translation_a: str, 
                                    translation_b: str, target_lang: str) -> str:
        """Build prompt for reconciling translations (TRAPD/ISPOR-aligned)."""
        from_lang = "English"
        body = [
            "You are reconciling a survey translation (TRAPD/ISPOR step).",
            "Inputs:",
            f"1) ORIGINAL (in {from_lang})",
            f"2) FORWARD (in {target_lang})",
            f"3) BACK (in {from_lang})",
            "Tasks:",
            "- Compare ORIGINAL vs BACK to detect meaning shifts, omissions, or added nuances.",
            f"- Accept FORWARD if conceptually equivalent and natural in {target_lang}.",
            "- If revision is needed, adjust FORWARD to match ORIGINAL meaning precisely while keeping tone/register and length roughly similar.",
            "Return a JSON object with exactly these keys:",
            '{"revised": "<revised ' + target_lang + ' item only>", "explanation": "<one short sentence on any change>"}.',
            "No other text.",
            "",
            "---",
            "ORIGINAL:",
            original,
            "BACK:",
            translation_b,
            "FORWARD:",
            translation_a,
        ]
        return "\n".join(body)

    def call_prompt(self, prompt: str) -> str:
        """Call the underlying model with a custom prompt and clean the output."""
        response = self._call_llm(prompt)
        return self._clean_response(response)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM API."""
        logger.info(f"Calling {self.model} with prompt (first 100 chars): {prompt[:100]}")
        
        if self.model_spec.provider == "openai":
            return self._call_openai(prompt)
        else:
            return self._call_gemini(prompt)
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        if self.model_spec.type == "reasoning":
            try:
                return self._call_openai_reasoning(prompt)
            except Exception as e:
                logger.warning(f"Reasoning API failed: {e}. Falling back to chat.")
                return self._call_openai_chat(prompt, "gpt-4.1-mini")
        else:
            return self._call_openai_chat(prompt, self.model)
    
    def _call_openai_chat(self, prompt: str, model: str) -> str:
        """Call OpenAI chat completions API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.text
            except:
                pass
            raise RuntimeError(
                f"OpenAI API error (status {e.response.status_code}): {error_detail}\n"
                f"Please check your API key and quota."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error calling OpenAI API: {e}") from e
        
        data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"]["message"])
        
        return data["choices"][0]["message"]["content"]
    
    def _call_openai_reasoning(self, prompt: str) -> str:
        """Call OpenAI reasoning responses API."""
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "reasoning": {"effort": self.effort},
            "max_output_tokens": 2048
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.text
            except:
                pass
            raise RuntimeError(
                f"OpenAI API error (status {e.response.status_code}): {error_detail}\n"
                f"Please check your API key and quota."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error calling OpenAI API: {e}") from e
        
        data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"]["message"])
        
        return data.get("output_text", "")
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.gemini_api_key}"
        )
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {}
        }
        
        if self.temperature is not None:
            payload["generationConfig"]["temperature"] = self.temperature
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404 and self.model == "gemini-1.0-pro":
                logger.warning("gemini-1.0-pro not found, falling back to gemini-1.5-pro")
                self.model = "gemini-1.5-pro"
                return self._call_gemini(prompt)
            # Better error message
            error_detail = ""
            try:
                error_detail = e.response.text
            except:
                pass
            raise RuntimeError(
                f"Gemini API error (status {e.response.status_code}): {error_detail}\n"
                f"Please check your API key and quota."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error calling Gemini API: {e}") from e
        
        data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"]["message"])
        
        parts = data["candidates"][0]["content"]["parts"]
        return "\n".join(part["text"] for part in parts)
    
    def _clean_response(self, response: str) -> str:
        """Clean LLM response text."""
        # Remove common prefixes
        response = response.strip()
        
        # Remove A), B), etc. prefixes
        import re
        response = re.sub(r"^[A-Z]\)\s*", "", response)
        
        # Remove markdown code fences
        response = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", response)
        response = re.sub(r"```\s*$", "", response)
        
        return response.strip()
    
    def _parse_reconciliation(self, response: str) -> Dict[str, str]:
        """Parse reconciliation response."""
        import json
        
        cleaned = self._clean_response(response)
        
        # Try to parse as JSON
        try:
            result = json.loads(cleaned)
            if "revised" in result and "explanation" in result:
                return result
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse first line as revised, rest as explanation
        lines = cleaned.split("\n")
        if lines:
            return {
                "revised": lines[0],
                "explanation": "\n".join(lines[1:]) if len(lines) > 1 else ""
            }
        
        return {"revised": cleaned, "explanation": ""}

