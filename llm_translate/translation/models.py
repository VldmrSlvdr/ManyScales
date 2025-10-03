"""Model specifications and configuration for LLM providers."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelSpec:
    """Specification for an LLM model."""
    
    name: str
    provider: str  # "openai" or "gemini"
    type: str  # "chat" or "reasoning"
    supports_temp: bool
    
    @classmethod
    def get_supported_models(cls) -> List["ModelSpec"]:
        """Get list of all supported models."""
        return [
            # OpenAI chat models
            cls("gpt-4o-mini", "openai", "chat", True),
            cls("gpt-4o", "openai", "chat", True),
            cls("gpt-4.1-mini", "openai", "chat", True),
            cls("gpt-4.1", "openai", "chat", True),
            
            # OpenAI reasoning models (o-series)
            cls("o3-mini", "openai", "reasoning", False),
            cls("o1-mini", "openai", "reasoning", False),
            cls("o1", "openai", "reasoning", False),
            
            # OpenAI reasoning models (GPT-5 family)
            cls("gpt-5", "openai", "reasoning", False),
            cls("gpt-5-mini", "openai", "reasoning", False),
            cls("gpt-5-nano", "openai", "reasoning", False),
            
            # Gemini 2.5 models (stable)
            cls("gemini-2.5-flash", "gemini", "chat", True),
            cls("gemini-2.5-pro", "gemini", "chat", True),
            cls("gemini-2.5-flash-lite", "gemini", "chat", True),
            
            # Gemini 2.0 models (stable)
            cls("gemini-2.0-flash", "gemini", "chat", True),
            cls("gemini-2.0-flash-001", "gemini", "chat", True),
            cls("gemini-2.0-flash-lite", "gemini", "chat", True),
            cls("gemini-2.0-flash-lite-001", "gemini", "chat", True),
            
            # Gemini latest aliases
            cls("gemini-flash-latest", "gemini", "chat", True),
            cls("gemini-pro-latest", "gemini", "chat", True),
            cls("gemini-flash-lite-latest", "gemini", "chat", True),
            
            # Legacy models (kept for backward compatibility)
            cls("gemini-1.5-flash", "gemini", "chat", True),
            cls("gemini-1.5-pro", "gemini", "chat", True),
            cls("gemini-1.0-pro", "gemini", "chat", True),
        ]
    
    @classmethod
    def get_model_map(cls) -> Dict[str, "ModelSpec"]:
        """Get dictionary mapping model names to specs."""
        return {model.name: model for model in cls.get_supported_models()}
    
    @classmethod
    def normalize_model_name(cls, name: str) -> str:
        """Normalize model name aliases to canonical names."""
        normalize_map = {
            # o-series
            "o1": "o1",
            "o1-mini": "o1-mini",
            "o3-mini": "o3-mini",
            
            # GPT-4 family
            "4o-mini": "gpt-4o-mini",
            "4o": "gpt-4o",
            "4.1": "gpt-4.1",
            "4.1-mini": "gpt-4.1-mini",
            
            # GPT-5 family
            "5": "gpt-5",
            "5-mini": "gpt-5-mini",
            "5mini": "gpt-5-mini",
            "5-nano": "gpt-5-nano",
            "5nano": "gpt-5-nano",
            "gpt5": "gpt-5",
            "gpt5-mini": "gpt-5-mini",
            "gpt5mini": "gpt-5-mini",
            "gpt5-nano": "gpt-5-nano",
            "gpt5nano": "gpt-5-nano",
            
            # Gemini aliases
            "gemini-flash": "gemini-flash-latest",
            "gemini-pro": "gemini-pro-latest",
        }
        
        normalized = name.strip().lower().replace(" ", "")
        return normalize_map.get(normalized, name)
    
    @classmethod
    def get_model_spec(cls, model_name: str) -> "ModelSpec":
        """Get model specification by name."""
        canonical_name = cls.normalize_model_name(model_name)
        model_map = cls.get_model_map()
        
        if canonical_name not in model_map:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model_map[canonical_name]

