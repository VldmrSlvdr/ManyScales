"""Configuration validation."""

import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validate configuration dictionaries."""
    
    @staticmethod
    def validate_translation_config(config: Dict[str, Any]) -> None:
        """
        Validate translation configuration.
        
        Args:
            config: Configuration dictionary
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ["model", "source_lang", "target_lang"]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in translation config: {key}")
        
        # Validate model
        if not isinstance(config["model"], str) or not config["model"].strip():
            raise ValueError("Model must be a non-empty string")
        
        # Validate languages
        for lang_key in ["source_lang", "target_lang"]:
            if not isinstance(config[lang_key], str) or not config[lang_key].strip():
                raise ValueError(f"{lang_key} must be a non-empty string")
        
        # Validate optional fields
        if "temperature" in config:
            temp = config["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise ValueError("Temperature must be a number between 0 and 2")
        
        if "effort" in config:
            effort = config["effort"]
            if effort not in ["low", "medium", "high"]:
                raise ValueError("Effort must be one of: low, medium, high")
        
        # Validate workflow flags
        if "do_back" in config and not isinstance(config["do_back"], bool):
            raise ValueError("do_back must be a boolean")
        if "do_recon" in config and not isinstance(config["do_recon"], bool):
            raise ValueError("do_recon must be a boolean")
        if "batch_mode" in config and not isinstance(config["batch_mode"], bool):
            raise ValueError("batch_mode must be a boolean")

        if "batch_size" in config:
            batch_size = config["batch_size"]
            if not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError("batch_size must be a positive integer")

        # Optional custom prompts
        for key in [
            "forward_prompt",
            "back_prompt",
            "recon_prompt",
            "batch_forward_prompt",
            "batch_back_prompt",
            "batch_recon_prompt"
        ]:
            if key in config and not isinstance(config[key], str):
                raise ValueError(f"{key} must be a string if provided")

        # Optional prompt file
        if "prompt_file" in config and not isinstance(config["prompt_file"], str):
            raise ValueError("prompt_file must be a string path if provided")

        logger.info("Translation config validation passed")
    
    @staticmethod
    def validate_analysis_config(config: Dict[str, Any]) -> None:
        """
        Validate analysis configuration.
        
        Args:
            config: Configuration dictionary
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ["models", "distance_metric"]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in analysis config: {key}")
        
        # Validate models
        models = config["models"]
        if not isinstance(models, list) or len(models) == 0:
            raise ValueError("Models must be a non-empty list")
        
        for model in models:
            if not isinstance(model, dict):
                raise ValueError("Each model must be a dictionary")
            
            if "name" not in model or "checkpoint" not in model:
                raise ValueError("Each model must have 'name' and 'checkpoint' keys")
        
        # Validate distance metric
        valid_metrics = ["kl_half", "euclidean", "manhattan", "cosine"]
        metric = config["distance_metric"]
        
        if metric not in valid_metrics:
            raise ValueError(f"Distance metric must be one of: {', '.join(valid_metrics)}")
        
        # Validate optional fields
        if "batch_size" in config:
            batch_size = config["batch_size"]
            if not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError("Batch size must be a positive integer")
        
        if "seed" in config:
            seed = config["seed"]
            if not isinstance(seed, int):
                raise ValueError("Seed must be an integer")
        
        logger.info("Analysis config validation passed")
    
    @staticmethod
    def validate_data_config(config: Dict[str, Any]) -> None:
        """
        Validate data configuration.
        
        Args:
            config: Configuration dictionary
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ["input_file"]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in data config: {key}")
        
        # Validate input file
        if not isinstance(config["input_file"], str) or not config["input_file"].strip():
            raise ValueError("Input file must be a non-empty string")
        
        # Validate required columns
        if "required_columns" in config:
            columns = config["required_columns"]
            if not isinstance(columns, list):
                raise ValueError("Required columns must be a list")
        
        logger.info("Data config validation passed")

