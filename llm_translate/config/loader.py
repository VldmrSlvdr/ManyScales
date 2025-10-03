"""Configuration loading from YAML files."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and parse YAML configuration files."""
    
    @staticmethod
    def load_yaml(config_path: Path) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            Configuration dictionary
        """
        logger.info(f"Loading config from: {config_path}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Empty config file: {config_path}")
        
        logger.info(f"Loaded config with keys: {list(config.keys())}")
        return config
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], output_path: Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            output_path: Path to save YAML file
        """
        logger.info(f"Saving config to: {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info("Config saved successfully")
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        Later configs override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
        
        Returns:
            Merged configuration
        """
        result = {}
        
        for config in configs:
            result = ConfigLoader._deep_merge(result, config)
        
        return result

    @staticmethod
    def maybe_load_prompt_file(config: Dict[str, Any], base_dir: Path = Path(".")) -> Dict[str, Any]:
        """
        If config has 'prompt_file', load it and merge 'forward_prompt', 'back_prompt', 'recon_prompt'.
        Values in the prompt file override inline fields.
        """
        prompt_path = config.get("prompt_file")
        if not prompt_path:
            return config

        p = Path(prompt_path)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")

        with open(p, "r") as f:
            prompts = yaml.safe_load(f) or {}

        # Only merge known keys
        merged = config.copy()
        for k in ["forward_prompt", "back_prompt", "recon_prompt"]:
            if k in prompts:
                merged[k] = prompts[k]
        return merged
    
    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

