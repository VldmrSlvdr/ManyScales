#!/usr/bin/env python3
"""
List available Gemini models.

Usage:
    python scripts/list_gemini_models.py
"""

import os
import sys
from pathlib import Path
import requests
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def list_gemini_models(api_key: str = None):
    """List available Gemini models."""
    # Get API key
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    # Try loading from config if not in env
    if not api_key:
        try:
            from llm_translate.config import ConfigLoader
            config = ConfigLoader.load_yaml(Path("configs/translation_config.yaml"))
            api_key = config.get("gemini_api_key")
        except:
            pass
    
    if not api_key:
        print("✗ Error: GEMINI_API_KEY not found")
        print("\nPlease set it as an environment variable or in configs/translation_config.yaml")
        print("  export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    
    print(f"Using API key: {api_key[:10]}...{api_key[-4:]}")
    print("\nFetching available Gemini models...\n")
    
    # Try different API versions
    api_versions = ["v1beta", "v1"]
    
    for version in api_versions:
        print(f"=== Trying API version: {version} ===")
        url = f"https://generativelanguage.googleapis.com/{version}/models?key={api_key}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if "models" in data:
                models = data["models"]
                print(f"✓ Found {len(models)} models\n")
                
                for model in models:
                    name = model.get("name", "Unknown")
                    display_name = model.get("displayName", "")
                    description = model.get("description", "")
                    supported_methods = model.get("supportedGenerationMethods", [])
                    
                    # Extract model ID from full name (e.g., "models/gemini-pro" -> "gemini-pro")
                    model_id = name.split("/")[-1] if "/" in name else name
                    
                    print(f"Model ID: {model_id}")
                    if display_name:
                        print(f"  Display Name: {display_name}")
                    if description:
                        print(f"  Description: {description[:100]}...")
                    if supported_methods:
                        print(f"  Supported Methods: {', '.join(supported_methods)}")
                    print()
                
                # Show which models support generateContent
                content_models = [
                    m.get("name").split("/")[-1] 
                    for m in models 
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                
                if content_models:
                    print(f"Models supporting 'generateContent' (usable for translation):")
                    for m in content_models:
                        print(f"  - {m}")
                    print()
                
            else:
                print(f"⚠ No models found in response")
                print(f"Response: {json.dumps(data, indent=2)[:500]}")
            
            print()
            
        except requests.exceptions.HTTPError as e:
            print(f"✗ HTTP Error: {e}")
            try:
                error_data = e.response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Response text: {e.response.text[:500]}")
            print()
        
        except Exception as e:
            print(f"✗ Error: {e}")
            print()


if __name__ == "__main__":
    print("=" * 70)
    print("Gemini Models List")
    print("=" * 70)
    print()
    
    list_gemini_models()

