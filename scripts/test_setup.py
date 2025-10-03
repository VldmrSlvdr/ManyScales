#!/usr/bin/env python3
"""
Test script to verify the setup without making API calls.

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from llm_translate.translation import Translator, ModelSpec
        from llm_translate.analysis import Embedder, MetricsCalculator, StatisticalAnalyzer, Visualizer
        from llm_translate.config import ConfigLoader, ConfigValidator
        from llm_translate.utils import setup_logging, load_data, save_results, set_seed
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_csv_loading():
    """Test loading sample CSV files."""
    print("\nTesting CSV loading...")
    try:
        from llm_translate.utils import load_data
        
        # Test sample_items.csv
        df1 = load_data("data/sample_items.csv")
        print(f"✓ Loaded sample_items.csv: {len(df1)} rows")
        
        # Check for commas in text
        for idx, row in df1.iterrows():
            text = row['text']
            if ',' in text:
                print(f"  - Row {idx+1} has comma: '{text[:50]}...'")
        
        # Test sample_analysis.csv
        df2 = load_data("data/sample_analysis.csv")
        print(f"✓ Loaded sample_analysis.csv: {len(df2)} rows")
        
        return True
    except Exception as e:
        print(f"✗ CSV loading error: {e}")
        return False


def test_config_loading():
    """Test loading configuration files."""
    print("\nTesting config loading...")
    try:
        from llm_translate.config import ConfigLoader, ConfigValidator
        
        # Translation config
        trans_config = ConfigLoader.load_yaml(Path("configs/translation_config.yaml"))
        print(f"✓ Loaded translation config")
        print(f"  - Model: {trans_config.get('model')}")
        print(f"  - Source lang: {trans_config.get('source_lang')}")
        print(f"  - Target lang: {trans_config.get('target_lang')}")
        
        # Analysis config
        analysis_config = ConfigLoader.load_yaml(Path("configs/analysis_config.yaml"))
        print(f"✓ Loaded analysis config")
        print(f"  - Models: {len(analysis_config.get('models', []))}")
        print(f"  - Distance metric: {analysis_config.get('distance_metric')}")
        
        return True
    except Exception as e:
        print(f"✗ Config loading error: {e}")
        return False


def test_api_keys():
    """Test API key availability."""
    print("\nChecking API keys...")
    import os
    
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if openai_key:
        print(f"✓ OPENAI_API_KEY found (length: {len(openai_key)})")
    else:
        print("⚠ OPENAI_API_KEY not found in environment")
    
    if gemini_key:
        print(f"✓ GEMINI_API_KEY found (length: {len(gemini_key)})")
    else:
        print("⚠ GEMINI_API_KEY not found in environment")
    
    if not openai_key and not gemini_key:
        print("\n⚠ No API keys found in environment variables.")
        print("  You can set them in your config files or as environment variables:")
        print("    export OPENAI_API_KEY='your-key-here'")
        print("    export GEMINI_API_KEY='your-key-here'")
        return False
    
    return True


def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    deps = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("sentence_transformers", "SentenceTransformers"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML"),
        ("requests", "requests"),
    ]
    
    all_ok = True
    for module_name, display_name in deps:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} not installed")
            all_ok = False
    
    if not all_ok:
        print("\n⚠ Some dependencies are missing. Install with:")
        print("  pip install -r requirements.txt")
    
    return all_ok


def main():
    print("=" * 60)
    print("LLM Translate - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("CSV Loading", test_csv_loading),
        ("Config Loading", test_config_loading),
        ("API Keys", test_api_keys),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Make sure you have valid API keys")
        print("  2. Run: python scripts/translate.py --config configs/translation_config.yaml")
        print("  3. Run: python scripts/analyze.py --config configs/analysis_config.yaml")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

