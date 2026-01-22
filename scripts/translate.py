#!/usr/bin/env python3
"""
Script for translating survey items using LLMs.

Usage:
    python scripts/translate.py --config configs/translation_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_translate.config import ConfigLoader, ConfigValidator
from llm_translate.translation import Translator
from llm_translate.utils import setup_logging, load_data, save_results


def main():
    parser = argparse.ArgumentParser(description="Translate survey items using LLMs")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to translation configuration YAML file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input file path (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (overrides config)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load config
    config = ConfigLoader.load_yaml(args.config)
    # Merge prompt file if provided
    config = ConfigLoader.maybe_load_prompt_file(config, base_dir=args.config.parent)
    ConfigValidator.validate_translation_config(config)
    
    # Override with CLI arguments
    if args.input:
        config["input_file"] = str(args.input)
    if args.output:
        config["output_file"] = str(args.output)
    
    # Load input data
    input_file = Path(config["input_file"])
    df = load_data(input_file, required_columns=["item_id", "text"])
    
    # Initialize translator
    translator = Translator(
        model=config["model"],
        openai_api_key=config.get("openai_api_key"),
        gemini_api_key=config.get("gemini_api_key"),
        temperature=config.get("temperature", 0.0),
        effort=config.get("effort", "medium")
    )
    
    # Translate
    do_back = bool(config.get("do_back", False))
    do_recon = bool(config.get("do_recon", False)) and do_back
    batch_mode = bool(config.get("batch_mode", False))
    batch_size = int(config.get("batch_size", len(df))) if batch_mode else len(df)

    print(
        f"Translating {len(df)} items from {config['source_lang']} to {config['target_lang']}"
        + (" (batch mode)" if batch_mode else "")
    )
    
    translations = []
    context = config.get("context")

    if batch_mode:
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start + batch_size]
            texts = chunk["text"].tolist()
            item_ids = chunk["item_id"].tolist()

            print(f"  [batch {start+1}-{start+len(chunk)}] Translating {len(chunk)} items...")

            forward_outs = translator.batch_translate_llm(
                texts=texts,
                source_lang=config.get("source_lang", "en"),
                target_lang=config["target_lang"],
                context=context,
                prompt_override=config.get("batch_forward_prompt")
            )

            back_outs = None
            recon_outs = None
            if do_back:
                back_outs = translator.batch_back_translate_llm(
                    texts=forward_outs,
                    source_lang=config.get("source_lang", "en"),
                    target_lang=config["target_lang"],
                    prompt_override=config.get("batch_back_prompt")
                )

            if do_recon and back_outs is not None:
                recon_outs = translator.batch_reconcile_llm(
                    originals=texts,
                    forwards=forward_outs,
                    backs=back_outs,
                    source_lang=config.get("source_lang", "en"),
                    target_lang=config["target_lang"],
                    prompt_override=config.get("batch_recon_prompt")
                )

            for idx, (item_id, text, forward_out) in enumerate(zip(item_ids, texts, forward_outs)):
                back_out = back_outs[idx] if back_outs is not None else None
                recon_out = recon_outs[idx]["revised"] if recon_outs is not None else None
                recon_expl = recon_outs[idx]["explanation"] if recon_outs is not None else None
                translations.append({
                    "item_id": item_id,
                    "original": text,
                    "forward": forward_out,
                    **({"back": back_out} if back_out is not None else {}),
                    **({"reconciled": recon_out, "recon_explanation": recon_expl} if recon_out is not None else {}),
                    "model": config["model"],
                    "target_lang": config["target_lang"]
                })
    else:
        for idx, row in df.iterrows():
            text = row["text"]
            
            print(f"  [{idx+1}/{len(df)}] Translating: {text[:50]}...")
            
            # Optional custom forward prompt
            forward_prompt = config.get("forward_prompt")
            if forward_prompt:
                fwd_prompt = (
                    forward_prompt.replace("{from_lang}", config.get("source_lang", "en"))
                    .replace("{to_lang}", config["target_lang"])
                    .replace("{text}", text)
                )
                forward_out = translator.call_prompt(fwd_prompt)
            else:
                forward_out = translator.translate(
                    text=text,
                    target_lang=config["target_lang"],
                    context=context
                )
            
            back_out = None
            recon_out = None
            recon_expl = None

            if do_back:
                back_prompt = config.get("back_prompt")
                if back_prompt:
                    b_prompt = (
                        back_prompt.replace("{from_lang}", config.get("source_lang", "en"))
                        .replace("{to_lang}", config["target_lang"])
                        .replace("{text}", forward_out)
                    )
                    back_out = translator.call_prompt(b_prompt)
                else:
                    back_out = translator.back_translate(forward_out, source_lang=config.get("source_lang", "en"))
            
            if do_recon and back_out is not None:
                recon_prompt = config.get("recon_prompt")
                if recon_prompt:
                    # Build the header per recon default and append sections
                    r_body = (
                        recon_prompt.replace("{from_lang}", config.get("source_lang", "en"))
                        .replace("{to_lang}", config["target_lang"])
                        .strip()
                    )
                    full_prompt = "\n".join([
                        r_body,
                        "",
                        "---",
                        "ORIGINAL:",
                        text,
                        "BACK:",
                        back_out,
                        "FORWARD:",
                        forward_out,
                    ])
                    parsed = translator._parse_reconciliation(translator.call_prompt(full_prompt))
                    recon_out = parsed.get("revised")
                    recon_expl = parsed.get("explanation")
                else:
                    parsed = translator.reconcile(
                        original=text,
                        translation_a=forward_out,
                        translation_b=back_out,
                        target_lang=config["target_lang"]
                    )
                    recon_out = parsed.get("revised")
                    recon_expl = parsed.get("explanation")
            
            translations.append({
                "item_id": row["item_id"],
                "original": text,
                "forward": forward_out,
                **({"back": back_out} if back_out is not None else {}),
                **({"reconciled": recon_out, "recon_explanation": recon_expl} if recon_out is not None else {}),
                "model": config["model"],
                "target_lang": config["target_lang"]
            })
    
    # Save results
    import pandas as pd
    results_df = pd.DataFrame(translations)
    
    output_file = Path(config.get("output_file", "outputs/translations.csv"))
    save_results(results_df, output_file)
    
    print(f"\nTranslation complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()

