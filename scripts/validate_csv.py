#!/usr/bin/env python3
"""
Utility script to validate and fix CSV files with proper quoting.

Usage:
    python scripts/validate_csv.py --input data/my_file.csv --output data/my_file_fixed.csv
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def validate_and_fix_csv(input_file: Path, output_file: Path = None, force_quote: bool = True):
    """
    Validate and fix CSV file with proper quoting.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path (None = overwrite input)
        force_quote: Force quote all non-numeric fields
    """
    print(f"Reading CSV: {input_file}")
    
    try:
        # Read the CSV
        df = pd.read_csv(input_file)
        print(f"✓ Successfully loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for potential issues
        issues = []
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                has_commas = df[col].astype(str).str.contains(',').any()
                has_quotes = df[col].astype(str).str.contains('"').any()
                has_newlines = df[col].astype(str).str.contains('\n').any()
                
                if has_commas:
                    issues.append(f"  - Column '{col}' contains commas")
                if has_quotes:
                    issues.append(f"  - Column '{col}' contains quotes")
                if has_newlines:
                    issues.append(f"  - Column '{col}' contains newlines")
        
        if issues:
            print("\n⚠ Found potential issues:")
            for issue in issues:
                print(issue)
        else:
            print("\n✓ No quoting issues detected")
        
        # Save with proper quoting
        output_file = output_file or input_file
        
        if force_quote:
            # Quote all non-numeric fields
            df.to_csv(
                output_file,
                index=False,
                quoting=1,  # QUOTE_ALL for text fields
                escapechar='\\'
            )
            print(f"\n✓ Saved properly quoted CSV to: {output_file}")
        else:
            # Quote only when necessary
            df.to_csv(
                output_file,
                index=False,
                quoting=0,  # QUOTE_MINIMAL
                escapechar='\\'
            )
            print(f"\n✓ Saved CSV with minimal quoting to: {output_file}")
        
        # Verify by re-reading
        df_verify = pd.read_csv(output_file)
        if df.equals(df_verify):
            print("✓ Verification passed: File reads correctly")
        else:
            print("⚠ Warning: Re-read data differs slightly (may be due to type conversions)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Validate and fix CSV files with proper quoting"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path (default: overwrite input)"
    )
    parser.add_argument(
        "--no-force-quote",
        action="store_true",
        help="Don't force quote all fields, only quote when necessary"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"✗ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    validate_and_fix_csv(
        args.input,
        args.output,
        force_quote=not args.no_force_quote
    )


if __name__ == "__main__":
    main()

