#!/usr/bin/env python3
"""
Script for analyzing translation quality using embeddings.

Usage:
    python scripts/analyze.py --config configs/analysis_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_translate.config import ConfigLoader, ConfigValidator
from llm_translate.analysis import Embedder, MetricsCalculator, StatisticalAnalyzer, Visualizer
from llm_translate.utils import setup_logging, load_data, save_results, set_seed, ensure_dir, validate_item_dataframe


def main():
    parser = argparse.ArgumentParser(description="Analyze translation quality")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to analysis configuration YAML file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input file path (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (overrides config)"
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
    ConfigValidator.validate_analysis_config(config)
    
    # Override with CLI arguments
    if args.input:
        config["input_file"] = str(args.input)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    
    # Set seed
    seed = config.get("seed", 2025)
    set_seed(seed)
    
    # Setup directories
    output_dir = ensure_dir(Path(config.get("output_dir", "outputs")))
    cache_dir = ensure_dir(Path(config.get("cache_dir", "cache/embeddings")))
    figures_dir = ensure_dir(output_dir / "figures")
    
    # Load input data
    print("Loading input data...")
    input_file = Path(config["input_file"])
    df = load_data(input_file)
    validate_item_dataframe(df)
    
    print(f"Loaded {len(df)} items")
    print(f"  Languages: {df['lang'].unique()}")
    print(f"  Translation types: {df['trans_type'].unique()}")
    
    # Extract model configurations
    models = [(m["name"], m["checkpoint"]) for m in config["models"]]
    print(f"\nUsing {len(models)} embedding models:")
    for name, checkpoint in models:
        print(f"  - {name}: {checkpoint}")
    
    # Generate embeddings
    print("\n=== Generating Embeddings ===")
    embedder = Embedder(cache_dir=cache_dir)
    
    texts = df["text"].tolist()
    embeddings_dict = embedder.embed_multiple_models(
        texts=texts,
        models=models,
        device=config.get("device"),
        batch_size=config.get("batch_size", 64)
    )
    
    # Whitening and alignment
    print("\n=== Whitening and Alignment ===")
    reference_model = config.get("reference_model", models[0][0])
    
    aligned_embeddings, transform_params = embedder.whiten_and_align(
        embeddings_dict=embeddings_dict,
        reference_model=reference_model
    )
    
    # Compute metrics
    print("\n=== Computing Metrics ===")
    metrics_calc = MetricsCalculator()
    distance_metric = config["distance_metric"]
    
    metrics_df = metrics_calc.compute_item_metrics(
        df=df,
        aligned_embeddings=aligned_embeddings,
        distance_metric=distance_metric,
        n_draws=config.get("n_draws", 200),
        seed=seed
    )
    
    print(f"Computed metrics for {len(metrics_df)} item-model pairs")
    
    # Statistical analysis
    print("\n=== Statistical Analysis ===")
    stats_analyzer = StatisticalAnalyzer()
    
    # Paired comparisons
    paired_results = stats_analyzer.paired_comparison(metrics_df, metric=distance_metric)
    print(f"Paired comparison results:")
    print(paired_results.to_string(index=False))
    
    # Cross-model agreement
    agreement = stats_analyzer.cross_model_agreement(metrics_df, metric=distance_metric)
    print(f"\nCross-model agreement (ICC): {agreement['icc']:.3f}")
    print("Spearman correlation matrix:")
    print(agreement["spearman_matrix"])
    
    # Summary statistics
    summary = stats_analyzer.summary_statistics(metrics_df, metric=distance_metric, group_by="model")
    
    # Visualizations
    if config.get("create_visualizations", True):
        print("\n=== Creating Visualizations ===")
        visualizer = Visualizer(output_dir=figures_dir, fig_format=config.get("fig_format", "png"))
        
        # Distribution comparisons
        for model_name in metrics_df["model"].unique():
            visualizer.plot_distribution_comparison(
                metrics_df, model_name, metric=distance_metric
            )
        
        # Model comparison
        visualizer.plot_model_comparison(metrics_df, metric=distance_metric)
        
        # MDS plots
        for model_name in aligned_embeddings.keys():
            visualizer.plot_mds_centroids(
                df, aligned_embeddings[model_name], model_name
            )
        
        # Correlation heatmap
        if len(agreement["models"]) > 1:
            visualizer.plot_correlation_heatmap(
                agreement["spearman_matrix"],
                agreement["models"],
                title="Model Agreement (Spearman)"
            )
        
        print(f"Figures saved to: {figures_dir}")
    
    # Save results
    print("\n=== Saving Results ===")
    save_results(metrics_df, output_dir / "item_metrics.csv")
    save_results(paired_results, output_dir / "paired_tests.csv")
    save_results(summary, output_dir / "summary_statistics.csv")
    
    # Save agreement results
    import numpy as np
    np.savetxt(output_dir / "spearman_matrix.txt", agreement["spearman_matrix"], fmt="%.4f")
    
    import json
    with open(output_dir / "agreement.json", "w") as f:
        json.dump({
            "models": agreement["models"],
            "icc": float(agreement["icc"])
        }, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"  - item_metrics.csv")
    print(f"  - paired_tests.csv")
    print(f"  - summary_statistics.csv")
    print(f"  - spearman_matrix.txt")
    print(f"  - agreement.json")


if __name__ == "__main__":
    main()

