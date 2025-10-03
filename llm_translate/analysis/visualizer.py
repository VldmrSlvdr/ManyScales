"""Visualization utilities for translation analysis."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances


logger = logging.getLogger(__name__)


class Visualizer:
    """Create visualizations for translation analysis."""
    
    def __init__(self, output_dir: Optional[Path] = None, fig_format: str = "png"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures (None = don't save)
            fig_format: Figure format (png, pdf, svg)
        """
        self.output_dir = output_dir
        self.fig_format = fig_format
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_distribution_comparison(
        self,
        df: pd.DataFrame,
        model_name: str,
        metric: str = "kl_half",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot distribution of metric by translation type.
        
        Args:
            df: DataFrame with metrics
            model_name: Model name to plot
            metric: Metric column name
            filename: Filename to save (None = auto-generate)
        """
        logger.info(f"Plotting distribution comparison for {model_name}")
        
        df_model = df[df["model"] == model_name]
        
        vals_llm = df_model[df_model["trans_type"] == "llm"][metric].values
        vals_hum = df_model[df_model["trans_type"] == "human"][metric].values
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot([vals_llm, vals_hum], labels=["LLM", "Human"])
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by translation type — {model_name}")
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if self.output_dir:
            fname = filename or f"distribution_{model_name}_{metric}.{self.fig_format}"
            save_path = self.output_dir / fname
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure: {save_path}")
        
        plt.close()
    
    def plot_mds_centroids(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        model_name: str,
        filename: Optional[str] = None
    ) -> None:
        """
        Plot MDS map of group centroids.
        
        Args:
            df: DataFrame with item metadata
            embeddings: Aligned embeddings
            model_name: Model name for title
            filename: Filename to save (None = auto-generate)
        """
        logger.info(f"Creating MDS plot for {model_name}")
        
        # Compute centroids
        centroids = []
        labels = []
        
        # English
        mask_en = (df["lang"] == "en").values
        if mask_en.sum() > 0:
            centroids.append(embeddings[mask_en].mean(axis=0))
            labels.append("EN")
        
        # Japanese LLM
        mask_llm = ((df["lang"] == "ja") & (df["trans_type"] == "llm")).values
        if mask_llm.sum() > 0:
            centroids.append(embeddings[mask_llm].mean(axis=0))
            labels.append("JP-LLM")
        
        # Japanese Human
        mask_hum = ((df["lang"] == "ja") & (df["trans_type"] == "human")).values
        if mask_hum.sum() > 0:
            centroids.append(embeddings[mask_hum].mean(axis=0))
            labels.append("JP-HUMAN")
        
        if len(centroids) < 2:
            logger.warning("Not enough centroids for MDS plot")
            return
        
        C = np.vstack(centroids)
        D = pairwise_distances(C, metric="euclidean")
        
        # MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=2025)
        Y = mds.fit_transform(D)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(Y[:, 0], Y[:, 1], s=200, alpha=0.7)
        
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (Y[i, 0], Y[i, 1]),
                fontsize=12,
                ha="center",
                va="center"
            )
        
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(f"MDS map of centroids — {model_name}")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if self.output_dir:
            fname = filename or f"mds_{model_name}.{self.fig_format}"
            save_path = self.output_dir / fname
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure: {save_path}")
        
        plt.close()
    
    def plot_model_comparison(
        self,
        df: pd.DataFrame,
        metric: str = "kl_half",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot comparison across models.
        
        Args:
            df: DataFrame with metrics
            metric: Metric column name
            filename: Filename to save
        """
        logger.info(f"Plotting model comparison for {metric}")
        
        models = sorted(df["model"].unique())
        trans_types = ["llm", "human"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, trans_type in enumerate(trans_types):
            ax = axes[idx]
            df_type = df[df["trans_type"] == trans_type]
            
            data = [df_type[df_type["model"] == m][metric].values for m in models]
            
            ax.boxplot(data, labels=models)
            ax.set_ylabel(metric)
            ax.set_xlabel("Model")
            ax.set_title(f"{trans_type.upper()} translations")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if self.output_dir:
            fname = filename or f"model_comparison_{metric}.{self.fig_format}"
            save_path = self.output_dir / fname
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure: {save_path}")
        
        plt.close()
    
    def plot_correlation_heatmap(
        self,
        correlation_matrix: np.ndarray,
        labels: list,
        title: str = "Model Agreement (Spearman)",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            labels: Labels for rows/columns
            title: Plot title
            filename: Filename to save
        """
        logger.info("Creating correlation heatmap")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(correlation_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation", rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(
                    j, i, f"{correlation_matrix[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=10
                )
        
        ax.set_title(title)
        plt.tight_layout()
        
        if self.output_dir:
            fname = filename or f"correlation_heatmap.{self.fig_format}"
            save_path = self.output_dir / fname
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved figure: {save_path}")
        
        plt.close()

