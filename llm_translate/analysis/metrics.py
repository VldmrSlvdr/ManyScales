"""Metrics calculation for translation quality assessment."""

import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various distance and similarity metrics."""
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between vectors.
        
        Args:
            a: First vector or matrix
            b: Second vector or matrix
        
        Returns:
            Cosine similarity values
        """
        a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
        b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
        return (a_norm * b_norm).sum(axis=-1)
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance between vectors.
        
        Args:
            a: First vector or matrix
            b: Second vector or matrix
        
        Returns:
            Euclidean distances
        """
        diff = a - b
        return np.sqrt((diff * diff).sum(axis=-1))
    
    @staticmethod
    def kl_divergence_centered_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute centered-Euclidean approximation to KL divergence: KL ≈ 0.5 * ||Δ||²
        
        Args:
            a: First vector or matrix
            b: Second vector or matrix
        
        Returns:
            KL approximation values
        """
        diff = a - b
        return 0.5 * (diff * diff).sum(axis=-1)
    
    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute Manhattan (L1) distance.
        
        Args:
            a: First vector or matrix
            b: Second vector or matrix
        
        Returns:
            Manhattan distances
        """
        return np.abs(a - b).sum(axis=-1)
    
    @staticmethod
    def neg_baseline_quantiles(
        X: np.ndarray,
        pos_pairs: List[Tuple[int, int]],
        n_draws: int = 200,
        seed: int = 2025
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute quantiles of positive pairs against negative baseline.
        
        Args:
            X: Embedding matrix
            pos_pairs: List of (index_a, index_b) positive pairs
            n_draws: Number of random negative samples
            seed: Random seed
        
        Returns:
            Tuple of (positive_similarities, quantiles)
        """
        if not pos_pairs:
            return np.array([]), np.array([])
        
        rng = np.random.RandomState(seed)
        
        en_idx = np.array([i for i, _ in pos_pairs])
        jp_idx = np.array([j for _, j in pos_pairs])
        
        # Positive similarities
        sims_pos = MetricsCalculator.cosine_similarity(X[en_idx], X[jp_idx])
        
        # Negative baseline
        neg_scores = []
        for _ in range(n_draws):
            perm = rng.permutation(len(jp_idx))
            neg_scores.append(
                MetricsCalculator.cosine_similarity(X[en_idx], X[jp_idx][perm])
            )
        
        neg_scores = np.concatenate(neg_scores)
        neg_sorted = np.sort(neg_scores)
        
        # Compute quantiles
        ranks = np.searchsorted(neg_sorted, sims_pos, side="right")
        quantiles = ranks / max(1, len(neg_scores))
        
        return sims_pos, quantiles
    
    def compute_item_metrics(
        self,
        df: pd.DataFrame,
        aligned_embeddings: Dict[str, np.ndarray],
        distance_metric: str = "kl_half",
        n_draws: int = 200,
        seed: int = 2025
    ) -> pd.DataFrame:
        """
        Compute item-level metrics for all models.
        
        Args:
            df: DataFrame with columns: item_id, lang, trans_type, text
            aligned_embeddings: Dictionary of model name to aligned embeddings
            distance_metric: Primary distance metric ("kl_half", "euclidean", "manhattan", "cosine")
            n_draws: Number of draws for quantile baseline
            seed: Random seed
        
        Returns:
            DataFrame with item-level metrics
        """
        logger.info(f"Computing item-level metrics with distance={distance_metric}")
        
        # Separate by language and type
        en_df = df[df["lang"] == "en"].copy().reset_index(drop=True)
        jp_llm = df[(df["lang"] == "ja") & (df["trans_type"] == "llm")].copy().reset_index(drop=True)
        jp_hum = df[(df["lang"] == "ja") & (df["trans_type"] == "human")].copy().reset_index(drop=True)
        
        # Find common item IDs
        common_ids_llm = sorted(set(en_df["item_id"]) & set(jp_llm["item_id"]))
        common_ids_hum = sorted(set(en_df["item_id"]) & set(jp_hum["item_id"]))
        
        # Build index maps
        idx_en = {row.item_id: row.Index for row in en_df.itertuples()}
        idx_llm = {row.item_id: row.Index for row in jp_llm.itertuples()}
        idx_hum = {row.item_id: row.Index for row in jp_hum.itertuples()}
        
        records = []
        
        for model_name, X in aligned_embeddings.items():
            # Build pairs
            pos_llm = [(idx_en[iid], idx_llm[iid]) for iid in common_ids_llm if iid in idx_llm]
            pos_hum = [(idx_en[iid], idx_hum[iid]) for iid in common_ids_hum if iid in idx_hum]
            
            # Compute quantiles
            sims_llm, q_llm = self.neg_baseline_quantiles(X, pos_llm, n_draws, seed)
            sims_hum, q_hum = self.neg_baseline_quantiles(X, pos_hum, n_draws, seed)
            
            # Compute distance metric
            dist_llm = self._compute_distances(X, pos_llm, distance_metric)
            dist_hum = self._compute_distances(X, pos_hum, distance_metric)
            
            # Record LLM translations
            for k, iid in enumerate(common_ids_llm):
                if k < len(dist_llm):
                    records.append({
                        "model": model_name,
                        "item_id": iid,
                        "trans_type": "llm",
                        "cosine": float(sims_llm[k]),
                        "quantile": float(q_llm[k]),
                        distance_metric: float(dist_llm[k])
                    })
            
            # Record human translations
            for k, iid in enumerate(common_ids_hum):
                if k < len(dist_hum):
                    records.append({
                        "model": model_name,
                        "item_id": iid,
                        "trans_type": "human",
                        "cosine": float(sims_hum[k]),
                        "quantile": float(q_hum[k]),
                        distance_metric: float(dist_hum[k])
                    })
        
        return pd.DataFrame(records)
    
    def _compute_distances(
        self,
        X: np.ndarray,
        pairs: List[Tuple[int, int]],
        metric: str
    ) -> np.ndarray:
        """Compute distances for pairs using specified metric."""
        if not pairs:
            return np.array([])
        
        idx_a = np.array([i for i, _ in pairs])
        idx_b = np.array([j for _, j in pairs])
        
        if metric == "kl_half":
            return self.kl_divergence_centered_euclidean(X[idx_a], X[idx_b])
        elif metric == "euclidean":
            return self.euclidean_distance(X[idx_a], X[idx_b])
        elif metric == "manhattan":
            return self.manhattan_distance(X[idx_a], X[idx_b])
        elif metric == "cosine":
            # Return 1 - cosine similarity as distance
            return 1.0 - self.cosine_similarity(X[idx_a], X[idx_b])
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

