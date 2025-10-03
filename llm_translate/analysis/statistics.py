"""Statistical analysis for comparing translations."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon


logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Perform statistical tests and comparisons."""
    
    @staticmethod
    def icc2_1(X: np.ndarray) -> float:
        """
        Compute ICC(2,1) - Intraclass Correlation Coefficient.
        
        Args:
            X: Matrix of shape (n_items, n_models)
        
        Returns:
            ICC(2,1) value
        """
        X = np.asarray(X, dtype=float)
        n, k = X.shape
        
        if n < 2 or k < 2:
            return np.nan
        
        mean_rows = X.mean(axis=1, keepdims=True)
        mean_cols = X.mean(axis=0, keepdims=True)
        grand = X.mean()
        
        MSR = k * ((mean_rows - grand) ** 2).sum() / (n - 1)
        MSC = n * ((mean_cols - grand) ** 2).sum() / (k - 1)
        MSE = ((X - mean_rows - mean_cols + grand) ** 2).sum() / ((n - 1) * (k - 1))
        
        icc = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
        
        return float(icc)
    
    def paired_comparison(
        self,
        df: pd.DataFrame,
        metric: str = "kl_half"
    ) -> pd.DataFrame:
        """
        Compare LLM vs Human translations using paired tests.
        
        Args:
            df: DataFrame with columns: model, item_id, trans_type, metric
            metric: Metric column name to compare
        
        Returns:
            DataFrame with comparison statistics per model
        """
        logger.info(f"Performing paired comparisons for metric: {metric}")
        
        models = df["model"].unique()
        results = []
        
        for model in models:
            df_model = df[df["model"] == model]
            stats = self._paired_compare_model(df_model, metric)
            
            if stats is not None:
                results.append({"model": model, "metric": metric, **stats})
        
        return pd.DataFrame(results)
    
    def _paired_compare_model(
        self,
        df_model: pd.DataFrame,
        metric: str
    ) -> Optional[Dict]:
        """Perform paired comparison for a single model."""
        llm_df = df_model[df_model["trans_type"] == "llm"].set_index("item_id")
        hum_df = df_model[df_model["trans_type"] == "human"].set_index("item_id")
        
        # Find common items
        common_items = sorted(set(llm_df.index) & set(hum_df.index))
        
        if not common_items:
            return None
        
        a = llm_df.loc[common_items, metric].values
        b = hum_df.loc[common_items, metric].values
        
        # Handle missing values
        valid_mask = ~(np.isnan(a) | np.isnan(b))
        a = a[valid_mask]
        b = b[valid_mask]
        
        if len(a) < 2:
            return None
        
        # Paired t-test
        try:
            t_stat, t_pval = ttest_rel(a, b)
        except Exception as e:
            logger.warning(f"T-test failed: {e}")
            t_stat, t_pval = np.nan, np.nan
        
        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            w_stat, w_pval = np.nan, np.nan
        
        return {
            "n": len(a),
            "mean_llm": float(np.mean(a)),
            "mean_human": float(np.mean(b)),
            "std_llm": float(np.std(a)),
            "std_human": float(np.std(b)),
            "t_stat": float(t_stat),
            "t_pval": float(t_pval),
            "w_stat": float(w_stat),
            "w_pval": float(w_pval)
        }
    
    def cross_model_agreement(
        self,
        df: pd.DataFrame,
        metric: str = "kl_half"
    ) -> Dict:
        """
        Compute cross-model agreement using Spearman correlation and ICC.
        
        Args:
            df: DataFrame with columns: model, item_id, trans_type, metric
            metric: Metric column name
        
        Returns:
            Dictionary with correlation matrix and ICC
        """
        logger.info(f"Computing cross-model agreement for metric: {metric}")
        
        # Pivot to wide format
        pivot = df.pivot_table(
            index=["item_id", "trans_type"],
            columns="model",
            values=metric
        ).dropna()
        
        if pivot.empty or len(pivot.columns) < 2:
            logger.warning("Not enough data for cross-model agreement")
            return {
                "models": [],
                "spearman_matrix": np.array([]),
                "icc": np.nan
            }
        
        models = list(pivot.columns)
        n_models = len(models)
        
        # Compute Spearman correlations
        spearman_matrix = np.zeros((n_models, n_models))
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i == j:
                    spearman_matrix[i, j] = 1.0
                else:
                    rho, _ = spearmanr(pivot[model1], pivot[model2])
                    spearman_matrix[i, j] = rho
        
        # Compute ICC(2,1)
        icc = self.icc2_1(pivot.values)
        
        return {
            "models": models,
            "spearman_matrix": spearman_matrix,
            "icc": icc
        }
    
    def summary_statistics(
        self,
        df: pd.DataFrame,
        metric: str = "kl_half",
        group_by: str = "model"
    ) -> pd.DataFrame:
        """
        Compute summary statistics grouped by specified column.
        
        Args:
            df: DataFrame with metrics
            metric: Metric column name
            group_by: Column to group by
        
        Returns:
            DataFrame with summary statistics
        """
        return df.groupby(group_by)[metric].agg([
            ("count", "count"),
            ("mean", "mean"),
            ("std", "std"),
            ("min", "min"),
            ("25%", lambda x: np.percentile(x, 25)),
            ("median", "median"),
            ("75%", lambda x: np.percentile(x, 75)),
            ("max", "max")
        ]).reset_index()

