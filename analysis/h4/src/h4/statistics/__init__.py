"""Statistical layer for H4. Embeddings are not computed here."""

from h4.statistics.bootstrap import hierarchical_bootstrap
from h4.statistics.mixed import fit_hierarchical
from h4.statistics.paired import descriptive_paired_ttest, wide_delta
from h4.statistics.specification_curve import specification_curve

__all__ = [
    "descriptive_paired_ttest",
    "fit_hierarchical",
    "hierarchical_bootstrap",
    "specification_curve",
    "wide_delta",
]
