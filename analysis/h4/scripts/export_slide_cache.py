"""Write the published tables the notebook and the R report plot.

The confirmatory cosine file is not read or rewritten. Token matrices are
read from the exploratory cache produced when a config sets
``hara_pooling_analysis.save_token_embeddings: true``. Pass that directory
with ``--tokens``. Sentence vectors for the UMAP coordinates are read with
``--embeddings``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from h4.exploratory.distributional_distance import pairwise_euclidean, wasserstein


def main() -> None:
    parser = argparse.ArgumentParser(description="Export slide-walkthrough caches.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    distances = _token_distances(args.results, args.tokens)
    distances.to_csv(args.out / "token_item_distance.csv", index=False)
    _example_cloud(args.tokens, args.out / "token_cloud_example.csv")
    _pack_primary_tokens(args.tokens, distances, args.out / "mpnet_primary_tokens.npz")
    _copy_tables(args.results, args.out)
    _judgment_slopes(args.results, args.out / "judgment_slopes.csv")
    _umap_coordinates(args.embeddings, args.out / "umap_ja_displacement.csv")
    print(
        {
            "token_rows": int(len(distances)),
            "primary_rows": int(distances["in_primary"].sum()),
            "models": sorted(distances["short_name"].unique()),
        }
    )


def _token_distances(results: Path, tokens: Path) -> pd.DataFrame:
    pairs = pd.read_csv(results / "exploratory/pooling/token_distance_pairs.csv")
    water = pd.read_csv(results / "exploratory/geometry/wasserstein.csv")
    pairs["scale_id"] = pairs["scale_id"].map(_key)
    pairs["item_id"] = pairs["item_id"].map(_key)
    water["scale_id"] = water["scale_id"].map(_key)
    water["item_id"] = water["item_id"].map(_key)
    water = water.loc[
        :,
        ["scale_id", "item_id", "comparison", "short_name", "distance_human", "distance_llm", "delta_distance"],
    ]
    merged = pairs.merge(water, on=["scale_id", "item_id", "comparison", "short_name"], how="left")
    merged["in_primary"] = merged["scale_id"].ne("11")
    merged["w1_gap"] = -(merged["distance_human"] - merged["distance_llm"])
    order2 = _wasserstein2(tokens, merged)
    merged = merged.merge(order2, on=["scale_id", "item_id", "comparison", "short_name"], how="left")
    keep = [
        "scale_id",
        "item_id",
        "comparison",
        "short_name",
        "in_primary",
        "d_mu_human",
        "d_mu_llm",
        "delta_d_mu",
        "d_sigma_human",
        "d_sigma_llm",
        "delta_d_sigma",
        "distance_human",
        "distance_llm",
        "w1_gap",
        "w2_human",
        "w2_llm",
        "w2_gap",
    ]
    return merged.loc[:, keep].sort_values(["short_name", "comparison", "scale_id", "item_id"])


def _wasserstein2(tokens: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Squared-Euclidean transport on the stored MPNet back-translation clouds."""

    rows = []
    part = frame.loc[frame["short_name"].eq("mpnet") & frame["comparison"].eq("back_translation")]
    for record in part.itertuples(index=False):
        stem = f"{record.scale_id}_{record.item_id}"
        original = np.load(tokens / "mpnet" / f"{stem}_original_en.npy")
        human = _w2(original, np.load(tokens / "mpnet" / f"{stem}_human_back_en.npy"))
        llm = _w2(original, np.load(tokens / "mpnet" / f"{stem}_llm_back_en.npy"))
        rows.append(
            {
                "scale_id": record.scale_id,
                "item_id": record.item_id,
                "comparison": record.comparison,
                "short_name": record.short_name,
                "w2_human": human,
                "w2_llm": llm,
                "w2_gap": -(human - llm),
            }
        )
    return pd.DataFrame(rows)


def _w2(reference: np.ndarray, other: np.ndarray) -> float:
    cost = pairwise_euclidean(reference, other)
    transported = wasserstein(cost * cost)["distance"]
    return float(np.sqrt(max(transported, 0.0)))


def _example_cloud(tokens: Path, path: Path) -> None:
    rows = []
    for field in ("original_en", "human_back_en", "llm_back_en"):
        matrix = np.load(tokens / "mpnet" / f"01_01_{field}.npy")
        for index, vector in enumerate(matrix):
            row = {"scale_id": "01", "item_id": "01", "field": field, "token_index": index}
            row.update({f"d{axis}": float(value) for axis, value in enumerate(vector)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _pack_primary_tokens(tokens: Path, distances: pd.DataFrame, path: Path) -> None:
    primary = distances.loc[
        distances["short_name"].eq("mpnet")
        & distances["comparison"].eq("back_translation")
        & distances["in_primary"]
    ]
    payload = {}
    for record in primary.itertuples(index=False):
        stem = f"{record.scale_id}_{record.item_id}"
        for field in ("original_en", "human_back_en", "llm_back_en"):
            payload[f"{stem}_{field}"] = np.load(tokens / "mpnet" / f"{stem}_{field}.npy")
    np.savez_compressed(path, **payload)


def _copy_tables(results: Path, out: Path) -> None:
    curve = pd.read_csv(results / "no_tivs/tables/specification_curve.csv")
    curve.to_csv(out / "specification_curve_primary.csv", index=False)
    beta = pd.read_csv(results / "no_tivs/tables/beta_estimate.csv")
    beta.loc[
        :,
        ["comparison", "mean_human", "mean_llm", "precision", "mean_difference", "ci_low", "ci_high", "n_items", "n_scales"],
    ].to_csv(out / "beta_density_primary.csv", index=False)
    pd.read_csv(results / "no_tivs/tables/bootstrap_summary.csv").to_csv(out / "bootstrap_primary.csv", index=False)
    pd.read_csv(results / "wasserstein_models/comparison.csv").to_csv(out / "wasserstein_comparison.csv", index=False)


def _judgment_slopes(results: Path, path: Path) -> None:
    folders = {
        "primary_21": results / "delta_z_no_tivs/fits",
        "with_tivs_22": results / "delta_z/fits",
        "without_heema_tivs_20": results / "delta_z_no_heema_tivs/fits",
    }
    names = {
        "naturalness_gap": "naturalness",
        "expert_semantic_gap": "semantic",
        "expert_cultural_validity_gap": "cultural_validity",
        "expert_choice_fit_gap": "choice_fit",
        "expert_naturalness_gap": "expert_naturalness",
        "preference": "preference",
        "guess": "guess",
    }
    rows = []
    for sample, folder in folders.items():
        for stem, outcome in names.items():
            coef = pd.read_csv(folder / f"{stem}_coef.csv")
            hit = coef.loc[coef["term"].eq("delta_z")].iloc[0]
            rows.append(
                {
                    "sample": sample,
                    "outcome": outcome,
                    "estimate": hit["estimate"],
                    "se": hit["se"],
                    "ci_low": hit["ci_low"],
                    "ci_high": hit["ci_high"],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _umap_coordinates(embeddings: Path, path: Path) -> None:
    import umap

    frame = pd.read_parquet(embeddings)
    frame["scale_id"] = frame["scale_id"].map(_key)
    frame["item_id"] = frame["item_id"].map(_key)
    frame = frame.loc[frame["scale_id"].ne("11") & frame["text_field"].isin(["original_en", "human_ja", "llm_ja"])]
    keys = ["scale_id", "item_id"]
    original = frame.loc[frame["text_field"].eq("original_en"), [*keys, "embedding"]]
    rows = []
    for field in ("human_ja", "llm_ja"):
        other = frame.loc[frame["text_field"].eq(field), [*keys, "embedding"]]
        paired = original.merge(other, on=keys, suffixes=("_original", "_translation"))
        for record in paired.itertuples(index=False):
            difference = np.asarray(record.embedding_translation, dtype=np.float64) - np.asarray(
                record.embedding_original, dtype=np.float64
            )
            rows.append(
                {
                    "scale_id": record.scale_id,
                    "item_id": record.item_id,
                    "text_field": field,
                    "displacement": float(np.linalg.norm(difference)),
                    "vector": difference,
                }
            )
    built = pd.DataFrame(rows)
    matrix = np.vstack(built["vector"].to_numpy())
    projected = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.5,
        metric="euclidean",
        random_state=20260928,
    ).fit_transform(matrix)
    built["dim1"] = projected[:, 0]
    built["dim2"] = projected[:, 1]
    built["dim3"] = projected[:, 2]
    built.drop(columns=["vector"]).to_csv(path, index=False)


def _key(value: object) -> str:
    token = str(value).strip()
    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]
    if token.isdigit():
        return token.zfill(2)
    return token


if __name__ == "__main__":
    main()
