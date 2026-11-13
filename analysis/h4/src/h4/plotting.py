"""Figures for the specification curve, scale estimates, and exploratory scatter."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_figures(results: Path) -> list[Path]:
    written: list[Path] = []
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    curve_path = results / "tables" / "specification_curve.csv"
    if curve_path.exists():
        written.extend(_specification_curve(pd.read_csv(curve_path), figures))
    scale_path = results / "tables" / "scale_estimates.csv"
    if scale_path.exists():
        written.extend(
            _scale_estimates(
                pd.read_csv(scale_path),
                figures,
                stem="scale_estimates",
                title="Scale-level human − LLM differences",
                xlabel="Mean item Δ within scale (human − LLM)",
            )
        )
    direct_path = results / "tables" / "direct_scale_estimates.csv"
    if direct_path.exists():
        written.extend(
            _scale_estimates(
                pd.read_csv(direct_path),
                figures,
                stem="direct_scale_estimates",
                title="Direct test: original versus Japanese translations",
                xlabel="Mean item Δ within scale (human Japanese − LLM Japanese)",
            )
        )
    beta_path = results / "tables" / "beta_estimate.csv"
    if beta_path.exists():
        written.extend(_beta_estimates(pd.read_csv(beta_path), figures))
    written.extend(_embedding_maps(results, figures))
    socm_path = results / "exploratory" / "pooling" / "socm_pairs.csv"
    if socm_path.exists():
        written.extend(_socm_scatter(pd.read_csv(socm_path), figures))
    distance_path = results / "exploratory" / "pooling" / "token_distance_pairs.csv"
    if distance_path.exists():
        written.extend(_token_distance_scatter(pd.read_csv(distance_path), figures))
    return written


def _specification_curve(table: pd.DataFrame, figures: Path) -> list[Path]:
    independent = table.loc[table["metric"].astype(str) == "cosine"].copy()
    if independent.empty:
        return []
    independent = independent.iloc[::-1].reset_index(drop=True)
    independent["label"] = (
        independent["short_name"].astype(str) + " / " + independent["comparison"].astype(str)
    )
    figure, axis = plt.subplots(figsize=(7.2, max(3.2, 0.45 * len(independent) + 1.4)))
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    for tier, color, size in (("robustness", "#4C78A8", 6), ("confirmatory", "#1B1B1B", 8)):
        part = independent.loc[independent["analysis_tier"] == tier]
        if part.empty:
            continue
        axis.errorbar(
            part["estimate"].to_numpy(),
            part.index.to_numpy(),
            xerr=_xerr(part),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=size,
            label=tier.capitalize(),
        )
    axis.set_yticks(independent.index.tolist())
    axis.set_yticklabels(independent["label"].tolist())
    axis.set_xlabel("Hierarchical estimate of human − LLM similarity")
    axis.set_title("Specification curve")
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    figure.tight_layout()
    return _save(figure, figures, "specification_curve")


def _scale_estimates(
    table: pd.DataFrame,
    figures: Path,
    *,
    stem: str,
    title: str,
    xlabel: str,
) -> list[Path]:
    ordered = table.sort_values("mean_delta")
    labels = _scale_labels(ordered)
    figure, axis = plt.subplots(figsize=(8.4, max(3.4, 0.38 * len(ordered) + 1.3)))
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.plot(ordered["mean_delta"], range(len(ordered)), "o", color="#1B1B1B")
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels(labels)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    figure.tight_layout()
    return _save(figure, figures, stem)


def _scale_labels(table: pd.DataFrame) -> list[str]:
    if "scale_name" not in table.columns:
        return table["scale_id"].astype(str).tolist()
    names = table["scale_name"].astype(str).str.strip()
    missing = names.eq("") | names.str.lower().eq("nan")
    return names.where(~missing, table["scale_id"].astype(str)).tolist()


def _beta_estimates(table: pd.DataFrame, figures: Path) -> list[Path]:
    from scipy import stats

    labels = {
        "back_translation": "Back-translation",
        "direct_crosslingual": "Direct Japanese",
    }
    present = table["comparison"].astype(str).tolist()
    order = [name for name in ("back_translation", "direct_crosslingual") if name in present]
    order.extend(name for name in present if name not in order)
    if not order:
        return []
    figure = plt.figure(figsize=(8.8, 6.6))
    grid = figure.add_gridspec(2, len(order), height_ratios=[1.45, 1.0], hspace=0.38, wspace=0.22)
    density_axes = []
    x = np.linspace(0.45, 0.98, 500)
    for column, comparison in enumerate(order):
        row = table.loc[table["comparison"].astype(str) == comparison].iloc[0]
        axis = figure.add_subplot(grid[0, column])
        density_axes.append(axis)
        curves = []
        handles = []
        for mean, color, name in (
            (float(row["mean_human"]), "#1B1B1B", "Human"),
            (float(row["mean_llm"]), "#4C78A8", "LLM"),
        ):
            precision = float(row["precision"])
            shape_a = mean * precision
            shape_b = (1.0 - mean) * precision
            density = stats.beta.pdf(x, shape_a, shape_b)
            (line,) = axis.plot(x, density, color=color, linewidth=1.8, label=name)
            axis.axvline(mean, color=color, linewidth=0.7, linestyle="--")
            handles.append(line)
            curves.append(density[np.isfinite(density)])
        finite = np.concatenate(curves) if curves else np.array([1.0])
        axis.set_ylim(0, float(np.max(finite)) * 1.08)
        axis.set_xlim(0.45, 1.0)
        axis.set_title(labels.get(comparison, comparison))
        axis.set_xlabel("Cosine similarity")
        if column == 0:
            axis.set_ylabel("Fitted beta density")
        difference = float(row["mean_difference"])
        axis.text(
            0.03,
            0.97,
            f"Δ = {difference:.3f}\n95% CI [{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
    figure.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.98),
        borderaxespad=0.0,
    )
    forest = figure.add_subplot(grid[1, :])
    forest.axvline(0.0, color="#444444", linewidth=0.8)
    positions = np.arange(len(order))[::-1]
    for position, comparison in zip(positions, order):
        row = table.loc[table["comparison"].astype(str) == comparison].iloc[0]
        estimate = float(row["mean_difference"])
        forest.errorbar(
            estimate,
            position,
            xerr=[[estimate - float(row["ci_low"])], [float(row["ci_high"]) - estimate]],
            fmt="o",
            color="#1B1B1B",
            ecolor="#1B1B1B",
            capsize=3,
            markersize=6,
        )
    forest.set_yticks(positions)
    forest.set_yticklabels([labels.get(name, name) for name in order])
    forest.set_xlabel("Beta mean difference (human − LLM)")
    forest.set_title("Bootstrap interval for the beta test")
    figure.subplots_adjust(right=0.84, hspace=0.42, wspace=0.28, top=0.92)
    return _save(figure, figures, "beta_estimate")


_FIELD_STYLE = {
    "original_en": ("Original English", "#1B1B1B", "o"),
    "human_ja": ("Human Japanese", "#4C78A8", "^"),
    "llm_ja": ("LLM Japanese", "#F58518", "s"),
    "human_back_en": ("Human back-translation", "#54A24B", "D"),
    "llm_back_en": ("LLM back-translation", "#E45756", "v"),
}


def _embedding_maps(results: Path, figures: Path) -> list[Path]:
    cache = results / "sentence_embeddings" / "mpnet.parquet"
    if not cache.exists():
        return []
    from h4.io import read_embeddings

    frame = _projection_frame(read_embeddings(cache), results)
    written = _project_and_draw(
        frame,
        figures,
        metric="cosine",
        drawers=(_draw_embedding_map_2d, _draw_embedding_map_3d),
    )
    written.extend(
        _project_and_draw(
            _difference_frame(frame),
            figures,
            metric="euclidean",
            perplexity=8,
            early_exaggeration=24,
            drawers=(_draw_difference_map_2d, _draw_difference_map_3d),
        )
    )
    written.extend(
        _project_and_draw(
            _difference_frame(frame, fields=("human_ja", "llm_ja")),
            figures,
            metric="euclidean",
            perplexity=8,
            early_exaggeration=24,
            drawers=(_draw_difference_ja_map_2d, _draw_difference_ja_map_3d),
        )
    )
    written.extend(_umap_difference_maps(_difference_frame(frame, fields=("human_ja", "llm_ja")), figures))
    return written


def _project_and_draw(
    frame: pd.DataFrame,
    figures: Path,
    *,
    metric: str,
    drawers: tuple,
    perplexity: float = 30,
    early_exaggeration: float = 12,
) -> list[Path]:
    from sklearn.manifold import TSNE

    matrix = np.vstack([np.asarray(vector, dtype=np.float64) for vector in frame["embedding"]])
    written: list[Path] = []
    for components, drawer in zip((2, 3), drawers):
        projected = TSNE(
            n_components=components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            metric=metric,
            init="pca",
            learning_rate="auto",
            max_iter=750,
            random_state=20260928,
        ).fit_transform(matrix)
        coordinates = frame.drop(columns=["embedding"]).copy()
        for axis_index in range(components):
            coordinates[f"dim{axis_index + 1}"] = projected[:, axis_index]
        written.extend(drawer(coordinates, figures))
    return written


def _difference_frame(
    frame: pd.DataFrame,
    fields: tuple[str, ...] = ("human_ja", "llm_ja", "human_back_en", "llm_back_en"),
) -> pd.DataFrame:
    """One row per translation, with the embedding replaced by translation minus original."""

    keys = ["scale_id", "item_id"]
    original = frame.loc[frame["text_field"] == "original_en", [*keys, "scale_name", "embedding"]]
    rows: list[pd.DataFrame] = []
    for field in fields:
        other = frame.loc[frame["text_field"] == field, [*keys, "embedding"]]
        paired = original.merge(other, on=keys, suffixes=("_original", "_translation"))
        difference = [
            np.asarray(translation, dtype=np.float64) - np.asarray(source, dtype=np.float64)
            for source, translation in zip(paired["embedding_original"], paired["embedding_translation"])
        ]
        built = paired[keys].copy()
        built["scale_name"] = paired["scale_name"]
        built["text_field"] = field
        built["embedding"] = difference
        built["displacement"] = [float(np.linalg.norm(vector)) for vector in difference]
        rows.append(built)
    return pd.concat(rows, ignore_index=True)


def _projection_frame(embeddings: pd.DataFrame, results: Path) -> pd.DataFrame:
    frame = embeddings.copy()
    frame["scale_id"] = frame["scale_id"].map(_projection_scale_key)
    names = pd.Series(dtype=str)
    meta_path = results / "tables" / "scale_estimates.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "scale_name" in meta.columns:
            meta["scale_id"] = meta["scale_id"].map(_projection_scale_key)
            names = meta.drop_duplicates("scale_id").set_index("scale_id")["scale_name"].astype(str)
    frame["scale_name"] = frame["scale_id"].map(lambda key: names.get(key, key))
    return frame.reset_index(drop=True)


def _projection_scale_key(value: object) -> str:
    token = str(value).strip()
    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]
    if token.isdigit():
        return token.zfill(2)
    return token


def _scale_palette(scale_ids: list[str]) -> dict[str, tuple]:
    cmap = plt.colormaps["hsv"]
    return {scale_id: cmap(index / max(len(scale_ids), 1)) for index, scale_id in enumerate(scale_ids)}


def _scale_radii(coordinates: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    span = float(np.ptp(coordinates[columns].to_numpy()))
    floor = 0.03 * span if span > 0 else 0.1
    for scale_id, group in coordinates.groupby("scale_id", sort=True):
        center = group[columns].mean().to_numpy()
        distance = np.linalg.norm(group[columns].to_numpy() - center, axis=1)
        circle = max(float(np.sqrt(np.mean(distance**2))), floor)
        bloom = max(float(np.quantile(distance, 0.9)), circle)
        rows.append(
            {
                "scale_id": scale_id,
                "scale_name": str(group["scale_name"].iloc[0]),
                "radius": circle,
                "bloom": bloom,
                **{column: float(value) for column, value in zip(columns, center)},
            }
        )
    return pd.DataFrame(rows)


def _draw_difference_map_2d(coordinates: pd.DataFrame, figures: Path) -> list[Path]:
    return _draw_embedding_map_2d(
        coordinates,
        figures,
        stem="embedding_difference_2d",
        title="Displacement from the original",
        xlabel="t-SNE 1 of (translation − original)",
        ylabel="t-SNE 2 of (translation − original)",
    )


def _draw_difference_map_3d(coordinates: pd.DataFrame, figures: Path) -> list[Path]:
    return _draw_embedding_map_3d(
        coordinates,
        figures,
        stem="embedding_difference_3d",
        title="Displacement from the original",
    )


def _draw_difference_ja_map_2d(coordinates: pd.DataFrame, figures: Path) -> list[Path]:
    return _draw_embedding_map_2d(
        coordinates,
        figures,
        stem="embedding_difference_ja_2d",
        title="Japanese displacement from the original",
        xlabel="t-SNE 1 of (Japanese − original)",
        ylabel="t-SNE 2 of (Japanese − original)",
    )


def _umap_difference_maps(frame: pd.DataFrame, figures: Path) -> list[Path]:
    """Japanese displacements in UMAP, with a line joining the two translations of one item."""

    try:
        import umap
    except ImportError:
        return []
    matrix = np.vstack([np.asarray(vector, dtype=np.float64) for vector in frame["embedding"]])
    written: list[Path] = []
    for components, drawer in ((2, _draw_umap_pairs_2d), (3, _draw_umap_pairs_3d)):
        projected = umap.UMAP(
            n_components=components,
            n_neighbors=30,
            min_dist=0.5,
            metric="euclidean",
            random_state=20260928,
        ).fit_transform(matrix)
        coordinates = frame.drop(columns=["embedding"]).copy()
        for axis_index in range(components):
            coordinates[f"dim{axis_index + 1}"] = projected[:, axis_index]
        written.extend(drawer(coordinates, figures))
    return written


def _paired_japanese(coordinates: pd.DataFrame) -> pd.DataFrame:
    keys = ["scale_id", "item_id", "scale_name"]
    human = coordinates.loc[coordinates["text_field"] == "human_ja"]
    llm = coordinates.loc[coordinates["text_field"] == "llm_ja"]
    return human.merge(llm, on=keys, suffixes=("_h", "_l"))


def _draw_umap_pairs_2d(coordinates: pd.DataFrame, figures: Path) -> list[Path]:
    paired = _paired_japanese(coordinates)
    scales = sorted(paired["scale_id"].unique())
    columns = 6
    rows = int(np.ceil(len(scales) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.55, rows * 2.45))
    panels = np.atleast_1d(axes).ravel()
    for axis, scale_id in zip(panels, scales):
        group = paired.loc[paired["scale_id"] == scale_id]
        for row in group.itertuples(index=False):
            axis.plot([row.dim1_h, row.dim1_l], [row.dim2_h, row.dim2_l], color="#B3B3B3", lw=0.6, alpha=0.75, zorder=1)
        axis.scatter(group["dim1_h"], group["dim2_h"], s=16, c="#4C78A8", marker="^", linewidths=0, zorder=3)
        axis.scatter(group["dim1_l"], group["dim2_l"], s=14, c="#F58518", marker="s", linewidths=0, zorder=3)
        _zoom_to_points(axis, np.concatenate([group["dim1_h"], group["dim1_l"]]), np.concatenate([group["dim2_h"], group["dim2_l"]]))
        axis.set_title(str(group["scale_name"].iloc[0]), fontsize=8, pad=2)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in panels[len(scales) :]:
        axis.axis("off")
    figure.suptitle("Japanese displacement by scale", fontsize=12)
    _umap_legend(figure)
    figure.tight_layout(rect=(0, 0, 0.90, 0.96))
    return _save(figure, figures, "embedding_difference_ja_umap_2d")


def _draw_umap_pairs_3d(coordinates: pd.DataFrame, figures: Path) -> list[Path]:
    """One human center and one LLM center per scale, so the 3D view stays readable."""

    paired = _paired_japanese(coordinates)
    scales = sorted(paired["scale_id"].unique())
    palette = _scale_palette(scales)
    figure = plt.figure(figsize=(9.4, 7.6))
    axis = figure.add_subplot(111, projection="3d")
    for scale_id in scales:
        group = paired.loc[paired["scale_id"] == scale_id]
        color = palette[scale_id]
        human_center = group[["dim1_h", "dim2_h", "dim3_h"]].mean().to_numpy()
        llm_center = group[["dim1_l", "dim2_l", "dim3_l"]].mean().to_numpy()
        axis.plot(*np.vstack([human_center, llm_center]).T, color=color, lw=1.3, alpha=0.95)
        axis.scatter(*human_center, s=36, c=[color], marker="^", depthshade=False)
        axis.scatter(*llm_center, s=32, c=[color], marker="s", depthshade=False)
        midpoint = (human_center + llm_center) / 2
        axis.text(*midpoint, str(group["scale_name"].iloc[0]), fontsize=7, color="#222222")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_zlabel("UMAP 3")
    axis.set_title("Scale centers of Japanese displacement")
    axis.view_init(elev=18, azim=-58)
    _umap_legend(figure)
    figure.subplots_adjust(right=0.82)
    return _save(figure, figures, "embedding_difference_ja_umap_3d")


def _zoom_to_points(axis, x_values: np.ndarray, y_values: np.ndarray) -> None:
    x_low, x_high = float(np.min(x_values)), float(np.max(x_values))
    y_low, y_high = float(np.min(y_values)), float(np.max(y_values))
    x_pad = max((x_high - x_low) * 0.14, 0.08)
    y_pad = max((y_high - y_low) * 0.14, 0.08)
    axis.set_xlim(x_low - x_pad, x_high + x_pad)
    axis.set_ylim(y_low - y_pad, y_high + y_pad)
    axis.set_aspect("equal", adjustable="box")


def _umap_legend(figure) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#333333", markersize=7, label="Human Japanese"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#333333", markersize=7, label="LLM Japanese"),
    ]
    figure.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 0.98), borderaxespad=0.0)


def _draw_difference_ja_map_3d(coordinates: pd.DataFrame, figures: Path) -> list[Path]:
    return _draw_embedding_map_3d(
        coordinates,
        figures,
        stem="embedding_difference_ja_3d",
        title="Japanese displacement from the original",
    )


def _point_sizes(coordinates: pd.DataFrame, *, base: float) -> np.ndarray:
    if "displacement" not in coordinates.columns:
        return np.full(len(coordinates), base)
    length = coordinates["displacement"].to_numpy(dtype=np.float64)
    span = float(np.ptp(length))
    if span <= 0:
        return np.full(len(coordinates), base)
    return base + 28.0 * (length - float(length.min())) / span


def _draw_embedding_map_2d(
    coordinates: pd.DataFrame,
    figures: Path,
    *,
    stem: str = "embedding_map_2d",
    title: str = "Item embeddings by scale",
    xlabel: str = "t-SNE 1",
    ylabel: str = "t-SNE 2",
) -> list[Path]:
    from matplotlib.patches import Circle

    scales = _scale_radii(coordinates, ["dim1", "dim2"])
    palette = _scale_palette(scales["scale_id"].tolist())
    figure, axis = plt.subplots(figsize=(9.2, 7.4))
    for record in scales.itertuples(index=False):
        color = palette[record.scale_id]
        center = (record.dim1, record.dim2)
        axis.add_patch(Circle(center, record.bloom, facecolor=color, edgecolor="none", alpha=0.13, zorder=1))
        axis.add_patch(
            Circle(center, record.radius, facecolor="none", edgecolor=color, linewidth=1.15, alpha=0.95, zorder=2)
        )
        axis.text(
            record.dim1,
            record.dim2,
            record.scale_name,
            ha="center",
            va="center",
            fontsize=7,
            color="#222222",
            zorder=4,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
    handles = []
    for field, (label, color, marker) in _FIELD_STYLE.items():
        part = coordinates.loc[coordinates["text_field"] == field]
        if part.empty:
            continue
        handle = axis.scatter(
            part["dim1"],
            part["dim2"],
            s=_point_sizes(part, base=14),
            c=color,
            marker=marker,
            alpha=0.55,
            linewidths=0,
            zorder=3,
            label=label,
        )
        handles.append(handle)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="datalim")
    figure.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 0.98), borderaxespad=0.0)
    figure.tight_layout()
    return _save(figure, figures, stem)


def _draw_embedding_map_3d(
    coordinates: pd.DataFrame,
    figures: Path,
    *,
    stem: str = "embedding_map_3d",
    title: str = "Item embeddings by scale",
) -> list[Path]:
    scales = _scale_radii(coordinates, ["dim1", "dim2", "dim3"])
    palette = _scale_palette(scales["scale_id"].tolist())
    figure = plt.figure(figsize=(9.4, 7.6))
    axis = figure.add_subplot(111, projection="3d")
    theta = np.linspace(0, 2 * np.pi, 48)
    phi = np.linspace(0, np.pi, 24)
    for record in scales.itertuples(index=False):
        color = palette[record.scale_id]
        bloom_x = record.dim1 + record.bloom * np.outer(np.cos(theta), np.sin(phi))
        bloom_y = record.dim2 + record.bloom * np.outer(np.sin(theta), np.sin(phi))
        bloom_z = record.dim3 + record.bloom * np.outer(np.ones_like(theta), np.cos(phi))
        axis.plot_surface(bloom_x, bloom_y, bloom_z, color=color, alpha=0.045, linewidth=0, shade=False, antialiased=False)
        axis.plot(
            record.dim1 + record.radius * np.cos(theta),
            record.dim2 + record.radius * np.sin(theta),
            np.full_like(theta, record.dim3),
            color=color,
            linewidth=1.15,
            alpha=0.95,
        )
        axis.text(record.dim1, record.dim2, record.dim3 + 0.35 * record.radius, record.scale_name, fontsize=7, color="#222222")
    handles = []
    for field, (label, color, marker) in _FIELD_STYLE.items():
        part = coordinates.loc[coordinates["text_field"] == field]
        if part.empty:
            continue
        handle = axis.scatter(
            part["dim1"],
            part["dim2"],
            part["dim3"],
            s=_point_sizes(part, base=12),
            c=color,
            marker=marker,
            alpha=0.55,
            depthshade=False,
            label=label,
        )
        handles.append(handle)
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.set_zlabel("t-SNE 3")
    axis.set_title(title)
    axis.view_init(elev=18, azim=-58)
    figure.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 0.92), borderaxespad=0.0)
    figure.subplots_adjust(right=0.82)
    return _save(figure, figures, stem)


def _token_distance_scatter(table: pd.DataFrame, figures: Path) -> list[Path]:
    """Hara mean distance against second-order distance, for the frozen model."""

    subset = table.loc[table["short_name"] == "mpnet"].copy()
    if subset.empty:
        return []
    panels = (
        ("back_translation", "Back-translation"),
        ("direct_crosslingual", "Direct Japanese"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharex=True, sharey=True)
    for index, (axis, (comparison, title)) in enumerate(zip(axes, panels)):
        part = subset.loc[subset["comparison"] == comparison]
        axis.scatter(
            part["d_mu_human"],
            part["d_sigma_human"],
            s=16,
            c="#4C78A8",
            marker="^",
            alpha=0.55,
            linewidths=0,
            label="Human" if index == 0 else None,
        )
        axis.scatter(
            part["d_mu_llm"],
            part["d_sigma_llm"],
            s=16,
            c="#F58518",
            marker="s",
            alpha=0.55,
            linewidths=0,
            label="LLM" if index == 0 else None,
        )
        axis.set_title(title)
        axis.set_xlabel("Mean distance dμ")
        axis.set_ylabel("Second-order distance dΣ")
    figure.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 0.98), borderaxespad=0.0)
    figure.suptitle("Paired token distances from the original", fontsize=12)
    figure.tight_layout(rect=(0, 0, 0.88, 0.94))
    return _save(figure, figures, "exploratory_token_distances")


def _socm_scatter(table: pd.DataFrame, figures: Path) -> list[Path]:
    subset = table.loc[table["comparison"] == "back_translation"].copy()
    if subset.empty:
        return []
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for translation, color in (("human", "#1B1B1B"), ("llm", "#4C78A8")):
        part = subset.loc[subset["translation"] == translation]
        axis.scatter(part["cosine"], part["d_sigma"], s=28, c=color, label=translation)
    axis.set_xlabel("Cosine of mean token vectors")
    axis.set_ylabel("Second-order distance")
    axis.set_title("Exploratory: mean similarity and second-order difference")
    axis.legend(frameon=False)
    figure.tight_layout()
    return _save(figure, figures, "exploratory_cosine_vs_second_order")


def _xerr(frame: pd.DataFrame):
    if frame.empty:
        return None
    low = frame["estimate"] - frame["ci_low"]
    high = frame["ci_high"] - frame["estimate"]
    return [low.to_numpy(), high.to_numpy()]


def _save(figure, figures: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = figures / f"{stem}.{suffix}"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths
