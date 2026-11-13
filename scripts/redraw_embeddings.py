#!/usr/bin/env python3
"""Redraw the three displacement-space figures from saved projection coordinates."""
import argparse
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AXES = ["dim1", "dim2", "dim3"]

def draw(centers: pd.DataFrame, method: str, out: Path) -> None:
    # Slide/report-sized quantitative grid; categorical IDs also encode scale identity.
    fig = plt.figure(figsize=(11.5, 8.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    colors = list(plt.colormaps["tab20"].colors) + [(0.15, 0.15, 0.15)]
    scale_names = []
    for color, (sid, group) in zip(colors, centers.groupby("scale_id", sort=True)):
        h = group.loc[group.text_field.eq("human_ja"), AXES].iloc[0].to_numpy(float)
        l = group.loc[group.text_field.eq("llm_ja"), AXES].iloc[0].to_numpy(float)
        ax.plot(*np.vstack([h, l]).T, color=color, linewidth=1.5)
        ax.scatter(*h, s=45, color=[color], marker="^", depthshade=False)
        ax.scatter(*l, s=38, color=[color], marker="s", depthshade=False)
        ax.text(*((h + l) / 2), sid, fontsize=8, color="#222222")
        scale_names.append(f"{sid}  {group.scale_name.iloc[0]}")
    ax.set(xlabel=f"{method} 1", ylabel=f"{method} 2", zlabel=f"{method} 3")
    ax.view_init(elev=18, azim=-58)
    ax.set_title(f"{method}: scale centers of Japanese displacement", fontsize=15, pad=18)
    handles = [Line2D([], [], color="#333333", linestyle="none", marker=marker,
                      label=label, markersize=7)
               for marker, label in [("^", "Human Japanese"), ("s", "LLM Japanese")]]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.94), frameon=False)
    fig.text(0.79, 0.79, "Scale key (scale 11 excluded)", fontsize=10, weight="bold")
    for i, name in enumerate(scale_names):
        fig.text(0.79, 0.755 - i * 0.029, name, fontsize=9)
    fig.text(0.07, 0.06, "Displacement = Japanese embedding − original English embedding.\n"
             "972 item-version vectors; 486 items; 21 scales. Centers are means after projection.", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.76, bottom=0.13, top=0.91)
    stem = out / f"displacement_ja_{method.lower().replace('-', '')}_3d"
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate saved projections without drawing")
    args = parser.parse_args()
    out = ROOT / "figures/embeddings"
    names = pd.read_csv(ROOT / "analysis/scale_names.csv", dtype=str).set_index("scale_id").scale_name
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    for key, method in [("tsne", "t-SNE"), ("umap", "UMAP"), ("mds", "MDS")]:
        coords = pd.read_csv(out / f"{key}_item_coordinates.csv", dtype={"scale_id": str, "item_id": str})
        centers = pd.read_csv(out / f"{key}_scale_centers.csv", dtype={"scale_id": str})
        assert len(coords) == 972 and len(centers) == 42
        assert coords.scale_id.nunique() == 21 and not coords.scale_id.eq("11").any()
        calculated = coords.groupby(["scale_id", "text_field"])[AXES].mean().sort_index()
        stored = centers.set_index(["scale_id", "text_field"])[AXES].sort_index()
        # t-SNE means were computed in float32 before CSV export.
        np.testing.assert_allclose(calculated, stored, rtol=1e-6, atol=5e-6)
        centers["scale_name"] = centers.scale_id.map(names)
        assert centers.scale_name.notna().all()
        if not args.check:
            draw(centers, method, out)
        print(f"Validated {method}: 972 coordinates and 42 scale centers")


if __name__ == "__main__":
    main()
