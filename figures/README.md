# Manuscript figures

| Figure | Preview | Vector exports |
|---|---|---|
| H1: expert rating differences | [PNG](h1_ratings.png) | [PDF](h1_ratings.pdf), [SVG](h1_ratings.svg) |
| H3: citizen rating differences | [PNG](h3_ratings.png) | [PDF](h3_ratings.pdf), [SVG](h3_ratings.svg) |
| H4: mean item similarity gaps | [PNG](h4_similarity.png) | [PDF](h4_similarity.pdf), [SVG](h4_similarity.svg) |
| Japanese displacement space | [UMAP](embeddings/displacement_ja_umap_3d.png), [t-SNE](embeddings/displacement_ja_tsne_3d.png), [MDS](embeddings/displacement_ja_mds_3d.png) | PDF/SVG files beside each preview |

Differences are human minus LLM. H1/H3 show Bayesian 95% CrI; H4 shows frequentist 95% CI. Blue favors LLM, orange favors human, and gray includes zero. The final row is the overall estimate.

Numerical tables and projection coordinates belong to the [OSF data release](https://osf.io/hn4fr/) and are excluded from GitHub. After attaching local inputs, `make figures` redraws H1/H3 and synchronizes H4; `python scripts/redraw_embeddings.py` redraws the descriptive 3D views from saved coordinates.
