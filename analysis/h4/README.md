# H4: embedding similarity

H4 compares each English original with the English back-translations of the human and LLM Japanese versions. The primary instrument is MPNet cosine similarity; the item gap is human minus LLM, fitted with a random scale intercept by REML.

The primary result is **−0.041 (95% CI [−0.060, −0.022])**, across 486 items in 21 scales. Exploratory models relate each translation's own similarity to ordinal ratings and binary outcomes.

- [Japanese report](../../reports/h4_ja.html) · [English report](../../reports/h4_en.html)
- `notebooks/`: report sources and the detailed R walkthrough
- `src/h4/`, `scripts/`, `tests/`: pipeline, fitting/plotting helpers, and tests
- `configs/`: study and synthetic demonstration configurations

Study inputs, aggregates, embeddings, and model weights are not bundled. The [OSF project](https://osf.io/hn4fr/) is the data-release location; upload is pending. Follow the [replication guide](../../docs/reproduce.md) to attach inputs and run the pipeline.

From the repository root, tests need only the software dependencies:

```sh
python -m pip install -e 'analysis/h4[dev]'
(cd analysis/h4 && python -m pytest -q)
```

OCQ exclusions affect only OCQ responses. Cultural-validity export scores already run from low to high validity and receive no additional reversal.
