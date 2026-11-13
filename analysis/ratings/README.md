# Ratings and psychometrics

`primary/` contains the H1–H3 QMD sources, R functions, preprocessing code, and exploratory checks. H3 applies the OCQ-specific exclusion. `supplementary/` contains additional model checks and Stan code.

Run primary sources from `primary/`. Run supplementary scripts from this directory, marked by `.here`. Primary H1/H3 report human minus LLM; supplementary models retain LLM minus human.

Inputs and saved results are local files from the [OSF project](https://osf.io/hn4fr/), excluded from Git. See the [replication guide](../../docs/reproduce.md), [credits](../../CONTRIBUTORS.md), and shared [MIT license](../../LICENSE).
