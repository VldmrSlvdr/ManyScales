"""Load the declarative analysis configuration.

The preregistered primary instrument is fixed in three places that must
agree: ``FROZEN_PRIMARY_MODEL``, ``configs/models.yaml``, and
``configs/analysis.yaml``. The loader refuses configurations that would
choose a model from the human-versus-LLM result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from h4.errors import ConfigError

FROZEN_PRIMARY_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
FROZEN_PRIMARY_METRIC = "cosine"
FROZEN_PRIMARY_COMPARISON = "back_translation"
FROZEN_ESTIMAND = "human_similarity - llm_similarity"

FORBIDDEN_KEYS = frozenset(
    {
        "select_by_h4",
        "choose_best_model",
        "maximize_human_advantage",
        "pick_model_after_results",
    }
)

ALLOWED_TIERS = frozenset({"confirmatory", "robustness", "exploratory"})
SENTENCE_FIELDS = (
    "original_en",
    "human_back_en",
    "llm_back_en",
    "human_ja",
    "llm_ja",
)
STAGE_ORDER = (
    "validate",
    "embed",
    "similarity",
    "metric_validation",
    "confirmatory",
    "robustness",
    "figures",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    short_name: str
    family: str
    role: str
    normalize: bool
    expected_dimension: int | None = None
    dimension: int | None = None
    revision: str | None = None
    instruction: str | None = None
    prompt_policy: str | None = None

    @property
    def slug(self) -> str:
        return self.short_name


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_more_similar: bool
    independent_geometry: bool
    note: str = ""


@dataclass(frozen=True)
class Config:
    root: Path
    study_id: str
    preregistered: bool
    seed: int
    confidence_level: float
    primary: ModelSpec
    robustness_models: tuple[ModelSpec, ...]
    primary_metric: MetricSpec
    sensitivity_metrics: tuple[MetricSpec, ...]
    comparison_fields: dict[str, dict[str, str]]
    direct_crosslingual: bool
    bootstrap_iterations: int
    bootstrap_hierarchy: tuple[str, ...]
    validation_flags: dict[str, bool]
    exploratory: dict[str, Any]
    paths: dict[str, Path]
    runtime: dict[str, Any]
    engine: str
    expected_direction: str
    source_name: str

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        return (self.primary, *self.robustness_models)

    def with_paths(self, **updates: Path | str) -> Config:
        paths = dict(self.paths)
        for key, value in updates.items():
            paths[key] = Path(value)
        return replace(self, paths=paths)


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = _read_yaml(config_path)
    _reject_forbidden(raw)
    root = config_path.parent.parent
    paths = _resolve_paths(root, raw.get("paths") or {})
    models_doc = _read_yaml(paths["model_registry"])
    metrics_doc = _read_yaml(paths["metric_registry"])
    _reject_forbidden(models_doc)
    _reject_forbidden(metrics_doc)

    study = raw.get("study") or {}
    tiers = raw.get("analysis_tiers") or {}
    confirmatory = tiers.get("confirmatory") or {}
    robustness = tiers.get("robustness") or {}
    preregistered = bool(study.get("preregistered", False))

    primary = _primary_spec(models_doc)
    robustness_models = _robustness_specs(models_doc)
    _check_model_lists(raw, primary, robustness_models, preregistered)
    _check_confirmatory_family(primary, preregistered)

    primary_metric, sensitivity = _metrics(metrics_doc)
    metric_name = ((confirmatory.get("metric") or {}).get("name"))
    if metric_name != primary_metric.name:
        raise ConfigError(
            "configs/analysis.yaml metric.name must match configs/metrics.yaml primary metric."
        )
    if preregistered and primary_metric.name != FROZEN_PRIMARY_METRIC:
        raise ConfigError(
            f"The preregistered metric is {FROZEN_PRIMARY_METRIC}."
        )

    estimand = confirmatory.get("estimand") or {}
    expression = estimand.get("expression")
    direction = estimand.get("expected_direction")
    if expression != FROZEN_ESTIMAND:
        raise ConfigError(
            f"The implemented estimand is {FROZEN_ESTIMAND}."
        )
    if direction != "positive":
        raise ConfigError(
            "H4 is evaluated as human similarity minus LLM similarity, with a positive expected direction."
        )

    inference = confirmatory.get("inference") or {}
    kind = inference.get("method", inference.get("model"))
    if kind != "hierarchical":
        raise ConfigError(
            "Primary inference is the hierarchical random-intercept model."
        )
    if inference.get("grouping", "scale_id") != "scale_id":
        raise ConfigError("Primary grouping factor is scale_id.")
    engine = inference.get("engine", "statsmodels_mixedlm")
    if engine not in {"statsmodels_mixedlm", "bambi"}:
        raise ConfigError("inference.engine must be statsmodels_mixedlm or bambi.")
    confidence = float(inference.get("confidence_level", 0.95))
    if not 0 < confidence < 1:
        raise ConfigError("confidence_level must lie strictly between 0 and 1.")

    comparison = confirmatory.get("comparison") or {}
    if tuple(comparison.get(key) for key in ("source", "human", "llm")) != (
        "original_en",
        "human_back_en",
        "llm_back_en",
    ):
        raise ConfigError(
            "The confirmatory comparison is original_en against human_back_en and llm_back_en."
        )

    direct = robustness.get("direct_crosslingual") or {}
    direct_enabled = bool(direct.get("enabled", False))
    if direct_enabled and tuple(direct.get(key) for key in ("source", "human", "llm")) != (
        "original_en",
        "human_ja",
        "llm_ja",
    ):
        raise ConfigError(
            "Direct cross-lingual robustness compares original_en with human_ja and llm_ja."
        )

    bootstrap = robustness.get("hierarchical_bootstrap") or {}
    hierarchy = list(bootstrap.get("hierarchy") or [])
    if hierarchy != ["scale_id", "item_id"]:
        raise ConfigError(
            "The implemented hierarchical bootstrap resamples scale_id, then items within the scale."
        )
    iterations = int(bootstrap.get("iterations", 0))
    if iterations < 1:
        raise ConfigError("hierarchical_bootstrap.iterations must be positive.")

    runtime = dict(raw.get("runtime") or {})
    if preregistered and runtime.get("dtype", "float32") != "float32":
        raise ConfigError("The preregistered run uses float32 embeddings.")

    validation = raw.get("validation") or {}
    flags = {
        "identical_pairs": bool(validation.get("identical_pairs", False)),
        "within_scale_permutation": bool(validation.get("within_scale_permutation", False)),
        "between_scale_permutation": bool(validation.get("between_scale_permutation", False)),
        "negation_test": bool(validation.get("negation_test", False)),
    }
    if preregistered and not all(flags.values()):
        raise ConfigError(
            "The preregistered analysis keeps identical, within-scale, between-scale, and negation controls on."
        )

    exploratory = dict(tiers.get("exploratory") or {})
    return Config(
        root=root,
        study_id=str(study.get("id", "manyscales_h4")),
        preregistered=preregistered,
        seed=int(raw.get("seed", 0)),
        confidence_level=confidence,
        primary=primary,
        robustness_models=tuple(robustness_models),
        primary_metric=primary_metric,
        sensitivity_metrics=tuple(sensitivity),
        comparison_fields={
            "back_translation": {
                "reference": "original_en",
                "human": "human_back_en",
                "llm": "llm_back_en",
            },
            "direct_crosslingual": {
                "reference": "original_en",
                "human": "human_ja",
                "llm": "llm_ja",
            },
        },
        direct_crosslingual=direct_enabled,
        bootstrap_iterations=iterations,
        bootstrap_hierarchy=tuple(hierarchy),
        validation_flags=flags,
        exploratory=exploratory,
        paths=paths,
        runtime=runtime,
        engine=engine,
        expected_direction="positive",
        source_name=config_path.name,
    )


def assign_tier(comparison: str, model_name: str, metric: str, config: Config) -> str:
    """Label one specification. Exploratory work never receives a tier here."""

    if (
        comparison == FROZEN_PRIMARY_COMPARISON
        and model_name == config.primary.name
        and metric == config.primary_metric.name
    ):
        return "confirmatory"
    return "robustness"


def attach_tier(frame, tier: str):
    if tier not in ALLOWED_TIERS:
        from h4.errors import TierSeparationError

        raise TierSeparationError(
            "analysis_tier must be confirmatory, robustness, or exploratory."
        )
    out = frame.copy()
    out["analysis_tier"] = tier
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ConfigError(f"{path} must contain a YAML mapping.")
    return document


def _reject_forbidden(document: Any) -> None:
    if isinstance(document, dict):
        for key, value in document.items():
            if str(key) in FORBIDDEN_KEYS:
                raise ConfigError(
                    f"Configuration key {key!r} would select a model from the H4 result."
                )
            _reject_forbidden(value)
    elif isinstance(document, list):
        for item in document:
            _reject_forbidden(item)


def _resolve_paths(root: Path, raw_paths: dict[str, Any]) -> dict[str, Path]:
    required = ("items", "processed", "results", "model_registry", "metric_registry")
    missing = [key for key in required if key not in raw_paths]
    if missing:
        raise ConfigError(f"paths is missing: {', '.join(missing)}")
    resolved: dict[str, Path] = {}
    for key, value in raw_paths.items():
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        resolved[key] = path
    return resolved


def _model_from_mapping(raw: dict[str, Any], role: str) -> ModelSpec:
    family = str(raw["family"])
    if role == "confirmatory" and family == "api":
        raise ConfigError(
            "Confirmatory H4 uses a pinned open-weight embedder. API embeddings stay off that path."
        )
    return ModelSpec(
        name=str(raw["name"]),
        short_name=str(raw.get("short_name", raw["name"])),
        family=family,
        role=role,
        normalize=bool(raw.get("normalize", True)),
        expected_dimension=(
            int(raw["expected_dimension"]) if raw.get("expected_dimension") is not None else None
        ),
        dimension=int(raw["dimension"]) if raw.get("dimension") is not None else None,
        revision=raw.get("revision"),
        instruction=raw.get("instruction"),
        prompt_policy=raw.get("prompt_policy"),
    )


def _primary_spec(models_doc: dict[str, Any]) -> ModelSpec:
    block = (models_doc.get("models") or {}).get("primary")
    if not isinstance(block, dict):
        raise ConfigError("models.primary is missing from the model registry.")
    return _model_from_mapping(block, "confirmatory")


def _robustness_specs(models_doc: dict[str, Any]) -> list[ModelSpec]:
    block = (models_doc.get("models") or {}).get("robustness") or []
    specs = []
    for raw in block:
        spec = _model_from_mapping(raw, "robustness")
        if spec.family == "api":
            raise ConfigError(
                "API embeddings are not part of the default robustness registry. Add one in a separate, explicit config if needed."
            )
        specs.append(spec)
    return specs


def _check_model_lists(
    raw: dict[str, Any],
    primary: ModelSpec,
    robustness_models: list[ModelSpec],
    preregistered: bool,
) -> None:
    tiers = raw.get("analysis_tiers") or {}
    declared = (tiers.get("confirmatory") or {}).get("embedding", {}).get("model")
    if declared != primary.name:
        raise ConfigError(
            "The confirmatory embedding in the analysis config must match models.primary."
        )
    declared_robust = list(
        ((tiers.get("robustness") or {}).get("alternative_models") or [])
    )
    registry_robust = [spec.name for spec in robustness_models]
    if declared_robust != registry_robust:
        raise ConfigError(
            "robustness.alternative_models must match the model registry, in the same order."
        )
    if preregistered and primary.name != FROZEN_PRIMARY_MODEL:
        raise ConfigError(
            f"The preregistered embedding model is {FROZEN_PRIMARY_MODEL}."
        )
    if preregistered and primary.family != "sentence_transformer":
        raise ConfigError("The preregistered embedder family is sentence_transformer.")


def _check_confirmatory_family(primary: ModelSpec, preregistered: bool) -> None:
    if primary.family == "api":
        raise ConfigError("Confirmatory H4 refuses API embeddings.")
    if preregistered and primary.family == "deterministic":
        raise ConfigError(
            "The deterministic embedder is a software stand-in and cannot be the preregistered instrument."
        )


def _metrics(metrics_doc: dict[str, Any]) -> tuple[MetricSpec, list[MetricSpec]]:
    block = metrics_doc.get("similarity") or {}
    primary_raw = block.get("primary")
    if not isinstance(primary_raw, dict):
        raise ConfigError("metrics.yaml is missing similarity.primary.")
    primary = MetricSpec(
        name=str(primary_raw["name"]),
        higher_is_more_similar=bool(primary_raw.get("higher_is_more_similar", True)),
        independent_geometry=bool(primary_raw.get("independent_geometry", True)),
        note=str(primary_raw.get("note", "")),
    )
    sensitivity = []
    for raw in block.get("sensitivity") or []:
        sensitivity.append(
            MetricSpec(
                name=str(raw["name"]),
                higher_is_more_similar=bool(raw.get("higher_is_more_similar", False)),
                independent_geometry=bool(raw.get("independent_geometry", False)),
                note=str(raw.get("note", "")),
            )
        )
    return primary, sensitivity
