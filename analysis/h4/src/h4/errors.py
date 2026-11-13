"""Errors that stop the pipeline before an estimate is written."""


class ConfigError(ValueError):
    """The declarative configuration asks for an analysis this repo will not run."""


class SchemaError(ValueError):
    """The item table or a derived table violates the canonical schema."""


class TierSeparationError(RuntimeError):
    """A result was about to cross the confirmatory, robustness, or exploratory boundary."""
