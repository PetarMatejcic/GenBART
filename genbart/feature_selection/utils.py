"""Shared utilities for BART feature-selection routines."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np


UINT32_MAX = np.iinfo(np.uint32).max
VALID_IMPORTANCE_KINDS = ("raw", "logml")


def validate_xy(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce feature and response arrays.

    A one-dimensional X is interpreted as a single predictor and reshaped to
    ``(n_samples, 1)``. The response must be one-dimensional and have the same
    number of rows as X.
    """
    X_arr = np.asarray(X)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape((-1, 1))
    elif X_arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object.")

    y_arr = np.asarray(y)

    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D array-like object.")

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    return X_arr, y_arr


def validate_eval_data(
    X_eval: Any,
    y_eval: Any,
    *,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate evaluation data used by predictive-degradation selection."""
    X_eval_arr, y_eval_arr = validate_xy(X_eval, y_eval)

    if X_eval_arr.shape[1] != n_features:
        raise ValueError("X_eval must have the same number of columns as X.")

    return X_eval_arr, y_eval_arr


def get_feature_names(
    X: Any,
    feature_names: Sequence[str] | None,
    n_features: int,
) -> list[str]:
    """Resolve feature names from explicit names, DataFrame columns, or defaults."""
    if feature_names is not None:
        names = list(feature_names)
    elif hasattr(X, "columns"):
        names = list(X.columns)
    else:
        names = [f"x{j}" for j in range(n_features)]

    if len(names) != n_features:
        raise ValueError("feature_names must have length equal to X.shape[1].")

    return [str(name) for name in names]


def validate_model_class(model_cls: Any) -> None:
    """Validate that a model class or factory can be called."""
    if not callable(model_cls):
        raise TypeError("model_cls must be a callable class or factory.")


def get_model_spec(model: Any) -> tuple[type, dict]:
    """Extract a recreatable model class and parameter dictionary from an instance."""
    if isinstance(model, type):
        raise TypeError("Expected a model instance, not a model class.")

    if not hasattr(model, "get_params") or not callable(model.get_params):
        raise TypeError("model must implement get_params() so it can be recreated.")

    model_params = model.get_params()
    if not isinstance(model_params, dict):
        raise TypeError("model.get_params() must return a dictionary.")

    return type(model), model_params


def make_model(
    model_cls: Any,
    model_params: dict | None,
    *,
    random_state: int,
) -> Any:
    """Create a fresh model instance with a controlled random_state."""
    validate_model_class(model_cls)

    params = dict(model_params or {})
    params["random_state"] = int(random_state)

    return model_cls(**params)


def validate_importance_kind(importance_kind: str) -> None:
    """Validate the BART variable-inclusion importance kind."""
    if importance_kind not in VALID_IMPORTANCE_KINDS:
        allowed = ", ".join(repr(kind) for kind in VALID_IMPORTANCE_KINDS)
        raise ValueError(f"importance_kind must be one of {allowed}.")


def make_rng(random_state: int | None) -> np.random.Generator:
    """Create a NumPy random generator."""
    return np.random.default_rng(random_state)


def draw_uint32_seeds(
    rng: np.random.Generator,
    size: int | tuple[int, ...],
) -> np.ndarray:
    """Draw reproducible unsigned 32-bit seeds from an existing generator."""
    return rng.integers(
        low=0,
        high=UINT32_MAX,
        size=size,
        dtype=np.uint32,
    )


def make_permutation_selector_seeds(
    *,
    random_state: int | None,
    n_repeats: int,
    n_permutations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate seeds for permutation-null variable selection.

    Returns
    -------
    tuple
        Three arrays:

        - observed-response model seeds, shape ``(n_repeats,)``;
        - permuted-response model seeds, shape ``(n_permutations, n_repeats)``;
        - response-shuffle seeds, shape ``(n_permutations,)``.
    """
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    rng = make_rng(random_state)

    real_model_seeds = draw_uint32_seeds(rng, n_repeats)
    null_model_seeds = draw_uint32_seeds(rng, (n_permutations, n_repeats))
    shuffle_seeds = draw_uint32_seeds(rng, n_permutations)

    return real_model_seeds, null_model_seeds, shuffle_seeds


def make_repeated_model_seeds(
    *,
    random_state: int | None,
    n_repeats: int,
) -> np.ndarray:
    """Generate model seeds for repeated fitting."""
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")

    rng = make_rng(random_state)
    return draw_uint32_seeds(rng, n_repeats)


def permute_response(y: np.ndarray, seed: int) -> np.ndarray:
    """Return a reproducible random permutation of a response vector."""
    rng = make_rng(int(seed))
    return rng.permutation(np.asarray(y))


def permute_column(
    X: np.ndarray,
    column: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of X with one column randomly permuted."""
    X_arr = np.asarray(X)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_features = X_arr.shape[1]
    if column < 0 or column >= n_features:
        raise ValueError("column index out of range.")

    X_perm = X_arr.copy()
    X_perm[:, column] = rng.permutation(X_perm[:, column])

    return X_perm


__all__ = [
    "VALID_IMPORTANCE_KINDS",
    "validate_xy",
    "validate_eval_data",
    "get_feature_names",
    "validate_model_class",
    "get_model_spec",
    "make_model",
    "validate_importance_kind",
    "make_rng",
    "draw_uint32_seeds",
    "make_permutation_selector_seeds",
    "make_repeated_model_seeds",
    "permute_response",
    "permute_column",
    "check_positive_int",
    "check_probability",
]