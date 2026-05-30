"""Thresholding rules for BART feature-selection results."""

from __future__ import annotations
import numpy as np
from .results import ThresholdResult


VALID_THRESHOLD_METHODS = ("local", "global_max", "global_se")


def local_threshold(
    *,
    feature_names: list[str],
    importance: np.ndarray,
    null_importance: np.ndarray,
    alpha: float,
) -> ThresholdResult:
    """Compare each feature against its own permutation-null quantile."""
    importance, null_importance = _validate_threshold_inputs(
        importance=importance,
        null_importance=null_importance,
        feature_names=feature_names,
        alpha=alpha,
    )

    quantile = 1.0 - alpha
    thresholds = np.quantile(null_importance, quantile, axis=0)
    selected = importance > thresholds

    return ThresholdResult(
        method="local",
        feature_names=feature_names,
        importance=importance,
        thresholds=thresholds,
        selected=selected,
        quantile=quantile,
        extra={},
    )


def global_max_threshold(
    *,
    feature_names: list[str],
    importance: np.ndarray,
    null_importance: np.ndarray,
    alpha: float,
) -> ThresholdResult:
    """Compare each feature against a global max-null threshold."""
    importance, null_importance = _validate_threshold_inputs(
        importance=importance,
        null_importance=null_importance,
        feature_names=feature_names,
        alpha=alpha,
    )

    quantile = 1.0 - alpha
    max_null = np.max(null_importance, axis=1)
    threshold = float(np.quantile(max_null, quantile))
    thresholds = np.full_like(importance, threshold, dtype=float)
    selected = importance > thresholds

    return ThresholdResult(
        method="global_max",
        feature_names=feature_names,
        importance=importance,
        thresholds=thresholds,
        selected=selected,
        quantile=quantile,
        extra={
            "global_threshold": threshold,
            "max_null": max_null,
        },
    )


def global_se_threshold(
    *,
    feature_names: list[str],
    importance: np.ndarray,
    null_importance: np.ndarray,
    alpha: float,
) -> ThresholdResult:
    """Compare each feature against a globally standardized null threshold."""
    importance, null_importance = _validate_threshold_inputs(
        importance=importance,
        null_importance=null_importance,
        feature_names=feature_names,
        alpha=alpha,
    )

    quantile = 1.0 - alpha

    null_mean = null_importance.mean(axis=0)
    if null_importance.shape[0] > 1:
        null_sd = null_importance.std(axis=0, ddof=1)
    else:
        null_sd = np.zeros(null_importance.shape[1], dtype=float)

    safe_sd = null_sd.copy()
    safe_sd[safe_sd == 0.0] = 1.0

    standardized = (null_importance - null_mean) / safe_sd
    max_standardized = np.max(standardized, axis=1)

    global_se = float(np.quantile(max_standardized, quantile))
    thresholds = null_mean + global_se * null_sd
    selected = importance > thresholds

    return ThresholdResult(
        method="global_se",
        feature_names=feature_names,
        importance=importance,
        thresholds=thresholds,
        selected=selected,
        quantile=quantile,
        extra={
            "global_se": global_se,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "max_standardized": max_standardized,
        },
    )


def build_threshold_results(
    *,
    feature_names: list[str],
    importance: np.ndarray,
    null_importance: np.ndarray,
    alpha: float,
) -> dict[str, ThresholdResult]:
    """Build all supported thresholding results."""
    return {
        "local": local_threshold(
            feature_names=feature_names,
            importance=importance,
            null_importance=null_importance,
            alpha=alpha,
        ),
        "global_max": global_max_threshold(
            feature_names=feature_names,
            importance=importance,
            null_importance=null_importance,
            alpha=alpha,
        ),
        "global_se": global_se_threshold(
            feature_names=feature_names,
            importance=importance,
            null_importance=null_importance,
            alpha=alpha,
        ),
    }


def _validate_threshold_inputs(
    *,
    importance: np.ndarray,
    null_importance: np.ndarray,
    feature_names: list[str],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shared threshold inputs."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    importance = np.asarray(importance, dtype=float)
    null_importance = np.asarray(null_importance, dtype=float)

    if importance.ndim != 1:
        raise ValueError("importance must be a 1D array.")

    if null_importance.ndim != 2:
        raise ValueError("null_importance must be a 2D array.")

    if null_importance.shape[1] != importance.shape[0]:
        raise ValueError(
            "null_importance must have one column per feature in importance."
        )

    if len(feature_names) != importance.shape[0]:
        raise ValueError("feature_names must have length equal to importance.")

    if null_importance.shape[0] == 0:
        raise ValueError("null_importance must contain at least one null draw.")

    return importance, null_importance


__all__ = [
    "VALID_THRESHOLD_METHODS",
    "local_threshold",
    "global_max_threshold",
    "global_se_threshold",
    "build_threshold_results",
]