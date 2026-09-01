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
    """Compare each feature against the Bleich et al. global-SE threshold.

    The global-SE criterion chooses the smallest common nonnegative multiplier
    C* such that, for every feature, more than 1 - alpha of its permutation-null
    importance values lie below

        null_mean_j + C* * null_sd_j.

    Features are selected when their observed importance is strictly greater
    than the resulting feature-specific threshold.
    """
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

    n_null = null_importance.shape[0]

    # We need strictly more than (1 - alpha) of the null values to be
    # covered. If k is the required number of covered values, then
    #
    #     k = floor(n_null * (1 - alpha)) + 1.
    #
    # The corresponding zero-based order-statistic index is therefore
    # floor(n_null * (1 - alpha)).
    required_index = int(np.floor(n_null * quantile))

    positive_sd = null_sd > 0.0

    if np.any(positive_sd):
        standardized = (
            null_importance[:, positive_sd] - null_mean[positive_sd]
        ) / null_sd[positive_sd]

        required_by_feature = np.partition(
            standardized,
            required_index,
            axis=0,
        )[required_index, :]

        global_se = float(
            max(0.0, np.max(required_by_feature))
        )
    else:
        # Constant null distributions already have complete coverage at C = 0.
        global_se = 0.0

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