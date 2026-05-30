"""Result containers for BART feature-selection routines."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ThresholdResult:
    """Selection output for one thresholding rule.

    A ThresholdResult stores the observed feature-importance values, the
    threshold values used by one selection method, and the final selected mask.
    """

    method: str
    feature_names: list[str]
    importance: np.ndarray
    thresholds: np.ndarray
    selected: np.ndarray
    quantile: float
    extra: dict = field(default_factory=dict)

    def selected_features(self) -> list[str]:
        """Return the names of selected features."""
        return [
            self.feature_names[j]
            for j in range(len(self.feature_names))
            if self.selected[j]
        ]

    def selected_indices(self) -> np.ndarray:
        """Return selected feature indices."""
        return np.flatnonzero(self.selected)

    def selected_mask(self) -> np.ndarray:
        """Return a copy of the selected-feature mask."""
        return self.selected.copy()

    def n_selected(self) -> int:
        """Return the number of selected features."""
        return int(np.sum(self.selected))

    def threshold_for(self, feature: str | int) -> float:
        """Return the threshold assigned to one feature."""
        j = self._feature_index(feature)
        return float(self.thresholds[j])

    def importance_for(self, feature: str | int) -> float:
        """Return the observed importance value for one feature."""
        j = self._feature_index(feature)
        return float(self.importance[j])

    def _feature_index(self, feature: str | int) -> int:
        """Resolve a feature name or integer index."""
        if isinstance(feature, str):
            if feature not in self.feature_names:
                raise ValueError(f"Unknown feature: {feature}")
            return self.feature_names.index(feature)

        j = int(feature)
        if j < 0 or j >= len(self.feature_names):
            raise ValueError("feature index out of range.")
        return j


@dataclass
class VariableSelectionResult:
    """Result returned by permutation-null BART variable selection."""

    feature_names: list[str]
    importance: np.ndarray
    importance_repeats: np.ndarray
    importance_sd: np.ndarray
    null_importance: np.ndarray
    methods: dict[str, ThresholdResult]
    default_method: str = "global_se"
    importance_kind: str = "raw"

    @property
    def null_mean(self) -> np.ndarray:
        """Return variable-wise mean null importance."""
        return self.null_importance.mean(axis=0)

    @property
    def null_sd(self) -> np.ndarray:
        """Return variable-wise standard deviation of null importance."""
        if self.null_importance.shape[0] > 1:
            return self.null_importance.std(axis=0, ddof=1)
        return np.zeros(self.null_importance.shape[1])

    def selected_features(self, method: str | None = None) -> list[str]:
        """Return selected feature names for one thresholding method."""
        return self._method_result(method).selected_features()

    def selected_indices(self, method: str | None = None) -> np.ndarray:
        """Return selected feature indices for one thresholding method."""
        return self._method_result(method).selected_indices()

    def selected_mask(self, method: str | None = None) -> np.ndarray:
        """Return selected-feature mask for one thresholding method."""
        return self._method_result(method).selected_mask()

    def thresholds(self, method: str | None = None) -> np.ndarray:
        """Return thresholds for one thresholding method."""
        return self._method_result(method).thresholds.copy()

    def ranking(self) -> np.ndarray:
        """Return feature indices sorted by decreasing observed importance."""
        return np.argsort(-self.importance)

    def to_frame(self, method: str | None = None) -> list[dict]:
        """Return a row-wise summary sorted by decreasing importance.

        This returns a list of dictionaries.
        """
        method_result = self._method_result(method)
        order = self.ranking()

        rows = []
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "rank": rank,
                    "feature": self.feature_names[j],
                    "importance": float(self.importance[j]),
                    "importance_sd": float(self.importance_sd[j]),
                    "null_mean": float(self.null_mean[j]),
                    "null_sd": float(self.null_sd[j]),
                    "threshold": float(method_result.thresholds[j]),
                    "selected": bool(method_result.selected[j]),
                    "method": method_result.method,
                    "importance_kind": self.importance_kind,
                }
            )

        return rows

    def summary(self, method: str | None = None) -> dict:
        """Return a compact result summary."""
        method_result = self._method_result(method)
        top_idx = int(np.argmax(self.importance))

        return {
            "method": method_result.method,
            "importance_kind": self.importance_kind,
            "n_features": len(self.feature_names),
            "n_selected": method_result.n_selected(),
            "selected_features": method_result.selected_features(),
            "top_feature": self.feature_names[top_idx],
            "top_importance": float(self.importance[top_idx]),
        }

    def compare_methods(self) -> list[dict]:
        """Return selected-feature summaries for all thresholding methods."""
        rows = []

        for method, method_result in self.methods.items():
            rows.append(
                {
                    "method": method,
                    "n_selected": method_result.n_selected(),
                    "selected_features": method_result.selected_features(),
                }
            )

        return rows

    def _method_result(self, method: str | None = None) -> ThresholdResult:
        """Return the ThresholdResult for a requested method."""
        method = self.default_method if method is None else method

        if method not in self.methods:
            allowed = ", ".join(sorted(self.methods))
            raise ValueError(f"Unknown method: {method}. Available methods: {allowed}")

        return self.methods[method]


@dataclass
class PredictiveSelectionResult:
    """Result returned by predictive-degradation BART feature selection."""

    feature_names: list[str]
    importance_mean: np.ndarray
    importance_sd: np.ndarray
    prob_positive: np.ndarray
    degradation_samples: np.ndarray
    selected: np.ndarray
    loss_metric: str
    task: str
    selection_probability: float
    min_mean_degradation: float
    baseline_loss_mean: float
    n_repeats: int
    n_permutations: int
    use_posterior_draws: bool

    def selected_features(self) -> list[str]:
        """Return the names of selected features."""
        return [
            self.feature_names[j]
            for j in range(len(self.feature_names))
            if self.selected[j]
        ]

    def selected_indices(self) -> np.ndarray:
        """Return selected feature indices."""
        return np.flatnonzero(self.selected)

    def selected_mask(self) -> np.ndarray:
        """Return a copy of the selected-feature mask."""
        return self.selected.copy()

    def n_selected(self) -> int:
        """Return the number of selected features."""
        return int(np.sum(self.selected))

    def ranking(self) -> np.ndarray:
        """Return feature indices sorted by decreasing predictive degradation."""
        return np.argsort(-self.importance_mean)

    def to_frame(self) -> list[dict]:
        """Return a row-wise summary sorted by decreasing predictive degradation."""
        order = self.ranking()

        rows = []
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "rank": rank,
                    "feature": self.feature_names[j],
                    "importance_mean": float(self.importance_mean[j]),
                    "importance_sd": float(self.importance_sd[j]),
                    "prob_positive": float(self.prob_positive[j]),
                    "selected": bool(self.selected[j]),
                    "loss_metric": self.loss_metric,
                    "task": self.task,
                }
            )

        return rows

    def summary(self) -> dict:
        """Return a compact result summary."""
        top_idx = int(np.argmax(self.importance_mean))

        return {
            "task": self.task,
            "loss_metric": self.loss_metric,
            "n_features": len(self.feature_names),
            "n_selected": self.n_selected(),
            "selected_features": self.selected_features(),
            "top_feature": self.feature_names[top_idx],
            "top_importance": float(self.importance_mean[top_idx]),
            "selection_probability": float(self.selection_probability),
            "min_mean_degradation": float(self.min_mean_degradation),
            "baseline_loss_mean": float(self.baseline_loss_mean),
            "n_repeats": int(self.n_repeats),
            "n_permutation_repeats": int(self.n_permutation_repeats),
            "use_posterior_draws": bool(self.use_posterior_draws),
        }