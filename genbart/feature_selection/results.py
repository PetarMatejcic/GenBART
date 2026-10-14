"""Result containers for BART feature-selection routines."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ThresholdResult:
    """Selection result for one permutation-null thresholding rule.

    A ``ThresholdResult`` combines the observed variable-importance values with
    the thresholds produced by one selection rule. A feature is selected when its
    observed importance is strictly greater than its corresponding threshold.

    Attributes
    ----------
    method : str
        Name of the thresholding rule.

    feature_names : list of str
        Predictor names in the same order as the importance arrays.

    importance : np.ndarray
        Observed variable-importance values with shape ``(p,)``.

    thresholds : np.ndarray
        Selection thresholds with shape ``(p,)``.

        Some methods use feature-specific thresholds, while ``"global_max"``
        assigns the same threshold to every feature.

    selected : np.ndarray
        Boolean array with shape ``(p,)`` indicating which predictors were
        selected.

    quantile : float
        Null-distribution quantile associated with the configured ``alpha``.

    extra : dict
        Method-specific diagnostic information.

    Notes
    -----
    Selection always uses a strict comparison:

        importance > threshold

    A feature whose importance is exactly equal to its threshold is therefore not
    selected.
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
        """Return the selection threshold for one feature.

        Parameters
        ----------
        feature : str or int
            Feature name or zero-based column index.

        Returns
        -------
        float
            Threshold assigned to the requested feature.

        Raises
        ------
        ValueError
            If the feature name is unknown or the index is out of range.
        """
        j = self._feature_index(feature)
        return float(self.thresholds[j])

    def importance_for(self, feature: str | int) -> float:
        """Return the observed importance for one feature.

        Parameters
        ----------
        feature : str or int
            Feature name or zero-based column index.

        Returns
        -------
        float
            Observed variable-importance value.

        Raises
        ------
        ValueError
            If the feature name is unknown or the index is out of range.
        """
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
    """Permutation-null variable-selection result for one importance definition.

    ``VariableSelectionResult`` stores the observed importance estimates, their
    variation across repeated observed-response fits, the permutation-null
    importance distribution, and the results of every supported thresholding
    rule.

    One result object corresponds to one importance definition: either standard
    split-count importance or log marginal-likelihood-weighted importance.

    Attributes
    ----------
    feature_names : list of str
        Predictor names.

    importance : np.ndarray
        Mean observed importance across repeated BART fits, with shape ``(p,)``.

    importance_repeats : np.ndarray
        Importance values from the individual observed-response fits, with shape
        ``(n_repeats, p)``.

    importance_sd : np.ndarray
        Sample standard deviation of observed importance across repeated fits,
        with shape ``(p,)``.

    null_importance : np.ndarray
        Permutation-null importance matrix with shape
        ``(n_permutations, p)``.

        Each row corresponds to one permuted response. When ``n_repeats > 1``,
        that row is the mean importance across the repeated BART fits for that
        permutation.

    methods : dict[str, ThresholdResult]
        Threshold results for ``"local"``, ``"global_max"``, and ``"global_se"``.

    default_method : str
        Thresholding rule used when a method argument is omitted.

    importance_kind : {"raw", "logml"}
        Variable-importance definition represented by this result.
    """

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
        """Return selection thresholds for one thresholding method.

        Parameters
        ----------
        method : {"local", "global_max", "global_se"}, optional
            Thresholding rule. If omitted, ``default_method`` is used.

        Returns
        -------
        np.ndarray
            Copy of the feature thresholds with shape ``(p,)``.
        """
        return self._method_result(method).thresholds.copy()

    def ranking(self) -> np.ndarray:
        """Return feature indices ordered by decreasing observed importance.

        Returns
        -------
        np.ndarray
            Integer feature indices from highest to lowest mean observed importance.

        Notes
        -----
        The ranking depends only on ``importance`` and is therefore independent of the
        chosen thresholding method.
        """
        return np.argsort(-self.importance)

    def to_frame(self, method: str | None = None) -> list[dict]:
        """Return a row-wise feature-selection summary.

        Parameters
        ----------
        method : {"local", "global_max", "global_se"}, optional
            Thresholding rule used for the threshold and selected columns. If omitted,
            ``default_method`` is used.

        Returns
        -------
        list of dict
            One dictionary per feature, sorted by decreasing observed importance.

            Each row contains:

            - ``"rank"``
            - ``"feature"``
            - ``"importance"``
            - ``"importance_sd"``
            - ``"null_mean"``
            - ``"null_sd"``
            - ``"threshold"``
            - ``"selected"``
            - ``"method"``
            - ``"importance_kind"``

        Notes
        -----
        Despite the method name, this function currently returns a list of
        dictionaries rather than a pandas DataFrame.
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
        """Return a compact summary of the selection result.

        Parameters
        ----------
        method : {"local", "global_max", "global_se"}, optional
            Thresholding rule. If omitted, ``default_method`` is used.

        Returns
        -------
        dict
            Dictionary containing the selected method and importance definition,
            number of features, number and names of selected features, and the
            highest-ranked feature and its importance.
        """
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
        """Compare selections produced by all thresholding rules.

        Returns
        -------
        list of dict
            One row per thresholding method containing the method name, number of
            selected features, and selected feature names.

        Notes
        -----
        All methods are applied to the same observed and permutation-null importance
        values. Only the thresholding rule changes.
        """
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
    """Result returned by experimental predictive-degradation selection.

    This container stores degradation summaries and feature selections produced by
    :class:`BartPredictiveSelector`.

    Notes
    -----
    ``BartPredictiveSelector`` is experimental. Consequently, this result
    container and its attributes may change or be removed in a future release.
    """

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
            "n_permutations": int(self.n_permutations),
            "use_posterior_draws": bool(self.use_posterior_draws),
        }