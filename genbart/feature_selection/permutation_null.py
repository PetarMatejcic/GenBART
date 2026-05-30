"""Permutation-null BART variable selection."""

from __future__ import annotations
from typing import Any
import numpy as np

from .results import VariableSelectionResult
from .thresholds import VALID_THRESHOLD_METHODS, build_threshold_results
from .utils import (
    VALID_IMPORTANCE_KINDS,
    get_feature_names,
    get_model_spec,
    make_model,
    make_permutation_selector_seeds,
    permute_response,
    validate_importance_kind,
    validate_model_class,
    validate_xy,
)


class BartVariableSelector:
    """Permutation-null variable selector for BART models.

    The selector fits repeated BART models on the observed response, extracts
    variable-inclusion importance values, then fits repeated BART models on
    permuted responses to estimate a null importance distribution.

    Supported thresholding rules are:

    - ``"local"``
    - ``"global_max"``
    - ``"global_se"``
    """

    def __init__(
        self,
        model_cls: Any,
        model_params: dict | None = None,
        *,
        n_permutations: int = 20,
        n_repeats: int = 5,
        alpha: float = 0.05,
        method: str = "global_se",
        importance_kind: str = "raw",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        validate_model_class(model_cls)

        if n_permutations <= 0:
            raise ValueError("n_permutations must be positive.")

        if n_repeats <= 0:
            raise ValueError("n_repeats must be positive.")

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        if method not in VALID_THRESHOLD_METHODS:
            allowed = ", ".join(repr(name) for name in VALID_THRESHOLD_METHODS)
            raise ValueError(f"method must be one of {allowed}.")

        validate_importance_kind(importance_kind)

        self.model_cls = model_cls
        self.model_params = dict(model_params or {})

        self.n_permutations = int(n_permutations)
        self.n_repeats = int(n_repeats)
        self.alpha = float(alpha)
        self.method = method
        self.importance_kind = importance_kind
        self.random_state = random_state
        self.verbose = bool(verbose)

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        n_permutations: int = 20,
        n_repeats: int = 5,
        alpha: float = 0.05,
        method: str = "global_se",
        importance_kind: str = "raw",
        random_state: int | None = None,
        verbose: bool = False,
    ) -> BartVariableSelector:
        """Create a selector from an unfitted model instance."""
        model_cls, model_params = get_model_spec(model)

        return cls(
            model_cls=model_cls,
            model_params=model_params,
            n_permutations=n_permutations,
            n_repeats=n_repeats,
            alpha=alpha,
            method=method,
            importance_kind=importance_kind,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: list[str] | None = None,
    ) -> VariableSelectionResult:
        """Fit the permutation-null variable-selection procedure."""
        data_X = X
        X, y = validate_xy(X, y)
        _, n_features = X.shape

        self.feature_names_ = get_feature_names(data_X,
                                                feature_names,
                                                n_features)

        observed_model_seeds, null_model_seeds, shuffle_seeds = make_permutation_selector_seeds(
            random_state=self.random_state,
            n_repeats=self.n_repeats,
            n_permutations=self.n_permutations)

        self._fit_observed_response(X, y, observed_model_seeds)
        self._fit_null_distribution(X, y, null_model_seeds, shuffle_seeds)

        self.result_ = self._build_result()
        return self.result_

    def selected_features(self, method: str | None = None) -> list[str]:
        """Return selected feature names from the fitted result."""
        self._check_is_fitted()
        return self.result_.selected_features(method)

    def selected_indices(self, method: str | None = None) -> np.ndarray:
        """Return selected feature indices from the fitted result."""
        self._check_is_fitted()
        return self.result_.selected_indices(method)

    def selected_mask(self, method: str | None = None) -> np.ndarray:
        """Return selected-feature mask from the fitted result."""
        self._check_is_fitted()
        return self.result_.selected_mask(method)

    def ranking(self) -> np.ndarray:
        """Return feature indices sorted by decreasing observed importance."""
        self._check_is_fitted()
        return self.result_.ranking()

    def to_frame(self, method: str | None = None) -> list[dict]:
        """Return a row-wise fitted-result summary."""
        self._check_is_fitted()
        return self.result_.to_frame(method)

    def summary(self, method: str | None = None) -> dict:
        """Return a compact fitted-result summary."""
        self._check_is_fitted()
        return self.result_.summary(method)

    def compare_methods(self) -> list[dict]:
        """Compare all fitted thresholding methods."""
        self._check_is_fitted()
        return self.result_.compare_methods()

    def _fit_observed_response(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seeds: np.ndarray,
    ) -> None:
        """Fit repeated models on the observed response."""
        n_features = X.shape[1]
        importance_repeats = np.empty((self.n_repeats, n_features),
                                      dtype=float)

        for repeat_id, seed in enumerate(seeds):
            if self.verbose:
                print(f"Fitting observed-response BART "
                      f"{repeat_id + 1}/{self.n_repeats}")

            model = make_model(self.model_cls,
                               self.model_params,
                               random_state=seed)
            model.fit(X, y)

            importance_repeats[repeat_id, :] = self._extract_importance(model,
                                                                        n_features=n_features)

        self.importance_repeats_ = importance_repeats
        self.importance_ = importance_repeats.mean(axis=0)

        if self.n_repeats > 1:
            self.importance_sd_ = importance_repeats.std(axis=0, ddof=1)
        else:
            self.importance_sd_ = np.zeros(n_features, dtype=float)

    def _fit_null_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_seeds: np.ndarray,
        shuffle_seeds: np.ndarray,
    ) -> None:
        """Fit repeated models on permuted responses."""
        n_features = X.shape[1]
        null_importance = np.empty(
            (self.n_permutations, n_features),
            dtype=float,
        )

        for permutation_id, shuffle_seed in enumerate(shuffle_seeds):
            if self.verbose:
                print(f"Fitting permuted-response BART "
                      f"{permutation_id + 1}/{self.n_permutations}")

            y_perm = permute_response(y, int(shuffle_seed))
            permutation_importance = np.empty((self.n_repeats, n_features),
                                              dtype=float)

            for repeat_id, model_seed in enumerate(model_seeds[permutation_id]):
                model = make_model(self.model_cls,
                                   self.model_params,
                                   random_state=model_seed)
                model.fit(X, y_perm)

                permutation_importance[repeat_id, :] = self._extract_importance(model,
                                                                                n_features=n_features)

            null_importance[permutation_id, :] = permutation_importance.mean(axis=0)

        self.null_importance_ = null_importance

    def _build_result(self) -> VariableSelectionResult:
        """Build the fitted result object."""
        methods = build_threshold_results(feature_names=self.feature_names_,
                                          importance=self.importance_,
                                          null_importance=self.null_importance_,
                                          alpha=self.alpha)

        return VariableSelectionResult(
            feature_names=self.feature_names_,
            importance=self.importance_,
            importance_repeats=self.importance_repeats_,
            importance_sd=self.importance_sd_,
            null_importance=self.null_importance_,
            methods=methods,
            default_method=self.method,
            importance_kind=self.importance_kind,
        )

    def _extract_importance(
        self,
        model: Any,
        *,
        n_features: int,
    ) -> np.ndarray:
        """Extract and validate variable-inclusion importance from a fitted model."""
        if not hasattr(model, "variable_inclusion"):
            raise TypeError("model must implement variable_inclusion().")

        importance = np.asarray(model.variable_inclusion(kind=self.importance_kind),
                                dtype=float)

        if importance.shape != (n_features,):
            raise RuntimeError(
                f"variable_inclusion returned shape {importance.shape}, "
                f"expected {(n_features,)}.")

        return importance

    def _check_is_fitted(self) -> None:
        """Raise if the selector has not been fitted."""
        if not hasattr(self, "result_"):
            raise RuntimeError("BartVariableSelector is not fitted.")


__all__ = [
    "BartVariableSelector",
]