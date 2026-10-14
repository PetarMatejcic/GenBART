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

    ``BartVariableSelector`` compares observed BART variable importance with the
    importance that would be expected after destroying the relationship between
    the predictors and response.

    For the observed response, ``n_repeats`` independent BART models are fitted
    and their variable-importance vectors are averaged.

    For each of ``n_permutations`` null datasets, the response is randomly
    permuted and ``n_repeats`` BART models are fitted. Their importance vectors
    are averaged to produce one permutation-null importance vector.

    Both standard split-count importance and log marginal-likelihood-weighted
    importance are computed during the same fitting procedure.

    Parameters
    ----------
    model_cls : callable
        BART estimator class or compatible factory.

        Fresh model instances are created repeatedly during selection. The model
        must support ``fit(X, y)`` and ``variable_inclusion(kind=...)`` and must
        accept ``random_state`` as a constructor argument.

    model_params : dict, optional
        Constructor arguments passed to each newly created BART model.

        ``random_state`` is controlled internally by the selector and overrides
        any value supplied here.

    n_permutations : int, default=20
        Number of independently permuted response datasets used to construct the
        null importance distribution.

    n_repeats : int, default=5
        Number of BART fits averaged for the observed response and separately for
        each permuted response.

    alpha : float, default=0.05
        Upper-tail probability used by the permutation-null thresholding rules.

        For example, ``alpha=0.05`` corresponds to using the upper 5% tail of the
        relevant null distribution.

    method : {"local", "global_max", "global_se"}, default="global_se"
        Thresholding rule used by the selector's default result.

    importance_kind : {"raw", "logml"}, default="raw"
        Variable-importance definition used by the default result and convenience
        methods.

        ``"raw"`` uses standard split-count variable inclusion.

        ``"logml"`` uses positive log marginal-likelihood-weighted variable
        inclusion.

        Both importance definitions are computed regardless of this setting and
        can be retrieved after fitting with :meth:`get_result`.

    random_state : int or None, optional
        Seed controlling response permutations and the seeds assigned to repeated
        BART fits.

    verbose : bool, default=False
        Whether to print progress while fitting repeated observed and null models.

    Notes
    -----
    The approximate number of BART fits is

    .. math::

        n_{\\mathrm{repeats}}
        \\left(1 + n_{\\mathrm{permutations}}\\right).

    Consequently, permutation-null selection can be substantially more expensive
    than fitting a single BART model.

    The resulting thresholds measure importance relative to a permuted-response
    null distribution. They should not be interpreted as posterior probabilities
    that individual predictors are relevant or as causal significance tests.
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
        """Create a selector from an existing model instance.

        The estimator class and constructor parameters are obtained from the model's
        :meth:`~genbart.base_bart.BaseBART.get_params` method. A fresh BART model is
        created for every repeated fit; fitted state from ``model`` is not reused.

        Parameters
        ----------
        model : object
            Model instance implementing ``get_params()``.

        n_permutations : int, default=20
            Number of permuted-response null datasets.

        n_repeats : int, default=5
            Number of repeated BART fits per observed or permuted response.

        alpha : float, default=0.05
            Upper-tail probability used by the thresholding rules.

        method : {"local", "global_max", "global_se"}, default="global_se"
            Default thresholding rule.

        importance_kind : {"raw", "logml"}, default="raw"
            Importance definition used by the default result.

        random_state : int or None, optional
            Seed controlling selector randomness.

        verbose : bool, default=False
            Whether to print fitting progress.

        Returns
        -------
        BartVariableSelector
            Configured selector using the supplied model's class and parameters.

        Raises
        ------
        TypeError
            If ``model`` is a class rather than an instance, does not implement
            ``get_params()``, or returns invalid parameter information.
        """
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
        """Fit the permutation-null variable-selection procedure.

        The observed-response importance is estimated by averaging ``n_repeats`` BART
        fits. A null distribution is then constructed from ``n_permutations`` shuffled
        responses, with each null importance vector also averaged over
        ``n_repeats`` BART fits.

        Both ``"raw"`` and ``"logml"`` importance definitions are collected from the
        same fitted models.

        Parameters
        ----------
        X : array-like
            Predictor matrix. A one-dimensional array is interpreted as a single
            predictor.

        y : array-like
            One-dimensional response array.

        feature_names : list of str, optional
            Names assigned to the predictors.

            If omitted and ``X`` provides column names, such as a pandas DataFrame,
            those names are used. Otherwise names ``"x0"``, ``"x1"``, ... are
            generated.

        Returns
        -------
        VariableSelectionResult
            Result corresponding to the selector's configured ``importance_kind``.

            Results for both importance definitions are retained internally and can be
            retrieved with :meth:`get_result`.

        Notes
        -----
        After fitting:

        ``importance_repeats_by_kind_``
            Contains observed-response importance vectors from the individual repeated
            fits.

        ``importance_by_kind_``
            Contains their mean importance vectors.

        ``null_importance_by_kind_``
            Contains one averaged null importance vector per response permutation.

        ``results_``
            Maps ``"raw"`` and ``"logml"`` to their respective
            :class:`VariableSelectionResult` objects.
        """
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

    def get_result(
        self,
        importance_kind: str | None = None,
    ) -> VariableSelectionResult:
        """Return the fitted result for one importance definition.

        Parameters
        ----------
        importance_kind : {"raw", "logml"}, optional
            Importance definition to retrieve. If omitted, the selector's configured
            ``importance_kind`` is used.

        Returns
        -------
        VariableSelectionResult
            Selection result containing observed importance, permutation-null
            importance, and all three thresholding methods for the requested
            importance definition.

        Raises
        ------
        RuntimeError
            If the selector has not been fitted.

        ValueError
            If ``importance_kind`` is not ``"raw"`` or ``"logml"``.

        Notes
        -----
        Both importance definitions are calculated during :meth:`fit`, so switching
        between ``"raw"`` and ``"logml"`` here does not refit the BART models.
        """
        self._check_is_fitted()

        if importance_kind is None:
            importance_kind = self.importance_kind

        validate_importance_kind(importance_kind)
        return self.results_[importance_kind]

    def _fit_observed_response(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seeds: np.ndarray,
    ) -> None:
        """Fit repeated models on the observed response.

        Every fitted BART model is used to extract both raw and logML variable
        inclusion values.
        """
        n_features = X.shape[1]

        importance_repeats_by_kind = {
            kind: np.empty((self.n_repeats, n_features), dtype=float)
            for kind in VALID_IMPORTANCE_KINDS
        }

        for repeat_id, seed in enumerate(seeds):
            if self.verbose:
                print(
                    f"Fitting observed-response BART "
                    f"{repeat_id + 1}/{self.n_repeats}"
                )

            model = make_model(
                self.model_cls,
                self.model_params,
                random_state=seed,
            )
            model.fit(X, y)

            importances = self._extract_importances(
                model,
                n_features=n_features,
            )

            for kind in VALID_IMPORTANCE_KINDS:
                importance_repeats_by_kind[kind][repeat_id, :] = importances[kind]

        self.importance_repeats_by_kind_ = importance_repeats_by_kind

        self.importance_by_kind_ = {
            kind: values.mean(axis=0)
            for kind, values in importance_repeats_by_kind.items()
        }

        self.importance_sd_by_kind_ = {}

        for kind, values in importance_repeats_by_kind.items():
            if self.n_repeats > 1:
                self.importance_sd_by_kind_[kind] = values.std(axis=0, ddof=1)
            else:
                self.importance_sd_by_kind_[kind] = np.zeros(
                    n_features,
                    dtype=float,
                )

    def _fit_null_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_seeds: np.ndarray,
        shuffle_seeds: np.ndarray,
    ) -> None:
        """Fit repeated models on permuted responses.

        Every fitted permutation-null BART model is used to extract both raw and
        logML variable inclusion values.
        """
        n_features = X.shape[1]

        null_importance_by_kind = {
            kind: np.empty(
                (self.n_permutations, n_features),
                dtype=float,
            )
            for kind in VALID_IMPORTANCE_KINDS
        }

        for permutation_id, shuffle_seed in enumerate(shuffle_seeds):
            if self.verbose:
                print(
                    f"Fitting permuted-response BART "
                    f"{permutation_id + 1}/{self.n_permutations}"
                )

            y_perm = permute_response(y, int(shuffle_seed))

            permutation_importance_by_kind = {
                kind: np.empty(
                    (self.n_repeats, n_features),
                    dtype=float,
                )
                for kind in VALID_IMPORTANCE_KINDS
            }

            for repeat_id, model_seed in enumerate(
                model_seeds[permutation_id]
            ):
                model = make_model(
                    self.model_cls,
                    self.model_params,
                    random_state=model_seed,
                )
                model.fit(X, y_perm)

                importances = self._extract_importances(
                    model,
                    n_features=n_features,
                )

                for kind in VALID_IMPORTANCE_KINDS:
                    permutation_importance_by_kind[kind][
                        repeat_id, :
                    ] = importances[kind]

            for kind in VALID_IMPORTANCE_KINDS:
                null_importance_by_kind[kind][
                    permutation_id, :
                ] = permutation_importance_by_kind[kind].mean(axis=0)

        self.null_importance_by_kind_ = null_importance_by_kind

    def _build_result(self) -> VariableSelectionResult:
        """Build fitted results for both raw and logML importance."""
        results = {}

        for kind in VALID_IMPORTANCE_KINDS:
            methods = build_threshold_results(
                feature_names=self.feature_names_,
                importance=self.importance_by_kind_[kind],
                null_importance=self.null_importance_by_kind_[kind],
                alpha=self.alpha,
            )

            results[kind] = VariableSelectionResult(
                feature_names=self.feature_names_,
                importance=self.importance_by_kind_[kind],
                importance_repeats=self.importance_repeats_by_kind_[kind],
                importance_sd=self.importance_sd_by_kind_[kind],
                null_importance=self.null_importance_by_kind_[kind],
                methods=methods,
                default_method=self.method,
                importance_kind=kind,
            )

        self.results_ = results

        default_result = results[self.importance_kind]

        # Backward-compatible selector attributes. These continue to refer to the
        # importance kind configured when the selector was created.
        self.importance_ = default_result.importance
        self.importance_repeats_ = default_result.importance_repeats
        self.importance_sd_ = default_result.importance_sd
        self.null_importance_ = default_result.null_importance

        return default_result

    def _extract_importances(
        self,
        model: Any,
        *,
        n_features: int,
    ) -> dict[str, np.ndarray]:
        """Extract and validate both variable-inclusion importance definitions."""
        if not hasattr(model, "variable_inclusion"):
            raise TypeError("model must implement variable_inclusion().")

        importances = {}

        for kind in VALID_IMPORTANCE_KINDS:
            importance = np.asarray(
                model.variable_inclusion(kind=kind),
                dtype=float,
            )

            if importance.shape != (n_features,):
                raise RuntimeError(
                    f"variable_inclusion(kind={kind!r}) returned shape "
                    f"{importance.shape}, expected {(n_features,)}."
                )

            importances[kind] = importance

        return importances

    def _check_is_fitted(self) -> None:
        """Raise if the selector has not been fitted."""
        if not hasattr(self, "result_"):
            raise RuntimeError("BartVariableSelector is not fitted.")


__all__ = [
    "BartVariableSelector",
]