"""Predictive-degradation BART feature selection."""

from __future__ import annotations
from typing import Any
import numpy as np
from .results import PredictiveSelectionResult
from .utils import (
    draw_uint32_seeds,
    get_feature_names,
    get_model_spec,
    make_model,
    make_rng,
    permute_column,
    validate_eval_data,
    validate_model_class,
    validate_xy,
)


VALID_TASKS = ("classification", "regression")

VALID_LOSSES = {
    "classification": ("brier", "log_loss"),
    "regression": ("mse", "rmse"),
}


class BartPredictiveSelector:
    """Predictive-degradation feature selector for BART models.

    The selector fits repeated BART models, computes a baseline predictive loss,
    then permutes each feature column and measures how much the predictive loss
    increases.

    A feature is selected when:

    - predictive degradation is positive in at least ``selection_probability`` of
      degradation samples; and
    - mean degradation is greater than ``min_mean_degradation``.
    """

    def __init__(
        self,
        model_cls: Any,
        model_params: dict | None = None,
        *,
        task: str = "classification",
        loss_metric: str = "brier",
        n_repeats: int = 3,
        n_permutations: int = 5,
        selection_probability: float = 0.95,
        min_mean_degradation: float = 0.0,
        use_posterior_draws: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        validate_model_class(model_cls)

        if task not in VALID_TASKS:
            allowed = ", ".join(repr(name) for name in VALID_TASKS)
            raise ValueError(f"task must be one of {allowed}.")

        if loss_metric not in VALID_LOSSES[task]:
            allowed = ", ".join(repr(name) for name in VALID_LOSSES[task])
            raise ValueError(
                f"loss_metric must be one of {allowed} for task={task!r}."
            )

        if n_repeats <= 0:
            raise ValueError("n_repeats must be positive.")

        if n_permutations <= 0:
            raise ValueError("n_permutation_repeats must be positive.")

        if not 0.0 < selection_probability < 1.0:
            raise ValueError("selection_probability must be between 0 and 1.")

        if min_mean_degradation < 0.0:
            raise ValueError("min_mean_degradation must be nonnegative.")

        self.model_cls = model_cls
        self.model_params = dict(model_params or {})

        self.task = task
        self.loss_metric = loss_metric
        self.n_repeats = int(n_repeats)
        self.n_permutation_repeats = int(n_permutations)
        self.selection_probability = float(selection_probability)
        self.min_mean_degradation = float(min_mean_degradation)
        self.use_posterior_draws = bool(use_posterior_draws)
        self.random_state = random_state
        self.verbose = bool(verbose)

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        task: str = "classification",
        loss_metric: str = "brier",
        n_repeats: int = 3,
        n_permutations: int = 5,
        selection_probability: float = 0.95,
        min_mean_degradation: float = 0.0,
        use_posterior_draws: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> BartPredictiveSelector:
        """Create a selector from an unfitted model instance."""
        model_cls, model_params = get_model_spec(model)

        return cls(
            model_cls=model_cls,
            model_params=model_params,
            task=task,
            loss_metric=loss_metric,
            n_repeats=n_repeats,
            n_permutation_repeats=n_permutations,
            selection_probability=selection_probability,
            min_mean_degradation=min_mean_degradation,
            use_posterior_draws=use_posterior_draws,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        X_eval: Any | None = None,
        y_eval: Any | None = None,
        feature_names: list[str] | None = None,
    ) -> PredictiveSelectionResult:
        """Fit the predictive-degradation selection procedure.

        Parameters
        ----------
        X, y
            Training data used to fit each repeated BART model.
        X_eval, y_eval
            Optional evaluation data used to compute baseline and permuted-feature
            losses. If omitted, the training data are reused.
        feature_names
            Optional feature names. If omitted and X has columns, those column names
            are used.
        """
        data_X = X
        X, y = validate_xy(X, y)
        _, n_features = X.shape

        if X_eval is None and y_eval is None:
            X_eval = X
            y_eval = y
        elif X_eval is None or y_eval is None:
            raise ValueError("X_eval and y_eval must either both be provided or both be omitted.")
        else:
            X_eval, y_eval = validate_eval_data(X_eval,
                                                y_eval,
                                                n_features=n_features)

        self.feature_names_ = get_feature_names(data_X,
                                                feature_names,
                                                n_features)

        rng = make_rng(self.random_state)
        model_seeds = draw_uint32_seeds(rng, self.n_repeats)

        degradation_by_feature = [[] for _ in range(n_features)]
        baseline_losses_all = []

        for repeat_id, seed in enumerate(model_seeds):
            if self.verbose:
                print(f"Fitting BART repeat {repeat_id + 1}/{self.n_repeats}")

            model = make_model(self.model_cls,
                               self.model_params,
                               random_state=int(seed))
            model.fit(X, y)

            baseline_draws = self._prediction_draws(model, X_eval)
            baseline_losses = self._loss_per_draw(y_eval, baseline_draws)
            baseline_losses_all.append(baseline_losses)

            for permutation_id in range(self.n_permutation_repeats):
                if self.verbose:
                    print(f"  Permutation repeat "
                          f"{permutation_id + 1}/{self.n_permutation_repeats}")

                for feature_id in range(n_features):
                    X_perm = permute_column(X_eval, feature_id, rng)

                    perturbed_draws = self._prediction_draws(model, X_perm)
                    perturbed_losses = self._loss_per_draw(y_eval, perturbed_draws)

                    if perturbed_losses.shape != baseline_losses.shape:
                        raise RuntimeError(
                            "Perturbed losses and baseline losses have different shapes."
                        )

                    degradation_by_feature[feature_id].append(
                        perturbed_losses - baseline_losses
                    )

        degradation_samples = np.column_stack(
            [
                np.concatenate(degradation_by_feature[feature_id])
                for feature_id in range(n_features)
            ]
        )

        importance_mean = degradation_samples.mean(axis=0)

        if degradation_samples.shape[0] > 1:
            importance_sd = degradation_samples.std(axis=0, ddof=1)
        else:
            importance_sd = np.zeros(n_features, dtype=float)

        prob_positive = (degradation_samples > 0.0).mean(axis=0)

        selected = ((prob_positive >= self.selection_probability)
                    & (importance_mean > self.min_mean_degradation))

        baseline_loss_mean = float(np.mean(np.concatenate(baseline_losses_all)))

        self.degradation_samples_ = degradation_samples
        self.importance_mean_ = importance_mean
        self.importance_sd_ = importance_sd
        self.prob_positive_ = prob_positive
        self.selected_ = selected
        self.baseline_loss_mean_ = baseline_loss_mean

        self.result_ = PredictiveSelectionResult(
            feature_names=self.feature_names_,
            importance_mean=importance_mean,
            importance_sd=importance_sd,
            prob_positive=prob_positive,
            degradation_samples=degradation_samples,
            selected=selected,
            loss_metric=self.loss_metric,
            task=self.task,
            selection_probability=self.selection_probability,
            min_mean_degradation=self.min_mean_degradation,
            baseline_loss_mean=baseline_loss_mean,
            n_repeats=self.n_repeats,
            n_permutations=self.n_permutation_repeats,
            use_posterior_draws=self.use_posterior_draws,
        )

        return self.result_

    def selected_features(self) -> list[str]:
        """Return selected feature names from the fitted result."""
        self._check_is_fitted()
        return self.result_.selected_features()

    def selected_indices(self) -> np.ndarray:
        """Return selected feature indices from the fitted result."""
        self._check_is_fitted()
        return self.result_.selected_indices()

    def selected_mask(self) -> np.ndarray:
        """Return selected-feature mask from the fitted result."""
        self._check_is_fitted()
        return self.result_.selected_mask()

    def ranking(self) -> np.ndarray:
        """Return feature indices sorted by decreasing predictive degradation."""
        self._check_is_fitted()
        return self.result_.ranking()

    def to_frame(self) -> list[dict]:
        """Return a row-wise fitted-result summary."""
        self._check_is_fitted()
        return self.result_.to_frame()

    def summary(self) -> dict:
        """Return a compact fitted-result summary."""
        self._check_is_fitted()
        return self.result_.summary()

    def _prediction_draws(self, model, X):
        if self.use_posterior_draws:
            if not hasattr(model, "posterior_sample_draws"):
                raise TypeError(
                    "use_posterior_draws=True requires a model with posterior_draws(X)."
                )
            return self._as_2d_draws(model.posterior_sample_draws(X))

        if self.task == "classification":
            probs = model.predict_probs(X)["probs"]
            return self._as_point_prediction_draws(probs)

        pred = model.predict(X, conf_int=False)["prediction"]
        return self._as_point_prediction_draws(pred)

    def _loss_per_draw(self, y: np.ndarray, pred_draws: np.ndarray) -> np.ndarray:
        """Compute predictive loss separately for each posterior draw."""
        y = np.asarray(y)
        pred_draws = self._as_2d_draws(pred_draws)

        if pred_draws.shape[1] != y.shape[0]:
            raise ValueError("Prediction draws and y have incompatible lengths.")

        if self.task == "classification":
            y_binary = y.astype(int)

            if not np.all((y_binary == 0) | (y_binary == 1)):
                raise ValueError("classification y must contain only 0/1 labels.")

            prediction = np.clip(pred_draws, 1e-12, 1.0 - 1e-12)
            yy = y_binary.reshape(1, -1)

            if self.loss_metric == "brier":
                return np.mean((prediction - yy) ** 2, axis=1)

            if self.loss_metric == "log_loss":
                return -np.mean(
                    yy * np.log(prediction) + (1 - yy) * np.log(1 - prediction),
                    axis=1,
                )

        if self.task == "regression":
            yy = y.astype(float).reshape(1, -1)
            errors2 = (pred_draws - yy) ** 2

            if self.loss_metric == "mse":
                return np.mean(errors2, axis=1)

            if self.loss_metric == "rmse":
                return np.sqrt(np.mean(errors2, axis=1))

        raise RuntimeError(
            f"Unsupported task/loss combination: task={self.task!r}, "
            f"loss_metric={self.loss_metric!r}."
        )

    def _as_2d_draws(self, draws: Any) -> np.ndarray:
        """Coerce posterior draws to shape (n_draws, n_observations)."""
        arr = np.asarray(draws, dtype=float)

        if arr.ndim == 0:
            return arr.reshape(1, 1)

        if arr.ndim == 1:
            return arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError("prediction draws must be a 1D or 2D array.")

        return arr

    def _as_point_prediction_draws(self, pred: Any) -> np.ndarray:
        """Coerce point predictions to a single-draw matrix."""
        arr = np.asarray(pred, dtype=float)

        if arr.ndim == 0:
            arr = arr.reshape(1)

        if arr.ndim != 1:
            raise ValueError("point predictions must be scalar or 1D.")

        return arr.reshape(1, -1)

    def _check_is_fitted(self) -> None:
        """Raise if the selector has not been fitted."""
        if not hasattr(self, "result_"):
            raise RuntimeError("BartPredictiveSelector is not fitted.")


__all__ = [
    "BartPredictiveSelector",
    "VALID_TASKS",
    "VALID_LOSSES",
]