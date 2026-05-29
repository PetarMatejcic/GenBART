"""Regression BART estimator for continuous-response modeling.

This module defines ``RegBart``, a Bayesian additive regression trees estimator
for continuous outcomes. The model rescales the response, samples posterior
sum-of-tree draws with Bayesian backfitting MCMC, stores retained forests in a
packed prediction format, and exposes prediction, interval, marginalization, and
variance-update utilities.
"""

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
)
from genbart.baseBART import BaseBART


class RegBart(BaseBART):
    """Bayesian additive regression trees model for continuous outcomes.

    ``RegBart`` fits a sum-of-trees regression model with Gaussian observation
    noise. Responses are shifted and scaled before sampling so the terminal-node
    prior can shrink each tree toward a weak contribution. Posterior forest draws
    are retained and used to compute point predictions, posterior intervals,
    variable-usage summaries, and approximate marginal effects.

    Attributes:
        nu: Degrees of freedom for the inverse-chi-square prior on ``sigma2``.
        q: Prior quantile level used to calibrate the scale parameter ``lambda_``.
        y_shift: Center used to transform the original response before fitting.
        y_scale: Scale used to transform the original response before fitting.
        lambda_: Scale parameter for the inverse-chi-square prior on ``sigma2``.
    """
    nu: float
    q: float
    y_shift: float
    y_scale: float
    lambda_: float

    def __init__(self,
                 m=200,
                 alpha=0.95,
                 beta=2.0,
                 k=2.0,
                 nu=3.0,
                 q=0.90,
                 n_burn=200,
                 n_samples=1000,
                 move_distribution=(0.25, 0.25, 0.40, 0.10),
                 random_state=0):
        """Initialize regression BART hyperparameters.

        Args:
            m: Number of trees in each posterior forest draw.
            alpha: Base probability parameter for the tree-splitting prior.
            beta: Depth penalty parameter for the tree-splitting prior.
            k: Shrinkage parameter controlling the terminal-node prior variance.
            nu: Degrees of freedom for the inverse-chi-square prior on observation
                variance.
            q: Quantile level used to calibrate the observation-variance prior.
            n_burn: Number of burn-in MCMC iterations to discard.
            n_samples: Number of posterior MCMC iterations to retain.
            move_distribution: Proposal probabilities for grow, prune, change, and
                swap moves, respectively.
            random_state: Seed for reproducible NumPy and backend randomness.
        """
        super().__init__(m=m,
                         alpha=alpha,
                         beta=beta,
                         k=k,
                         n_burn=n_burn,
                         n_samples=n_samples,
                         move_distribution=move_distribution,
                         random_state=random_state)
        self.nu = nu
        self.q = q

    def get_params(self):
        params_dict = super().get_params()
        params_dict["nu"] = self.nu
        params_dict["q"] = self.q
        return params_dict

    def fit(self, X, y):
        """Fit the regression BART model to a feature matrix and continuous response.

        The response is shifted and scaled to approximately the interval ``[-0.5, 0.5]``.
        The method initializes the backend forest, residual state, terminal-node prior,
        and observation-variance prior; runs burn-in iterations; then retains
        ``n_samples`` serialized posterior forest draws for later prediction.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)`` or a one-dimensional
                array for a single predictor.
            y: Continuous response array of length ``n_samples``.

        Returns:
            The fitted ``RegBart`` instance.

        Raises:
            ValueError: If ``X`` has more than two dimensions.
            RuntimeError: If backend forest serialization or packed-forest finalization
                fails.
        """
        # Transforming and storing data.
        self.X = np.asarray(X)
        if self.X.ndim == 1:
            self.X = self.X.reshape((-1, 1))
        elif self.X.ndim > 2:
            raise ValueError
        y = np.asarray(y)
        self.y_scale = max(y) - min(y)
        if self.y_scale < 1e-16:
            self.y_scale = 1.0
        self.y_shift = (min(y)+max(y)) / 2.0
        self.y_work = (y - self.y_shift) / self.y_scale
        self.n, self.p = self.X.shape

        # Initializing trees.
        self._initialize_fit_state()

        # Initializing mu_ij|T_j prior ~ N(0, sigma_mu2)
        self.sigma_mu2 = (0.5 / (self.k * np.sqrt(self.m)))**2
        # Initializing sigma2 prior.
        A_sigma = np.column_stack([self.X, np.ones(self.n)])
        denom = (self.n - np.linalg.matrix_rank(A_sigma))
        if denom > 0:
            sigma_linalg = np.linalg.lstsq(A_sigma, self.y_work, rcond=None)
            rss = np.sum((self.y_work - A_sigma @ sigma_linalg[0])**2)
            self.sigma2 = rss / denom
        else:
            self.sigma2 = np.var(self.y_work)
        self.sigma2 = max(self.sigma2, 1e-12)
        self.lambda_ = (self.sigma2 / self.nu) * chi2.ppf(1-self.q, df=self.nu)

        for _ in range(self.n_burn):
            self._one_mcmc_iteration()

        for sample_idx in range(self.n_samples):
            self._one_mcmc_iteration()
            self._store_posterior_sample(sample_idx)

        self._finalize_packed_forest()
        return self

    def predict(self, X,
                central_measure="mean",
                conf_int=True,
                level: float = 0.90):
        """Predict continuous responses from retained posterior forest draws.

        For each input row, this method evaluates every retained posterior forest draw,
        summarizes the posterior predictive mean function by either the mean or median,
        and optionally returns central posterior intervals. Predictions are transformed
        back to the original response scale before being returned.

        Args:
            X: Input feature matrix or a single feature row. A one-dimensional input is
                treated as one row when the model has multiple predictors, and as a
                column vector when the model has one predictor.
            central_measure: Posterior summary to report. Supported values are
                ``"mean"`` and ``"median"``.
            conf_int: Whether to include posterior interval bounds.
            level: Credible interval level used when ``conf_int`` is true.

        Returns:
            A dictionary containing ``"prediction"`` and, when requested,
            ``"conf_int_low"`` and ``"conf_int_high"``. Values are scalars for a single
            multi-feature row and NumPy arrays for multiple rows.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``central_measure`` is not ``"mean"`` or ``"median"``.
        """
        data = np.asarray(X)
        a = 1 - level
        out = {}

        if self.packed_forest is None:
            raise RuntimeError("Model not fitted!")

        if data.ndim == 1 and self.p > 1:
            predictions = self.packed_forest.draw_sums_row(data)
            if central_measure == "mean":
                out["prediction"] = self._inverse_transform_y(predictions.mean())
            elif central_measure == "median":
                out["prediction"] = self._inverse_transform_y(np.median(predictions))
            else:
                raise ValueError
            if conf_int:
                low_int, high_int = np.quantile(predictions,
                                                [a/2.0, 1 - a/2.0])
                out["conf_int_low"] = self._inverse_transform_y(low_int)
                out["conf_int_high"] = self._inverse_transform_y(high_int)
            return out
        else:
            if data.ndim == 1:
                data = data.reshape((-1, 1))
            predictions = self.packed_forest.draw_sums_matrix(data)
            if central_measure == "mean":
                out["prediction"] = self._inverse_transform_y(predictions.mean(axis=0))
            elif central_measure == "median":
                out["prediction"] = self._inverse_transform_y(np.median(predictions, axis=0))
            else:
                raise ValueError
            if conf_int:
                low_ints, high_ints = np.quantile(predictions,
                                                  [a/2.0, 1 - a/2.0], axis=0)
                out["conf_int_low"] = self._inverse_transform_y(low_ints)
                out["conf_int_high"] = self._inverse_transform_y(high_ints)
            return out

    def partial_dependence(self,
                           variable: int | tuple,
                           grid_samples = 100,
                           central_measure = "mean",
                           level = 0.9):
        alpha = 1 - level
        
        part_dep_preds = np.empty(grid_samples)
        part_dep_low = np.empty(grid_samples)
        part_dep_high = np.empty(grid_samples)

        if isinstance(variable, int):
            grid = np.linspace(self.X[:, variable].min(),
                            self.X[:, variable].max(),
                            grid_samples)
            average_X = self.X.mean(axis=0)
            for i, val in enumerate(grid):
                print(average_X.shape)
                print(val.shape)
                average_X[variable] = val
                preds = self.predict(average_X,
                                     central_measure=central_measure)
                part_dep_preds[i] = preds["prediction"]
                part_dep_low[i] = preds["conf_int_low"]
                part_dep_high[i] = preds["conf_int_high"]

        return {
            "prediction": part_dep_preds,
            "conf_int_low": part_dep_low,
            "conf_int_high": part_dep_high
        }


    
    def evaluate(self, X, y, central_measure: str = "mean"):
        """Evaluate regression predictive performance.

        Parameters
        ----------
        X : array-like
            Feature matrix or single observation.
        y : array-like
            Observed response values.
        central_measure : {"mean", "median"}, default="mean"
            Posterior summary used for point prediction.

        Returns
        -------
        out : dict
            Dictionary of scalar regression metrics.
        """
        y_true = np.asarray(y, dtype=float)
        if y_true.ndim == 0:
            raise ValueError("y must be a 1D array!")
        
        preds = self.predict(X,
                             central_measure=central_measure,
                             conf_int=False)
        y_pred = np.asanyarray(preds["prediction"], dtype=float)

        if y_pred.ndim == 0:
            y_pred = y_pred.reshape(1)

        if y_pred.ndim != 1:
            raise ValueError("Predictions must be a 1D array.")
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f"X and y have incompatible lengths: got {y_pred.shape[0]} "
                             f"predictions and {y_true.shape[0]} labels.")

        residuals = y_true - y_pred
        return {
            "n": int(y_true.shape[0]),
            "central_measure": central_measure,

            "mse": mean_squared_error(y_true, y_pred),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "median_absolute_error": median_absolute_error(y_true, y_pred),
            "max_error": max_error(y_true, y_pred),

            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred),

            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals, ddof=1)) if y_true.shape[0] > 1 else 0.0,
            "residual_median": float(np.median(residuals)),
            "residual_min": float(np.min(residuals)),
            "residual_max": float(np.max(residuals)),
        }

    def _one_mcmc_iteration(self):
        """Run one regression BART MCMC iteration.

        Performs a full Bayesian backfitting sweep over all trees, then draws a new
        observation variance ``sigma2`` from its full conditional distribution.
        """
        self._backfitting_sweep()
        self.sigma2 = self._draw_sigma()

    def _draw_sigma(self):
        """Draw the observation variance from its inverse-gamma full conditional.

        The draw is based on the current residual sum of squares, the prior degrees of
        freedom ``nu``, and the prior scale parameter ``lambda_``.

        Returns:
            A scalar draw of ``sigma2``.
        """
        sse = np.sum((self.residuals)**2)
        return 1.0 / self.rng.gamma(shape=0.5*(self.nu+self.n),
                                    scale=2.0/(self.nu*self.lambda_ + sse))

    def _inverse_transform_y(self, v: np.ndarray):
        """Transform fitted values from the working response scale to the original scale.

        Args:
            v: Scalar or NumPy array on the shifted-and-scaled response scale.

        Returns:
            ``v`` rescaled by ``y_scale`` and shifted by ``y_shift``.
        """
        return (v * self.y_scale) + self.y_shift
