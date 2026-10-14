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
from genbart.base_bart import BaseBART


class RegBart(BaseBART):
    """Bayesian additive regression trees model for continuous outcomes.

    ``RegBart`` models a continuous response as a sum of regression trees with
    Gaussian observation noise. Posterior tree ensembles are sampled using
    Bayesian backfitting MCMC.

    Before fitting, the response is shifted and scaled to approximately
    ``[-0.5, 0.5]``. Predictions are transformed back to the original response
    scale before being returned.

    Parameters
    ----------
    m : int, default=200
        Number of trees in the BART ensemble.

        Increasing ``m`` gives the model more trees over which to distribute the
        fitted regression function, but also increases sampling and storage cost.
        The terminal-node prior is scaled with ``m`` so that individual trees are
        more strongly shrunk as the ensemble becomes larger.

    alpha : float, default=0.95
        Base probability parameter in the tree-structure prior.

    beta : float, default=2.0
        Depth penalty in the tree-structure prior.

        A node at depth ``d`` is assigned splitting probability

        .. math::

            p_{\\mathrm{split}}(d)
            =
            \\frac{\\alpha}{(1+d)^{\\beta}}.

        Larger ``beta`` therefore favors shallower trees more strongly.

    k : float, default=2.0
        Shrinkage parameter controlling the prior variance of terminal-node
        means. For regression BART,

        .. math::

            \\sigma_{\\mu}^{2}
            =
            \\left(
            \\frac{0.5}{k\\sqrt{m}}
            \\right)^2.

        Larger ``k`` produces stronger shrinkage toward small individual-tree
        contributions.

    nu : float, default=3.0
        Degrees of freedom of the scaled inverse-chi-squared prior on the
        observation variance. Larger values make the prior more concentrated.

    q : float, default=0.90
        Quantile used to calibrate the observation-variance prior.

        The prior scale is chosen so that the initial residual variance estimate
        used during fitting is the ``q`` quantile of the prior distribution.
        Larger ``q`` shifts more prior probability toward smaller observation
        variances relative to that estimate.

    n_burn : int, default=200
        Number of initial MCMC iterations discarded as burn-in.

    n_samples : int, default=1000
        Number of posterior MCMC draws retained after burn-in.

    move_distribution : tuple of float, default=(0.25, 0.25, 0.40, 0.10)
        Proposal probabilities for the tree-structure moves

        ``(grow, prune, change, swap)``.

    random_state : int, default=0
        Seed used for reproducible NumPy and backend sampling.

    Attributes
    ----------
    y_shift : float
        Center used to transform the response before fitting.

    y_scale : float
        Scale used to transform the response before fitting.

    lambda_ : float
        Scale parameter of the observation-variance prior after calibration.

    Notes
    -----
    Each retained MCMC iteration stores one complete posterior forest draw. These
    draws are used for prediction, credible intervals, partial dependence, and
    variable-importance summaries.
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
        elif self.X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if self.X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have incompatible lengths: "
                f"{self.X.shape[0]} and {y.shape[0]}."
            )
        
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
        """Predict continuous responses from posterior regression-function draws.

        For each input observation, every retained posterior forest is evaluated to
        produce draws of the regression function

            f(x) = E[Y | X = x].

        The returned point prediction is the mean or median of those posterior draws.
        If ``conf_int=True``, the method also returns a central credible interval for
        ``f(x)``.

        The credible interval represents posterior uncertainty in the conditional
        mean response. It does not include observation noise and therefore is not a
        posterior predictive interval for a new observed response.

        Parameters
        ----------
        X : array-like
            Predictor values at which to evaluate the fitted model.

            For a model with multiple predictors, a one-dimensional array is treated
            as a single observation. For a model with one predictor, a one-dimensional
            array is treated as multiple observations.

        central_measure : {"mean", "median"}, default="mean"
            Posterior summary used for the point prediction.

        conf_int : bool, default=True
            Whether to return lower and upper credible bounds for the regression
            function.

        level : float, default=0.90
            Probability mass contained in the central credible interval.

        Returns
        -------
        dict
            Dictionary containing:

            - ``"prediction"``: posterior mean or median of the regression function.
            - ``"conf_int_low"``: lower credible bound, when ``conf_int=True``.
            - ``"conf_int_high"``: upper credible bound, when ``conf_int=True``.

            For a single multi-feature observation, values are scalars. For multiple
            observations, values are one-dimensional NumPy arrays.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        ValueError
            If ``central_measure`` is not ``"mean"`` or ``"median"``.

        Notes
        -----
        These intervals quantify uncertainty in the fitted regression function only.
        To construct posterior predictive intervals for future observations, draws of
        the observation noise would also need to be included.
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
        
    def posterior_sample_draws(self, X):
        """Return posterior draws of the regression function.

        Each retained posterior forest is evaluated at ``X``. The returned array
        therefore contains draws of

            f(x) = E[Y | X = x]

        on the original response scale.

        Parameters
        ----------
        X : array-like
            Predictor values at which to evaluate the fitted model.

            For a model with multiple predictors, a one-dimensional array is treated
            as a single observation. For a model with one predictor, a one-dimensional
            array is treated as multiple observations.

        Returns
        -------
        np.ndarray
            Array with shape ``(n_samples, n_observations)``.

            Each row corresponds to one retained posterior forest draw and each column
            corresponds to one observation in ``X``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Notes
        -----
        These are posterior draws of the regression function only. They do not
        include observation noise, so they should not be interpreted as posterior
        predictive draws for future observed responses.

        This method is useful when custom posterior summaries are needed instead of
        the mean, median, and credible intervals provided by :meth:`predict`.
        """
        if self.packed_forest is None:
            raise RuntimeError("Model not fitted.")

        data = np.asarray(X)

        if data.ndim == 1 and self.p > 1:
            draws = self.packed_forest.draw_sums_row(data).reshape(-1, 1)
        else:
            if data.ndim == 1:
                data = data.reshape((-1, 1))
            draws = self.packed_forest.draw_sums_matrix(data)

        return self._inverse_transform_y(draws)

    def partial_dependence(self,
                           variable: int,
                           grid_samples = 100,
                           central_measure = "mean",
                           level = 0.9):
        """Compute one-variable partial dependence.

        For each value on a grid spanning the observed range of the selected
        predictor, that predictor is replaced by the grid value for every training
        observation. The fitted model is then evaluated on all modified observations.

        Predictions are averaged over the training observations separately for each
        posterior draw. The resulting posterior partial-dependence draws are then
        summarized by their mean or median and by a central credible interval.

        Parameters
        ----------
        variable : int
            Column index of the predictor whose partial dependence is evaluated.

        grid_samples : int, default=100
            Number of evenly spaced grid values between the observed minimum and
            maximum of the selected predictor.

        central_measure : {"mean", "median"}, default="mean"
            Posterior summary used for the point estimate of partial dependence.

        level : float, default=0.9
            Probability mass contained in the central credible interval.

        Returns
        -------
        dict
            Dictionary containing:

            - ``"grid"``: predictor values at which partial dependence is evaluated.
            - ``"prediction"``: posterior mean or median partial-dependence value.
            - ``"conf_int_low"``: lower posterior credible bound.
            - ``"conf_int_high"``: upper posterior credible bound.

            Each returned array has length ``grid_samples``.

        Raises
        ------
        TypeError
            If ``variable`` is not an integer column index.

        ValueError
            If ``central_measure`` is not ``"mean"`` or ``"median"``.

        Notes
        -----
        For a predictor :math:`x_j`, the partial-dependence function is approximated by

        .. math::

            \\operatorname{PD}(x_j)
            =
            \\frac{1}{n}
            \\sum_{i=1}^{n}
            f(x_j, x_{i,-j})

        where the average is taken over the observed training values of all remaining
        predictors.

        This preserves the empirical distribution of the other predictors rather than
        fixing them to a single reference value.

        The credible interval describes posterior uncertainty in the partial-
        dependence function. It does not represent variability across individual
        observations.
        """
        if not isinstance(variable, int):
            raise TypeError("variable must be an integer column index.")

        grid = np.linspace(
            self.X[:, variable].min(),
            self.X[:, variable].max(),
            grid_samples,
        )

        part_dep_preds = np.empty(grid_samples)
        part_dep_low = np.empty(grid_samples)
        part_dep_high = np.empty(grid_samples)

        alpha = 1 - level

        for i, val in enumerate(grid):
            X_eval = self.X.copy()
            X_eval[:, variable] = val

            draws = self.posterior_sample_draws(X_eval)

            # One partial-dependence value per posterior draw.
            pd_draws = draws.mean(axis=1)

            if central_measure == "mean":
                part_dep_preds[i] = pd_draws.mean()
            elif central_measure == "median":
                part_dep_preds[i] = np.median(pd_draws)
            else:
                raise ValueError(
                    "central_measure must be 'mean' or 'median'."
                )

            part_dep_low[i], part_dep_high[i] = np.quantile(
                pd_draws,
                [alpha / 2.0, 1 - alpha / 2.0],
            )

        return {
            "grid": grid,
            "prediction": part_dep_preds,
            "conf_int_low": part_dep_low,
            "conf_int_high": part_dep_high,
        }


    def evaluate(self, X, y, central_measure: str = "mean"):
        """Evaluate regression predictive performance.

        Predictions are computed using :meth:`predict` with credible intervals
        disabled. All reported metrics therefore evaluate the posterior point
        prediction determined by ``central_measure``; posterior uncertainty is not
        included in the metrics.

        Parameters
        ----------
        X : array-like
            Predictor values for the observations to evaluate.

        y : array-like
            Observed continuous response values. Must contain one value for each
            observation in ``X``.

        central_measure : {"mean", "median"}, default="mean"
            Posterior summary used as the point prediction before computing the
            regression metrics.

        Returns
        -------
        dict
            Dictionary containing:

            - ``"n"``: number of evaluated observations.
            - ``"central_measure"``: posterior summary used for prediction.
            - ``"mse"``: mean squared error.
            - ``"rmse"``: root mean squared error.
            - ``"mae"``: mean absolute error.
            - ``"median_absolute_error"``: median absolute error.
            - ``"max_error"``: largest absolute residual.
            - ``"r2"``: coefficient of determination.
            - ``"explained_variance"``: explained-variance score.
            - ``"residual_mean"``: mean of ``y - prediction``.
            - ``"residual_std"``: sample standard deviation of the residuals.
            - ``"residual_median"``: median residual.
            - ``"residual_min"``: minimum residual.
            - ``"residual_max"``: maximum residual.

        Raises
        ------
        ValueError
            If ``y`` is not one-dimensional, if predictions are not one-dimensional,
            or if ``X`` and ``y`` contain different numbers of observations.

        Notes
        -----
        The residual convention used throughout this method is

            residual = observed response - predicted response.

        Metrics such as MSE, MAE, and R-squared describe point-prediction
        performance only. Use :meth:`predict` or :meth:`posterior_sample_draws` when
        posterior uncertainty is required.
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
