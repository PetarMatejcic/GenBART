"""Probit BART estimator for binary-response classification.

This module defines ``ProbitBart``, a Bayesian additive regression trees
classifier using a probit likelihood. The sampler introduces latent Gaussian
working responses, updates them with truncated-normal draws, fits posterior
sum-of-tree functions by Bayesian backfitting, and converts retained posterior
draws to class probabilities with the standard normal CDF.
"""

import numpy as np
from scipy.stats import truncnorm, norm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from .baseBART import BaseBART


class ProbitBart(BaseBART):
    """Bayesian additive regression trees classifier with a probit link.

    ``ProbitBart`` models binary outcomes through a latent Gaussian regression
    function. Conditional class probabilities are computed as ``Phi(G(x))``, where
    ``G(x)`` is the posterior sum-of-trees draw and ``Phi`` is the standard normal
    CDF. The observation variance is fixed to one, and the terminal-node prior is
    scaled for the latent probit scale.

    Attributes:
        y_obs: Observed binary response vector used to truncate latent Gaussian
            working responses during MCMC.
    """
    y_obs: np.ndarray

    def __init__(self,
                 m=200,
                 alpha=0.95,
                 beta=2.0,
                 k=2.0,
                 n_burn=200,
                 n_samples=1000,
                 move_distribution=(0.25, 0.25, 0.40, 0.10),
                 random_state=0):
        """Initialize probit BART hyperparameters.

        Args:
            m: Number of trees in each posterior forest draw.
            alpha: Base probability parameter for the tree-splitting prior.
            beta: Depth penalty parameter for the tree-splitting prior.
            k: Shrinkage parameter controlling terminal-node prior variance on the
                latent probit scale.
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

    def fit(self, X, y):
        """Fit the probit BART classifier to binary response data.

        The method stores the feature matrix and observed labels, initializes latent
        Gaussian working responses with signs consistent with the binary outcomes,
        sets the probit observation variance to one, initializes the terminal-node prior
        on the latent scale, runs burn-in iterations, and retains serialized posterior
        forest draws for probability prediction.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)`` or a one-dimensional
                array for a single predictor.
            y: Binary response array of length ``n_samples``. Values equal to ``1`` are
                treated as the positive class; all other values are treated as the
                negative class.

        Returns:
            The fitted ``ProbitBart`` instance.

        Raises:
            ValueError: If ``X`` has more than two dimensions.
            RuntimeError: If backend forest serialization or packed-forest finalization
                fails.
        """
        self.X = np.asarray(X)
        if self.X.ndim == 1:
            self.X = self.X.reshape((-1, 1))
        elif self.X.ndim > 2:
            raise ValueError
        self.y_obs = np.asarray(y)
        self.n, self.p = self.X.shape
        z = truncnorm(a=0, b=np.inf).rvs(self.n)
        self.y_work = np.where(self.y_obs == 1, z, -z)

        self._initialize_fit_state()

        self.sigma2 = 1.0
        self.sigma_mu2 = (3.0 / (self.k * np.sqrt(self.m))) ** 2

        for _ in range(self.n_burn):
            self._one_mcmc_iteration()

        for sample_idx in range(self.n_samples):
            self._one_mcmc_iteration()
            self._store_posterior_sample(sample_idx)
        
        self._finalize_packed_forest()
        return self

    def predict_probs(self, X, level: float = 0.90):
        """Estimate posterior class probabilities and credible intervals.

        Each retained posterior forest draw is evaluated at ``X`` to obtain latent
        function values, which are transformed through the standard normal CDF. The
        returned probability is the posterior mean of those transformed draws, with
        central posterior interval bounds computed from their quantiles.

        Args:
            X: Input feature matrix or a single feature row. A one-dimensional input is
                treated as one row when the model has multiple predictors, and as a
                column vector when the model has one predictor.
            level: Credible interval level for posterior probability intervals.

        Returns:
            A dictionary containing ``"probs"``, ``"conf_int_low"``, and
            ``"conf_int_high"``. Values are scalars for a single multi-feature row and
            NumPy arrays for multiple rows.

        Notes:
            This method requires a fitted model with a populated ``packed_forest``.
        """
        prob_draws = self._predict_prob_draws(X)
        a = 1 - level
        out = {}

        probs = prob_draws.mean(axis=0)
        conf_low, conf_high = np.quantile(prob_draws, [a/2.0, 1 - a/2.0], axis=0)

        if probs.shape[0] == 1:
            out["probs"] = probs[0]
            out["conf_int_low"] = conf_low[0]
            out["conf_int_high"] = conf_high[0]
        else:
            out["probs"] = probs
            out["conf_int_low"] = conf_low
            out["conf_int_high"] = conf_high

        return out

    def predict(self, X, threshold: float = 0.5):
        """Predict binary class labels from posterior mean class probabilities.

        Args:
            X: Input feature matrix or single feature row.
            threshold: Probability cutoff used to assign the positive class.

        Returns:
            A Boolean scalar or NumPy array indicating whether each posterior mean class
            probability is greater than or equal to ``threshold``.
        """
        probs = self.predict_probs(X)["probs"]
        return probs >= threshold

    def partial_dependence(self,
                           variable: int | tuple,
                           grid_samples = 100,
                           central_measure = "mean",
                           level = 0.9):
        """Compute a one-variable mean-reference probability curve.

        For a selected predictor, this method constructs an evenly spaced grid over the
        observed training range of that predictor. At each grid value, all other
        predictors are fixed to their training-sample means, the fitted probit BART model
        is evaluated at the resulting reference row, and the posterior class-probability
        summary and credible interval are stored.

        Parameters
        ----------
        variable : int or tuple
            Predictor to vary. Currently only an integer column index is implemented.
            Tuple input is reserved for future multi-variable effect surfaces.
        grid_samples : int, default=100
            Number of evenly spaced grid values between the observed minimum and maximum
            of the selected predictor.
        central_measure : {"mean", "median"}, default="mean"
            Reserved for API consistency with regression BART. Currently unused because
            ``predict_probs`` returns the posterior mean probability.
        level : float, default=0.9
            Credible interval level passed to ``predict_probs``.

        Returns
        -------
        dict
            Dictionary with three arrays, each of length ``grid_samples``:

            - ``"prediction"``: posterior mean probability at each grid value.
            - ``"conf_int_low"``: lower credible interval bound for the probability.
            - ``"conf_int_high"``: upper credible interval bound for the probability.

        Notes
        -----
        This is a mean-reference probability curve, not classical partial dependence.
        Classical partial dependence averages predictions over the observed training rows
        after replacing the selected variable by each grid value. Here, the complement
        variables are fixed to their column means instead.

        For clarity, consider renaming the returned ``"prediction"`` key to ``"probs"``
        or renaming the method to something like ``mean_effect`` or
        ``mean_reference_curve``.
        """
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
                average_X[variable] = val
                preds = self.predict_probs(average_X,
                                           level=level)
                part_dep_preds[i] = preds["probs"]
                part_dep_low[i] = preds["conf_int_low"]
                part_dep_high[i] = preds["conf_int_high"]

        return {
            "prediction": part_dep_preds,
            "conf_int_low": part_dep_low,
            "conf_int_high": part_dep_high
        }

    def evaluate(self, X, y, threshold: float = 0.5):
        """Evaluate binary classification performance.

        Parameters
        ----------
        X : array-like
            Feature matrix or single observation.
        y : array-like
            Observed binary labels, encoded as 0 and 1.
        threshold : float, default=0.5
            Probability cutoff used to convert predicted probabilities into
            hard class predictions.

        Returns
        -------
        out : dict
            Dictionary of scalar classification metrics.
        """
        y_true = np.asarray(y).astype(int)
        if y_true.ndim != 1:
            raise ValueError("y must be a 1D array")
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y must contain only binary labels 0 and 1.")
        
        prob_draws = self._predict_prob_draws(X)
        probs = prob_draws.mean(axis=0)

        if probs.shape[0] != y_true.shape[0]:
            raise ValueError(f"X and y have incompatible lengths: got {probs.shape[0]} "
                             f"predictions and {y_true.shape[0]} labels.")
        
        y_pred = (probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true,
                                          y_pred,
                                          labels=[0, 1]).ravel()
        
        out = {
                "n": int(y_true.shape[0]),
                "threshold": float(threshold),

                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
                "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),

                "brier_score": brier_score_loss(y_true, probs),
                "log_loss": log_loss(y_true, probs, labels=[0, 1]),

                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

        if np.unique(y_true).size == 2:
            out["roc_auc"] = roc_auc_score(y_true, probs)
            out["average_precision"] = average_precision_score(y_true, probs)
        else:
            out["roc_auc"] = np.nan
            out["average_precision"] = np.nan

        return out

    def calibration_curve(self,
                          X,
                          y,
                          n_bins: int = 10,
                          strategy: str = "uniform"):
        """Return calibration-curve data for predicted probabilities.

        Parameters
        ----------
        X : array-like
            Feature matrix or single observation.
        y : array-like
            Observed binary labels, encoded as 0 and 1.
        n_bins : int, default=10
            Number of probability bins.
        strategy : {"uniform", "quantile"}, default="uniform"
            Binning strategy. "uniform" uses equal-width bins on [0, 1].
            "quantile" uses bins with approximately equal numbers of observations.

        Returns
        -------
        out : dict
            Dictionary containing mean predicted probability, observed event
            frequency, bin counts, and bin edges.
        """
        y_true = np.asarray(y).astype(int)
        if y_true.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y must contain only binary labels 0 and 1.")

        if n_bins <= 0:
            raise ValueError("n_bins must be positive.")

        prob_draws = self._predict_prob_draws(X)
        probs = prob_draws.mean(axis=0)

        if probs.shape[0] != y_true.shape[0]:
            raise ValueError(f"X and y have incompatible lengths: got {probs.shape[0]} "
                             f"predictions and {y_true.shape[0]} labels.")

        if strategy == "uniform":
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        elif strategy == "quantile":
            bin_edges = np.quantile(
                probs,
                np.linspace(0.0, 1.0, n_bins + 1)
            )
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
            bin_edges = np.unique(bin_edges)

            if bin_edges.shape[0] < 2:
                raise ValueError(
                    "Cannot create quantile bins because all predicted "
                    "probabilities are identical."
                )
        else:
            raise ValueError("strategy must be either 'uniform' or 'quantile'.")

        bin_ids = np.digitize(probs, bin_edges[1:-1], right=True)

        prob_pred = []
        prob_true = []
        bin_count = []
        bin_left = []
        bin_right = []

        for b in range(len(bin_edges) - 1):
            mask = bin_ids == b
            if not np.any(mask):
                continue

            prob_pred.append(probs[mask].mean())
            prob_true.append(y_true[mask].mean())
            bin_count.append(mask.sum())
            bin_left.append(bin_edges[b])
            bin_right.append(bin_edges[b + 1])

        return {
            "prob_pred": np.asarray(prob_pred),
            "prob_true": np.asarray(prob_true),
            "bin_count": np.asarray(bin_count, dtype=int),
            "bin_left": np.asarray(bin_left),
            "bin_right": np.asarray(bin_right),
            "bin_edges": bin_edges,
            "n_bins": int(n_bins),
            "strategy": strategy,
        }

    def _predict_prob_draws(self, X):
        """Return posterior draws of P(Y = 1 | X).

        Returns
        -------
        probs : np.ndarray
            Array with shape (n_samples, n_observations). Each row is one
            posterior draw and each column is one observation.
        """
        data = np.asarray(X)

        if data.ndim == 1 and self.p > 1:
            g = self.packed_forest.draw_sums_row(data)
            return norm.cdf(g).reshape((-1, 1))
        else:
            if data.ndim == 1:
                data = data.reshape((-1, 1))
            g = self.packed_forest.draw_sums_matrix(data)
            return norm.cdf(g)

    def _one_mcmc_iteration(self):
        """Run one probit BART MCMC iteration.

        Draws new latent Gaussian working responses conditional on the observed binary
        labels and current fitted values, updates residuals to reflect the changed
        working response, and performs one full Bayesian backfitting sweep over the
        trees.
        """
        old_z = self.y_work.copy()
        self._draw_latent_z()
        self.residuals += self.y_work - old_z
        self._backfitting_sweep()

    def _draw_latent_z(self):
        """Draw latent Gaussian responses from outcome-constrained truncations.

        For observations in the positive class, latent values are sampled from a normal
        distribution truncated below at zero. For observations in the negative class,
        latent values are sampled from a normal distribution truncated above at zero.
        The normal locations are the current fitted latent sums.

        Notes:
            The method updates ``y_work`` in place and assumes ``y_obs``, ``y_work``,
            and ``residuals`` have already been initialized.
        """
        pos = self.y_obs == 1
        neg = ~pos
        fitted_sums = self.y_work - self.residuals

        a_pos = -fitted_sums[pos]
        b_pos = np.full(a_pos.shape, np.inf)
        self.y_work[pos] = truncnorm.rvs(a=a_pos,
                                         b=b_pos,
                                         loc=fitted_sums[pos],
                                         scale=1.0,
                                         random_state=self.rng,)

        a_neg = np.full(np.sum(neg), -np.inf)
        b_neg = -fitted_sums[neg]
        self.y_work[neg] = truncnorm.rvs(a=a_neg,
                                         b=b_neg,
                                         loc=fitted_sums[neg],
                                         scale=1.0,
                                         random_state=self.rng,)
