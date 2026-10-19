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
from .base_bart import BaseBART


class ProbitBart(BaseBART):
    """Bayesian additive regression trees classifier with a probit link.

    ``ProbitBart`` models a binary response through a latent Gaussian BART
    regression function. For a latent function ``G(x)``, the probability of the
    positive class is

    .. math::

        P(Y=1 \\mid X=x) = \\Phi(G(x)),

    where :math:`\\Phi` is the standard normal cumulative distribution function.

    Posterior forests are sampled using Bayesian backfitting MCMC. Each retained
    forest produces one posterior draw of ``G(x)``, which is transformed through
    the probit link to obtain a posterior draw of the class probability.

    Parameters
    ----------
    m : int, default=200
        Number of trees in the BART ensemble.

    alpha : float, default=0.95
        Base probability parameter in the tree-structure prior.

    beta : float, default=2.0
        Depth penalty in the tree-structure prior. A node at depth ``d`` has
        splitting probability

        .. math::

            p_{\\mathrm{split}}(d)
            =
            \\frac{\\alpha}{(1+d)^{\\beta}}.

    k : float, default=2.0
        Shrinkage parameter controlling the prior variance of terminal-node means
        on the latent probit scale:

        .. math::

            \\sigma_{\\mu}^{2}
            =
            \\left(
            \\frac{3}{k\\sqrt{m}}
            \\right)^2.

        Larger ``k`` produces stronger shrinkage toward small individual-tree
        contributions.

    n_burn : int, default=200
        Number of initial MCMC iterations discarded as burn-in.

    n_samples : int, default=1000
        Number of posterior forest draws retained after burn-in.

    move_distribution : tuple of float, default=(0.25, 0.25, 0.40, 0.10)
        Proposal probabilities for the tree-structure moves

        ``(grow, prune, change, swap)``.

    random_state : int, default=0
        Seed used for reproducible NumPy and backend sampling.

    Attributes
    ----------
    y_obs : np.ndarray
        Binary training labels encoded as 0 and 1.

    Notes
    -----
    The latent Gaussian error variance is fixed to one, as required for
    identifiability of the probit model.

    See the model-parameter guide for a more detailed interpretation of the BART
    prior and MCMC hyperparameters.
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
        """Fit the probit BART classifier.

        Binary outcomes are modeled through latent Gaussian variables. During MCMC,
        latent responses are sampled from truncated normal distributions whose sign
        is determined by the observed class label, and the latent regression
        function is updated using Bayesian backfitting.

        Parameters
        ----------
        X : array-like
            Feature matrix with shape ``(n_observations, n_features)``. A
            one-dimensional array is interpreted as a single predictor.

        y : array-like
            One-dimensional binary response array containing only ``0`` and ``1``.
            ``1`` denotes the positive class.

        Returns
        -------
        ProbitBart
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not one- or two-dimensional, if ``y`` is not one-dimensional,
            if ``X`` and ``y`` contain different numbers of observations, or if
            ``y`` contains values other than 0 and 1.

        Notes
        -----
        The latent Gaussian observation variance is fixed to one. After burn-in,
        ``n_samples`` complete posterior forest draws are retained for probability
        prediction, uncertainty summaries, partial dependence, and variable
        importance.
        """
        self.X = np.asarray(X)
        if self.X.ndim == 1:
            self.X = self.X.reshape((-1, 1))
        elif self.X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        
        self.y_obs = np.asarray(y)
        if self.y_obs.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if self.X.shape[0] != self.y_obs.shape[0]:
            raise ValueError(
                f"X and y have incompatible lengths: "
                f"{self.X.shape[0]} and {self.y_obs.shape[0]}."
            )
        if not np.all((self.y_obs == 0) | (self.y_obs == 1)):
            raise ValueError("y must contain only binary labels 0 and 1.")

        self.n, self.p = self.X.shape
        z = truncnorm(a=0, b=np.inf).rvs(self.n, random_state=self.rng)
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
        """Estimate posterior positive-class probabilities.

        Each retained posterior forest is evaluated at ``X`` to obtain a draw of the
        latent probit function ``G(x)``. Applying the standard normal CDF gives a
        posterior draw of

        .. math::

            P(Y=1 \\mid X=x) = \\Phi(G(x)).

        The returned probability is the posterior mean of these probability draws.
        The interval bounds are posterior credible intervals for the class
        probability.

        Parameters
        ----------
        X : array-like
            Predictor values at which to evaluate the fitted model.

            For a model with multiple predictors, a one-dimensional array is treated
            as a single observation. For a model with one predictor, a one-dimensional
            array is treated as multiple observations.

        level : float, default=0.90
            Probability mass contained in the central credible interval.

        Returns
        -------
        dict
            Dictionary containing:

            - ``"probs"``: posterior mean probability of the positive class.
            - ``"conf_int_low"``: lower posterior credible bound.
            - ``"conf_int_high"``: upper posterior credible bound.

            Values are scalars for one multi-feature observation and one-dimensional
            NumPy arrays for multiple observations.

        Notes
        -----
        The credible interval describes posterior uncertainty in the class
        probability. It is not a confidence interval for the observed class label.
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
        """Predict binary class labels.

        Posterior mean positive-class probabilities are obtained with
        :meth:`predict_probs` and converted to class labels using ``threshold``.

        Parameters
        ----------
        X : array-like
            Predictor values at which to classify observations.

        threshold : float, default=0.5
            Probability cutoff for assigning the positive class. Probabilities greater
            than or equal to ``threshold`` are assigned to class 1.

        Returns
        -------
        bool or np.ndarray
            Boolean class prediction or array of predictions. ``True`` corresponds to
            class 1 and ``False`` to class 0.

        Notes
        -----
        Changing ``threshold`` affects only the conversion from probabilities to
        class labels; it does not refit the BART model.
        """
        probs = self.predict_probs(X)["probs"]
        return probs >= threshold

    def partial_dependence(self,
                           variable: int,
                           grid_samples = 100,
                           central_measure = "mean",
                           level = 0.9):
        """Compute one-variable partial dependence for class probabilities.

        For each value on a grid spanning the observed range of the selected
        predictor, that predictor is replaced by the grid value for every training
        observation. Posterior class probabilities are then averaged over the
        training observations separately for each posterior draw.

        The resulting posterior partial-dependence draws are summarized by their mean
        or median and by a central credible interval.

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
            - ``"prediction"``: posterior mean or median partial-dependence probability.
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
        For predictor :math:`x_j`, the probability partial-dependence function is
        approximated by

        .. math::

            \\operatorname{PD}(x_j)
            =
            \\frac{1}{n}
            \\sum_{i=1}^{n}
            P(Y=1 \\mid x_j, x_{i,-j}).

        The credible interval describes posterior uncertainty in the
        partial-dependence function, not variability across individual observations.
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

            prob_draws = self.posterior_sample_draws(X_eval)

            # One partial-dependence probability per posterior draw.
            pd_draws = prob_draws.mean(axis=1)

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

    def evaluate(self, X, y, threshold: float = 0.5):
        """Evaluate binary classification performance.

        Metrics are computed from posterior mean positive-class probabilities.
        Probabilities are converted to class predictions using ``threshold``.

        Parameters
        ----------
        X : array-like
            Predictor values for the observations to evaluate.

        y : array-like
            Observed binary labels encoded as 0 and 1.

        threshold : float, default=0.5
            Probability cutoff used to assign the positive class.

        Returns
        -------
        dict
            Dictionary containing:

            - ``"n"``: number of evaluated observations.
            - ``"threshold"``: probability threshold used for classification.
            - ``"accuracy"``: fraction of correctly classified observations.
            - ``"balanced_accuracy"``: mean of sensitivity and specificity.
            - ``"sensitivity"``: true-positive rate.
            - ``"specificity"``: true-negative rate.
            - ``"precision"``: positive predictive value.
            - ``"f1"``: harmonic mean of precision and sensitivity.
            - ``"brier_score"``: mean squared error of predicted probabilities.
            - ``"log_loss"``: binary cross-entropy loss.
            - ``"tn"``, ``"fp"``, ``"fn"``, ``"tp"``: confusion-matrix counts.
            - ``"roc_auc"``: area under the ROC curve.
            - ``"average_precision"``: area under the precision-recall curve using
              the average-precision definition.

        Raises
        ------
        ValueError
            If ``y`` is not one-dimensional, contains labels other than 0 and 1, or
            contains a different number of observations than ``X``.

        Notes
        -----
        ``roc_auc`` and ``average_precision`` are returned as ``NaN`` when the
        evaluation data contain only one observed class.

        Threshold-independent metrics such as ROC AUC, average precision, Brier score,
        and log loss use the posterior mean probabilities directly.
        """
        y_true = np.asarray(y)
        if y_true.ndim != 1:
            raise ValueError("y must be a 1D array")
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y must contain only binary labels 0 and 1.")
        y_true = y_true.astype(int)
        
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
        """Summarize probability calibration in bins.

        Predicted positive-class probabilities are grouped into bins. For each
        non-empty bin, the method compares the mean predicted probability with the
        observed proportion of positive outcomes.

        Parameters
        ----------
        X : array-like
            Predictor values for the observations to evaluate.

        y : array-like
            Observed binary labels encoded as 0 and 1.

        n_bins : int, default=10
            Requested number of probability bins.

        strategy : {"uniform", "quantile"}, default="uniform"
            Method used to construct bins.

            ``"uniform"`` divides the interval ``[0, 1]`` into equal-width bins.

            ``"quantile"`` chooses boundaries from quantiles of the predicted
            probabilities so that bins contain approximately equal numbers of
            observations.

        Returns
        -------
        dict
            Dictionary containing:

            - ``"prob_pred"``: mean predicted probability in each non-empty bin.
            - ``"prob_true"``: observed positive-class frequency in each non-empty bin.
            - ``"bin_count"``: number of observations in each returned bin.
            - ``"bin_left"``: left edge of each returned bin.
            - ``"bin_right"``: right edge of each returned bin.
            - ``"bin_edges"``: complete array of constructed bin boundaries.
            - ``"n_bins"``: requested number of bins.
            - ``"strategy"``: binning strategy.

        Raises
        ------
        ValueError
            If ``y`` is invalid, ``X`` and ``y`` contain different numbers of
            observations, ``n_bins`` is not positive, ``strategy`` is unsupported, or
            quantile binning cannot construct distinct bin boundaries.

        Notes
        -----
        Empty bins are omitted from the per-bin output arrays.

        With quantile binning, repeated predicted probabilities can cause duplicate
        quantile boundaries to be removed, so the effective number of bins may be
        smaller than ``n_bins``.
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
    
    def posterior_sample_draws(self, X):
        """Return posterior draws of positive-class probabilities.

        Each retained posterior forest is evaluated at ``X`` and transformed through
        the standard normal CDF.

        Parameters
        ----------
        X : array-like
            Predictor values at which to evaluate the fitted model.

        Returns
        -------
        np.ndarray
            Array with shape ``(n_samples, n_observations)``.

            Each row corresponds to one retained posterior forest and each column to
            one observation. Each value is a posterior draw of
            :math:`P(Y=1 \\mid X=x)`.

        Notes
        -----
        This method is useful for custom posterior summaries that are not provided by
        :meth:`predict_probs`.
        """
        return self._predict_prob_draws(X)

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
