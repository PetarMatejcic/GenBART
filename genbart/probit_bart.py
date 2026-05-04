"""Probit BART estimator for binary-response classification.

This module defines ``ProbitBart``, a Bayesian additive regression trees
classifier using a probit likelihood. The sampler introduces latent Gaussian
working responses, updates them with truncated-normal draws, fits posterior
sum-of-tree functions by Bayesian backfitting, and converts retained posterior
draws to class probabilities with the standard normal CDF.
"""

import numpy as np
from scipy.stats import truncnorm, norm
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

        self._init_trees()
        self._init_common_arrays()
        self._init_packed_builder()
        self._vi_sum = np.zeros(self.p)

        self.sigma2 = 1.0
        self.sigma_mu2 = (3.0 / (self.k * np.sqrt(self.m))) ** 2

        for _ in range(self.n_burn):
            self._one_mcmc_iteration()

        for _ in range(self.n_samples):
            self._one_mcmc_iteration()

            variable, value, left, right, mu, tree_offset = self._serialize_forest()
            self._append_serialized_forest_block(variable,
                                                 value,
                                                 left,
                                                 right,
                                                 mu,
                                                 tree_offset)

            mask = variable >= 0
            if np.any(mask):
                variable_counts = np.bincount(variable[mask], minlength=self.p)
                variable_total = int(mask.sum())
                self._vi_sum += variable_counts / variable_total
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
        data = np.asarray(X)
        a = 1 - level
        out = {}

        if data.ndim == 1 and self.p > 1:
            g = self.packed_forest.draw_sums_row(data)
            probs = norm.cdf(g)
            out["probs"] = probs.mean()
            conf_low, conf_high = np.quantile(probs, [a/2.0, 1 - a/2.0])
            out["conf_int_low"] = conf_low
            out["conf_int_high"] = conf_high
            return out
        else:
            if data.ndim == 1:
                data = data.reshape((-1, 1))
            g = self.packed_forest.draw_sums_matrix(data)
            probs = norm.cdf(g)
            out["probs"] = probs.mean(axis=0)
            conf_low, conf_high = np.quantile(probs,
                                              [a/2.0, 1 - a/2.0], axis=0)
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
