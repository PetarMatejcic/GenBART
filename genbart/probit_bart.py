import numpy as np
from scipy.stats import truncnorm, norm
from .baseBART import BaseBART


class ProbitBart(BaseBART):
    y_obs: np.ndarray

    def __init__(self,
                 m=200,
                 alpha=0.95,
                 beta=2.0,
                 k=2.0,
                 n_burn=200,
                 n_samples=1000,
                 move_distribution=(0.25, 0.25, 0.40, 0.10),
                 random_state=None):
        
        super().__init__(m = m,
                            alpha = alpha,
                            beta = beta,
                            k = k,
                            n_burn = n_burn,
                            n_samples = n_samples,
                            move_distribution = move_distribution,
                            random_state=random_state)
    
    def fit(self, X, y):
        self.X = np.asarray(X)
        if self.X.ndim == 1:
            self.X = self.X.reshape((-1, 1))
        elif self.X.ndim > 2:
            raise ValueError
        self.y_obs = np.asarray(y)
        self.n, self.p = self.X.shape

        self._init_trees()
        self._init_common_arrays()

        self.sigma2 = 1.0
        self.sigma_mu2 = (3.0 / (self.k * np.sqrt(self.m))) ** 2

        z = truncnorm(a=0, b=np.inf).rvs(self.n)
        self.y_work = np.where(self.y_obs == 1, z, -z)

        for _ in range(self.n_burn):
            self._one_mcmc_iteration()

        for _ in range(self.n_samples):
            self._one_mcmc_iteration()
            sample = []
            for j in range(self.m):
                sample.append(self.trees[j].serialize())
            self.tree_sample.append({"sample": sample})
        return self
    
    def predict_probs(self, X, level: float = 0.90):
        data = np.asarray(X)
        a = 1 - level
        if data.ndim == 1 and self.p > 1:
            g = np.zeros(self.n_samples)
            for i in range(self.n_samples):
                for j in range(self.m):
                    g[i] += self._predict_serialized_tree_row(data,
                                                            self.tree_sample[i]["sample"][j])
            probs = norm.cdf(g)
            conf_int = np.quantile(probs, [a/2.0, 1 - a/2.0])
            return probs.mean(), conf_int
        else:
            if data.ndim == 1:
                data = data.reshape((-1, 1))
            g = np.zeros((X.shape[0], self.n_samples))
            for i in range(self.n_samples):
                for j in range(self.m):
                    g[:, i] += self._predict_serialized_tree_matrix(data,
                                                                    self.tree_sample[i]["sample"][j])
            probs = norm.cdf(g)
            conf_ints = np.quantile(probs, [a/2.0, 1 - a/2.0], axis=1)
            return probs.mean(axis=1), conf_ints
    
    def predict(self, X, threshold: float = 0.5):
        probs = self.predict_probs(X)[0]
        return probs >= threshold
    
    def _one_mcmc_iteration(self):
        self._draw_latent_z()
        for j in range(self.m):
            self._partial_residuals(j)
            self._draw_tree(j)
            self._draw_mu(j)
            self._update_tps_and_fitted_sums_incremental(j)

    def _draw_latent_z(self):
        pos = self.y_obs == 1
        neg = ~pos

        a_pos = -self.fitted_sums[pos]
        b_pos = np.full(a_pos.shape, np.inf)
        self.y_work[pos] = truncnorm.rvs(a=a_pos,
                                b=b_pos,
                                loc=self.fitted_sums[pos],
                                scale=1.0,
                                random_state=self.rng,)

        a_neg = np.full(np.sum(neg), -np.inf)
        b_neg = -self.fitted_sums[neg]
        self.y_work[neg] = truncnorm.rvs(a=a_neg,
                                b=b_neg,
                                loc=self.fitted_sums[neg],
                                scale=1.0,
                                random_state=self.rng,)
        
    
    