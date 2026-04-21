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
            
        for _ in range(self.n_samples):
            self._one_mcmc_iteration()
            
            variable, value, left, right, mu, tree_offset = self._serialize_forest()
            self._append_serialized_forest_block(variable, value, left, right, mu, tree_offset)

            mask = variable >= 0
            if np.any(mask):
                variable_counts = np.bincount(variable[mask], minlength=self.p)
                variable_total = int(mask.sum())
                self._vi_sum += variable_counts / variable_total
        self._finalize_packed_forest()
        return self
    
    def predict_probs(self, X, level: float = 0.90):
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
            conf_low, conf_high = np.quantile(probs, [a/2.0, 1 - a/2.0], axis=0)
            out["conf_int_low"] = conf_low
            out["conf_int_high"] = conf_high
            return out
    
    def predict(self, X, threshold: float = 0.5):
        probs = self.predict_probs(X)[0]
        return probs >= threshold
        
    def _one_mcmc_iteration(self):
        old_z = self.y_work.copy()
        self._draw_latent_z()
        self.residuals += self.y_work - old_z
        self._backfitting_sweep()

    def _draw_latent_z(self):
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
        
    
    