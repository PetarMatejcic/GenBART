import numpy as np
from scipy.stats import chi2
from genbart.baseBART import BaseBART


class RegBart(BaseBART):
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

    def fit(self, X, y):
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
        self._init_trees()
        self._init_common_arrays()
        self._init_packed_builder()
        self._vi_sum = np.zeros(self.p)

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
        self.lambda_ = (self.sigma2 / self.nu) * chi2.ppf(1-self.q, df=self.nu)

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

    def predict(self, X,
                central_measure="mean",
                conf_int=True,
                level: float = 0.90):
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

    def marginalize(self,
                    variable: int,
                    grid,
                    sampling_size: int = 100,
                    level=0.9):
        lows = [ev[0] for ev in self.extreme_values]
        highs = [ev[1] for ev in self.extreme_values]

        sample = np.random.uniform(low=lows,
                                   high=highs,
                                   size=(sampling_size, self.p))

        prediction = np.empty(grid.shape[0])
        conf_int_low = np.empty(grid.shape[0])
        conf_int_high = np.empty(grid.shape[0])
        a = 1 - level

        for i in range(grid.shape[0]):
            sample[:, variable] = np.full(sampling_size, grid[i])

            values = self.predict(sample, conf_int=False)
            conf_int = np.quantile(values["prediction"], [a/2.0, 1 - a/2.0])
            prediction[i] = values["prediction"].mean()
            conf_int_low[i] = conf_int[0]
            conf_int_high[i] = conf_int[1]
        out = {}
        out["prediction"] = prediction
        out["conf_int_low"] = conf_int_low
        out["conf_int_high"] = conf_int_high
        return out

    def _one_mcmc_iteration(self):
        self._backfitting_sweep()
        self.sigma2 = self._draw_sigma()

    def _draw_sigma(self):
        sse = np.sum((self.residuals)**2)
        return 1.0 / self.rng.gamma(shape=0.5*(self.nu+self.n),
                                    scale=2.0/(self.nu*self.lambda_ + sse))

    def _inverse_transform_y(self, v: np.ndarray):
        return (v * self.y_scale) + self.y_shift
