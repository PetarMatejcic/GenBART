import numpy as np

class BartVariableSelection:
    model_cls: None
    model_params: dict | None
    feature_names: list | None
    n_permutations: int
    n_repeats: int
    alpha: float
    method: str
    random_state: int
    n_jobs: int
    verbose: bool

    real_vips_repeats: np.ndarray
    real_vips: np.ndarray
    real_vips_sd: np.ndarray

    def __init__(self,
                 model_cls,
                 model_params=None,
                 n_permutations=100,
                 n_repeats=5,
                 alpha=0.05,
                 method="global_se",
                 random_state=None,
                 n_jobs=1,
                 verbose=False):
        if not callable(model_cls):
            raise TypeError("model_cls must be a callable class.")
        self.model_cls = model_cls
        self.model_params = dict(model_params or {})
        
        if n_repeats <= 0:
            raise ValueError("n_repeat must be positive.")
        self.n_repeats = n_repeats

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, feature_names = None):
        X, y = self._validate_xy(X, y)
        n, p = X.shape

        self.feature_names = self._get_feature_names(X, feature_names, p)
        seeds = self._make_seeds()

        real_vips_repeats = np.empty((self.n_repeats, p), dtype=float)

        for r, seed in enumerate(seeds):
            if self.verbose:
                print(f"Fitting real response BART repeat {r+1}/{self.n_repeats}")

            model = self._make_model(seed)
            model.fit(X, y)

            vip = model.variable_inclusion()
            real_vips_repeats[r, :] = vip
        
        self.real_vips_repeats = real_vips_repeats
        self.real_vips = real_vips_repeats.mean(axis=0)
        self.real_vips_sd = real_vips_repeats.std(axis=0, ddof=1) if self.n_repeats > 1 else np.zeros(p)

        return self

    def ranking(self):
        self._is_fitted()

        order = np.argsort(-self.real_vips)

        rows = []
        for rank, j in enumerate(order, start=1):
            rows.append({
                "rank": rank,
                "feature": self.feature_names[j],
                "vip": self.real_vips[j],
                "vip_sd": self.real_vips_sd[j]
            })

        return rows
    
    def selected_features(self, k=None):
        self._is_fitted()

        order = np.argsort(-self.real_vips)

        if k is None:
            return [self.feature_names[j] for j in order]
        
        if k <= 0:
            raise ValueError("k must be positive.")
        
        return [self.feature_names[j] for j in order[:k]]

    def _is_fitted(self):
        if not hasattr(self, "real_vips"):
            raise RuntimeError("BartVariableSeleciton is not fitted.")
        
    def _make_model(self, seed):
        params = dict(self.model_params)
        params["random_state"] = seed
        return self.model_cls(**params)
    
    def _make_seeds(self):
        rng = np.random.default_rng(self.random_state)
        return rng.integers(low=0,
                            high=np.iinfo(np.uint32).max,
                            size=self.n_repeats,
                            dtype=np.uint32)
    
    def _get_feature_names(self, X, feature_names, p):
        if feature_names is not None:
            names = list(feature_names)
        elif hasattr(X, "columns"):
            names = list(X.columns)
        else:
            names = [f"x{i}" for i in range(p)]

        if len(names) != p:
            raise ValueError("feature_names must have length equal to X.shape[1].")

        return names
    
    def _validate_xy(self, X, y):
        X_arr = np.asarray(X)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape((-1, 1))
        elif X_arr.ndim != 2:
            raise ValueError("X must be a 1D or 2D array-like object.")
        
        y_arr = np.asarray(y)

        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array-like object.")
        
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        
        return X_arr, y_arr