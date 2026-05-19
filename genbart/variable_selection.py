import numpy as np
from dataclasses import dataclass, field

@dataclass
class ThresholdResult:
    method: str
    feature_names: list
    real_vips: np.ndarray
    thresholds: np.ndarray
    selected: np.ndarray
    quantile: float
    extra: dict = field(default_factory=dict)

    def selected_features(self):
        return [self.feature_names[j]
                for j in range(len(self.feature_names))
                if self.selected[j]]
    
    def selected_indices(self):
        return np.flatnonzero(self.selected)
    
    def n_selected(self):
        return int(np.sum(self.selected))
    
    def threshold_for(self, feature):
        j = self._feature_index(feature)
        return self.thresholds[j]
    
    def vip_for(self, feature):
        j = self._feature_index(feature)
        return self.real_vips[j]
    
    def _feature_index(self, feature):
        if isinstance(feature, str):
            if feature not in self.feature_names:
                raise ValueError(f"Unknown feature: {feature}")
            return self.feature_names.index(feature)
        
        j = int(feature)
        if j < 0 or j >= len(self.feature_names):
            raise ValueError("feature index out of range.")
        return j

@dataclass
class VariableSelectionResult:
    feature_names: list
    real_vips: np.ndarray
    real_vips_repeats: np.ndarray
    real_vips_sd: np.ndarray
    null_vips: np.ndarray
    methods: dict
    default_method: str = "global_se"

    @property
    def null_mean(self):
        return self.null_vips.mean(axis=0)
    
    @property
    def null_sd(self):
        if self.null_vips.shape[0] > 1:
            return self.null_vips.std(axis=0, ddof=1)
        else:
            return np.zeros(self.null_vips.shape[1])
        
    def selected_features(self, method=None):
        method_result = self._method_result(method)
        return method_result.selected_features()
    
    def selected_indices(self, method=None):
        method_result = self._method_result(method)
        return method_result.selected_indices()
    
    def selected_mask(self, method=None):
        method_result = self._method_result(method)
        return method_result.selected.copy()
    
    def thresholds(self, method=None):
        method_result = self._method_result(method)
        return method_result.thresholds.copy()
    
    def ranking(self):
        return np.argsort(-self.real_vips)
    
    def to_frame(self, method=None):
        method_result = self._method_result(method)
        order = self.ranking()

        rows = []
        for rank, j in enumerate(order, start=1):
            rows.append({
                "rank": rank,
                "feature": self.feature_names[j],
                "vip": self.real_vips[j],
                "vip_sd": self.real_vips_sd[j],
                "null_mean": self.null_mean[j],
                "null_sd": self.null_sd[j],
                "threshold": method_result.thresholds[j],
                "selected": bool(method_result.selected[j]),
                "method": method_result.method,
            })

        return rows
    
    def summary(self, method=None):
        method_result = self._method_result(method)

        selected = method_result.selected_features()

        return {
            "method": method_result.method,
            "n_features": len(self.feature_names),
            "n_selected": method_result.n_selected(),
            "selected_features": selected,
            "top_feature": self.feature_names[int(np.argmax(self.real_vips))],
            "top_vip": float(np.max(self.real_vips)),
        }
    
    def compare_methods(self):
        rows = []

        for method, method_result in self.methods.items():
            rows.append({
                "method": method,
                "n_selected": method_result.n_selected(),
                "selected_features": method_result.selected_features(),
            })

        return rows
    
    def _method_result(self, method=None):
        method = self.default_method if method is None else method
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        return self.methods[method]

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

    real_vips_repeats_: np.ndarray
    real_vips_: np.ndarray
    real_vips_sd_: np.ndarray

    null_vips_: np.ndarray
    null_vips_mean_: np.ndarray
    null_vips_std_: np.ndarray

    result_: VariableSelectionResult

    def __init__(self,
                 model_cls,
                 model_params=None,
                 n_permutations=20,
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
        
        if n_permutations <= 0:
            raise ValueError("n_permutations must be positive.")
        self.n_permutations = n_permutations
        if n_repeats <= 0:
            raise ValueError("n_repeats must be positive.")
        self.n_repeats = n_repeats

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha

        if method not in ["local", "global_max", "global_se"]:
            raise ValueError("method must be 'local', 'global_max' or 'global_se'.")
        self.method = method

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    @classmethod
    def from_model(
        cls,
        model,
        *,
        n_permutations = 20,
        n_repeats = 5,
        alpha = 0.05,
        method = "global_se",
        random_state = None,
        n_jobs = 1,
        verbose = False
    ):
        if isinstance(model, type):
            raise TypeError(
                "from_model expects model instance, not a model class."
                "Use the  constructor with model_cls=... for classes."
            )
        
        if not hasattr(model, "get_params") or not callable(model.get_params):
            raise TypeError("model must implement get_params() so it can be recreated.")
        
        model_params = model.get_params()

        if not isinstance(model_params, dict):
            raise TypeError("model.get_params() must return a dictionary.")
        
        return cls(
             model_cls=type(model),
            model_params=model_params,
            n_permutations=n_permutations,
            n_repeats=n_repeats,
            alpha=alpha,
            method=method,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def fit(self,
            X,
            y,
            feature_names = None):
        data_X = X
        X, y = self._validate_xy(X, y)
        _, p = X.shape

        self.feature_names = self._get_feature_names(data_X, feature_names, p)
        
        real_model_seeds, perm_model_seeds, perm_shuffle_seeds = self._make_seeds()

        self._fit_real_response(X, y, real_model_seeds)

        self._fit_null_distribution(X, y, perm_model_seeds, perm_shuffle_seeds)

        self.result_ = self._build_result()
        
        return self.result_
    
    def _build_result(self):
        methods = {
            "local": self._local_result(),
            "global_max": self._global_max_result(),
            "global_se": self._global_se_result(),
        }

        result = VariableSelectionResult(
            feature_names=self.feature_names,
            real_vips=self.real_vips_,
            real_vips_repeats=self.real_vips_repeats_,
            real_vips_sd=self.real_vips_sd_,
            null_vips=self.null_vips_,
            methods=methods,
            default_method=self.method,
        )

        return result

    def _local_result(self):
        thresholds = np.quantile(self.null_vips_, 1 - self.alpha, axis=0)
        selected = self.real_vips_ > thresholds

        return ThresholdResult(
            method="local",
            feature_names=self.feature_names,
            real_vips=self.real_vips_,
            thresholds=thresholds,
            selected=selected,
            quantile=1-self.alpha,
            extra={},
        )
    
    def _global_max_result(self):
        max_null = np.max(self.null_vips_, axis=1)
        threshold = np.quantile(max_null, 1 - self.alpha)
        thresholds = np.full_like(self.real_vips_, threshold, dtype=float)
        selected = self.real_vips_ > thresholds

        return ThresholdResult(
            method="global_max",
            feature_names=self.feature_names,
            real_vips=self.real_vips_,
            thresholds=thresholds,
            selected=selected,
            quantile=1-self.alpha,
            extra={
                "global_threshold": threshold,
                "max_null": max_null,
            },
        )
        
    def _global_se_result(self):
        null_mean = self.null_vips_.mean(axis=0)

        if self.null_vips_.shape[0] > 1:
            null_sd = self.null_vips_.std(axis=0, ddof=1)
        else:
            null_sd = np.zeros(self.null_vips_.shape[1])

        safe_sd = null_sd.copy()
        safe_sd[safe_sd == 0.0] = 1.0

        standardized = (self.null_vips_ - null_mean) / safe_sd
        max_standardized = np.max(standardized, axis=1)

        global_se = np.quantile(max_standardized, 1 - self.alpha)
        thresholds = null_mean + global_se * null_sd
        selected = self.real_vips_ > thresholds

        return ThresholdResult(
            method="global_se",
            feature_names=self.feature_names,
            real_vips=self.real_vips_,
            thresholds=thresholds,
            selected=selected,
            quantile=1-self.alpha,
            extra={
                "global_se": global_se,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "max_standardized": max_standardized,
            },
        )
    
    def _fit_real_response(self, X, y, seeds):
        p = X.shape[1]
        real_vips_repeats = np.empty((self.n_repeats, p), dtype=float)

        for r, seed in enumerate(seeds):
            if self.verbose:
                print(f"Fitting real response BART repeat {r+1}/{self.n_repeats}")

            model = self._make_model(seed)
            model.fit(X, y)

            vip = model.variable_inclusion()
            real_vips_repeats[r, :] = vip
        
        self.real_vips_repeats_ = real_vips_repeats
        self.real_vips_ = real_vips_repeats.mean(axis=0)
        if self.n_repeats > 1:
            self.real_vips_sd_ = real_vips_repeats.std(axis=0, ddof=1)
        else:
            self.real_vips_sd_ = np.zeros(p)

    def _fit_null_distribution(self, X, y, model_seeds, shuffle_seeds):
        p = X.shape[1]
        null_vips = np.empty((self.n_permutations, p), dtype=float)

        for b, shuffle_seed in enumerate(shuffle_seeds):
            if self.verbose:
                print(f"Fitting permuted-response BART {b + 1}/{self.n_permutations}")
            
            y_perm = self._permute_response(y, shuffle_seed)
            perm_vips = np.empty((self.n_repeats, p), dtype=float)
            
            for r, model_seed in enumerate(model_seeds):
                model = self._make_model(int(model_seed))
                model.fit(X, y_perm)
                perm_vips[r, :] = model.variable_inclusion()

            null_vips[b, :] = perm_vips.mean(axis=0)
        
        self.null_vips_ = null_vips
        self.null_vips_mean_ = null_vips.mean(axis=0)
        if self.n_permutations > 1:
            self.null_vips_std_ = null_vips.std(axis=0, ddof=1)
        else:
            self.null_vips_std_ = np.zeros(p)

    def _permute_response(self, y: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        return rng.permutation(y)

    def _is_fitted(self):
        if not hasattr(self, "result_"):
            raise RuntimeError("BartVariableSelection is not fitted.")
        
    def _make_model(self, seed):
        params = dict(self.model_params)
        params["random_state"] = seed
        return self.model_cls(**params)
    
    def _make_seeds(self):
        rng = np.random.default_rng(self.random_state)

        real_model_seeds = rng.integers(
            low=0,
            high=np.iinfo(np.uint32).max,
            size=self.n_repeats,
            dtype=np.uint32,
        )

        perm_model_seeds = rng.integers(
            low=0,
            high=np.iinfo(np.uint32).max,
            size=self.n_repeats,
            dtype=np.uint32,
        )

        perm_shuffle_seeds = rng.integers(
            low=0,
            high=np.iinfo(np.uint32).max,
            size=self.n_permutations,
            dtype=np.uint32,
        )

        return real_model_seeds, perm_model_seeds, perm_shuffle_seeds
    
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