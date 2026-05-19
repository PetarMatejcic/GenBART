import numpy as np
from dataclasses import dataclass, field

__all__ = [
    "ThresholdResult",
    "VariableSelectionResult",
    "BartVariableSelector" 
]

@dataclass
class ThresholdResult:
    """Selection output for a single thresholding method.

    Stores the variable inclusion proportions, threshold values, and selected
    feature mask for one variable-selection rule such as local, global-max, or
    global-SE thresholding.
    """
    method: str
    feature_names: list
    real_vips: np.ndarray
    thresholds: np.ndarray
    selected: np.ndarray
    quantile: float
    extra: dict = field(default_factory=dict)

    def selected_features(self):
        """Return the names of features selected by this thresholding method."""
        return [self.feature_names[j]
                for j in range(len(self.feature_names))
                if self.selected[j]]
    
    def selected_indices(self):
        """Return the integer indices of features selected by this thresholding method."""
        return np.flatnonzero(self.selected)
    
    def n_selected(self):
        """Return the number of selected features."""
        return int(np.sum(self.selected))
    
    def threshold_for(self, feature):
        """Return the selection threshold for one feature.

        Parameters
        ----------
        feature : str or int
            Feature name or zero-based feature index.

        Returns
        -------
        float
            Threshold assigned to the requested feature.

        Raises
        ------
        ValueError
            If the feature name is unknown or the index is out of range.
        """
        j = self._feature_index(feature)
        return self.thresholds[j]
    
    def vip_for(self, feature):
        """Return the observed variable inclusion proportion for one feature.

        Parameters
        ----------
        feature : str or int
            Feature name or zero-based feature index.

        Returns
        -------
        float
            Observed variable inclusion proportion for the requested feature.

        Raises
        ------
        ValueError
            If the feature name is unknown or the index is out of range.
        """
        j = self._feature_index(feature)
        return self.real_vips[j]
    
    def _feature_index(self, feature):
        """Resolve a feature name or index to a zero-based feature index.

        Parameters
        ----------
        feature : str or int
            Feature name or zero-based feature index.

        Returns
        -------
        int
            Zero-based feature index.

        Raises
        ------
        ValueError
            If the feature name is unknown or the index is out of range.
        """
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
    """Result object returned by BART permutation-based variable selection.

    Stores observed variable inclusion proportions, permutation-null inclusion
    proportions, and thresholding results for all implemented selection methods.
    The default method is used when method-specific query functions are called
    without an explicit method argument.
    """
    feature_names: list
    real_vips: np.ndarray
    real_vips_repeats: np.ndarray
    real_vips_sd: np.ndarray
    null_vips: np.ndarray
    methods: dict
    default_method: str = "global_se"

    @property
    def null_mean(self):
        """Return the variable-wise mean of the permutation-null inclusion proportions."""
        return self.null_vips.mean(axis=0)
    
    @property
    def null_sd(self):
        """Return the variable-wise standard deviation of the permutation-null inclusion proportions."""
        if self.null_vips.shape[0] > 1:
            return self.null_vips.std(axis=0, ddof=1)
        else:
            return np.zeros(self.null_vips.shape[1])
        
    def selected_features(self, method=None):
        """Return selected feature names for a thresholding method.

        Parameters
        ----------
        method : str, optional
            Selection method to query. If None, the result's default method is used.

        Returns
        -------
        list
            Names of selected features.
        """
        method_result = self._method_result(method)
        return method_result.selected_features()
    
    def selected_indices(self, method=None):
        """Return selected feature indices for a thresholding method.

        Parameters
        ----------
        method : str, optional
            Selection method to query. If None, the result's default method is used.

        Returns
        -------
        np.ndarray
            Integer indices of selected features.
        """
        method_result = self._method_result(method)
        return method_result.selected_indices()
    
    def selected_mask(self, method=None):
        """Return a Boolean selected-feature mask for a thresholding method.

        Parameters
        ----------
        method : str, optional
            Selection method to query. If None, the result's default method is used.

        Returns
        -------
        np.ndarray
            Boolean array with one entry per feature.
        """
        method_result = self._method_result(method)
        return method_result.selected.copy()
    
    def thresholds(self, method=None):
        """Return selection thresholds for a thresholding method.

        Parameters
        ----------
        method : str, optional
            Selection method to query. If None, the result's default method is used.

        Returns
        -------
        np.ndarray
            Threshold array with one entry per feature.
        """
        method_result = self._method_result(method)
        return method_result.thresholds.copy()
    
    def ranking(self):
        """Return feature indices sorted by decreasing observed inclusion proportion."""
        return np.argsort(-self.real_vips)
    
    def to_frame(self, method=None):
        """Return a row-wise summary of variable-selection results.

        Parameters
        ----------
        method : str, optional
            Selection method to summarize. If None, the result's default method is used.

        Returns
        -------
        list of dict
            Rows sorted by decreasing observed inclusion proportion. Each row contains
            the feature rank, feature name, observed inclusion proportion, null summary,
            threshold, selected flag, and method name.
        """
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
        """Return a compact summary for a thresholding method.

        Parameters
        ----------
        method : str, optional
            Selection method to summarize. If None, the result's default method is used.

        Returns
        -------
        dict
            Summary containing the method name, number of features, number of selected
            features, selected feature names, top-ranked feature, and top inclusion
            proportion.
        """
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
        """Return selected-feature summaries for all thresholding methods.

        Returns
        -------
        list of dict
            One row per thresholding method, containing the method name, number of
            selected features, and selected feature names.
        """
        rows = []

        for method, method_result in self.methods.items():
            rows.append({
                "method": method,
                "n_selected": method_result.n_selected(),
                "selected_features": method_result.selected_features(),
            })

        return rows
    
    def _method_result(self, method=None):
        """Return the ThresholdResult for a requested method.

        Parameters
        ----------
        method : str, optional
            Selection method to retrieve. If None, the default method is used.

        Returns
        -------
        ThresholdResult
            Threshold result for the requested method.

        Raises
        ------
        ValueError
            If the method is unknown.
        """
        method = self.default_method if method is None else method
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        return self.methods[method]


class BartVariableSelector:
    """Permutation-based variable selector for BART models.

    Fits a BART model to the observed response, estimates variable inclusion
    proportions, then repeatedly permutes the response to construct a null
    distribution of inclusion proportions. The selector returns a
    VariableSelectionResult containing local, global-max, and global-SE
    thresholding results.

    The model class must be callable and must create objects with fit(X, y) and
    variable_inclusion() methods. The selector recreates fresh model instances for
    each observed-response and permuted-response fit.
    """
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
        """Initialize the BART variable selector.

        Parameters
        ----------
        model_cls : callable
            Model class used for repeated BART fits. Instances must implement fit(X, y)
            and variable_inclusion().
        model_params : dict, optional
            Constructor parameters passed to model_cls. The selector overrides
            random_state when creating repeated model instances.
        n_permutations : int, default=20
            Number of response permutations used to estimate the null distribution.
        n_repeats : int, default=5
            Number of repeated model fits used to average observed and permuted
            inclusion proportions.
        alpha : float, default=0.05
            Tail probability used for selection thresholds. Thresholds use the
            1 - alpha permutation quantile.
        method : {"local", "global_max", "global_se"}, default="global_se"
            Default thresholding method used by the returned result object.
        random_state : int, optional
            Seed controlling model-fit seeds and response permutations.
        n_jobs : int, default=1
            Reserved for future parallel execution.
        verbose : bool, default=False
            Whether to print progress messages during fitting.

        Raises
        ------
        TypeError
            If model_cls is not callable.
        ValueError
            If n_permutations, n_repeats, alpha, or method is invalid.
        """
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
        """Create a selector from an existing unfitted model instance.

        The model must implement get_params(), which is used to recover constructor
        parameters. The selector uses type(model) as the model class and recreates
        fresh model instances during fitting.

        Parameters
        ----------
        model : object
            Model instance implementing get_params().
        n_permutations : int, default=20
            Number of response permutations used to estimate the null distribution.
        n_repeats : int, default=5
            Number of repeated model fits used to average inclusion proportions.
        alpha : float, default=0.05
            Tail probability used for selection thresholds.
        method : {"local", "global_max", "global_se"}, default="global_se"
            Default thresholding method.
        random_state : int, optional
            Seed controlling model-fit seeds and response permutations.
        n_jobs : int, default=1
            Reserved for future parallel execution.
        verbose : bool, default=False
            Whether to print progress messages during fitting.

        Returns
        -------
        BartVariableSelector
            Selector configured with the model class and constructor parameters.

        Raises
        ------
        TypeError
            If model is a class rather than an instance, does not implement
            get_params(), or get_params() does not return a dictionary.
        """
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

    def fit(self, X, y, feature_names = None):
        """Fit the variable-selection procedure.

        Fits repeated BART models on the observed response, fits repeated BART models
        on permuted responses, constructs local, global-max, and global-SE thresholds,
        and returns a VariableSelectionResult.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Response vector.
        feature_names : sequence of str, optional
            Names to use for the features. If omitted and X has columns, those column
            names are used. Otherwise names are generated as x0, x1, ...

        Returns
        -------
        VariableSelectionResult
            Variable-selection result containing observed inclusion proportions,
            permutation-null inclusion proportions, thresholds, and selected features.

        Raises
        ------
        ValueError
            If X and y have incompatible shapes or feature_names has the wrong length.
        """
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
        """Build the VariableSelectionResult from fitted inclusion arrays.

        Returns
        -------
        VariableSelectionResult
            Result object containing all thresholding methods.
        """
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
        """Build the local-threshold selection result.

        Each feature is compared against its own permutation-null quantile.

        Returns
        -------
        ThresholdResult
            Local thresholding result.
        """
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
        """Build the global-max threshold selection result.

        Each permutation contributes the maximum null inclusion proportion across all
        features. The global threshold is the 1 - alpha quantile of these maxima and is
        applied uniformly to all features.

        Returns
        -------
        ThresholdResult
            Global-max thresholding result.
        """
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
        """Build the global-SE threshold selection result.

        Each feature's null inclusion proportions are standardized by their permutation
        mean and standard deviation. A global cutoff is computed from the maximum
        standardized null statistic across features, then mapped back to feature-specific
        thresholds.

        Returns
        -------
        ThresholdResult
            Global-SE thresholding result.
        """
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
        """Fit repeated BART models on the observed response.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Validated feature matrix.
        y : np.ndarray of shape (n_samples,)
            Validated response vector.
        seeds : array-like of int
            Random seeds used to create repeated model instances.

        Side Effects
        ------------
        Sets real_vips_repeats_, real_vips_, and real_vips_sd_.
        """
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
        """Fit repeated BART models on permuted responses.

        For each response permutation, fits n_repeats model instances and averages their
        variable inclusion proportions to form one row of the null inclusion matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Validated feature matrix.
        y : np.ndarray of shape (n_samples,)
            Validated response vector.
        model_seeds : np.ndarray of shape (n_permutations, n_repeats)
            Random seeds used to create model instances for permuted-response fits.
        shuffle_seeds : array-like of int
            Random seeds used to permute the response.

        Side Effects
        ------------
        Sets null_vips_, null_vips_mean_, and null_vips_std_.
        """
        p = X.shape[1]
        null_vips = np.empty((self.n_permutations, p), dtype=float)

        for b, shuffle_seed in enumerate(shuffle_seeds):
            if self.verbose:
                print(f"Fitting permuted-response BART {b + 1}/{self.n_permutations}")
            
            y_perm = self._permute_response(y, shuffle_seed)
            perm_vips = np.empty((self.n_repeats, p), dtype=float)
            
            for r, model_seed in enumerate(model_seeds[b]):
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
        """Return a reproducible random permutation of the response.

        Parameters
        ----------
        y : np.ndarray
            Response vector to permute.
        seed : int
            Seed used for the permutation.

        Returns
        -------
        np.ndarray
            Permuted response vector.
        """
        rng = np.random.default_rng(seed)
        return rng.permutation(y)

    def _is_fitted(self):
        """Raise an error if the selector has not been fitted.

        Raises
        ------
        RuntimeError
            If fit() has not completed successfully.
        """
        if not hasattr(self, "result_"):
            raise RuntimeError("BartVariableSelection is not fitted.")
        
    def _make_model(self, seed):
        """Create a fresh model instance with a controlled random seed.

        Parameters
        ----------
        seed : int
            Random seed assigned to the model instance.

        Returns
        -------
        object
            New instance of model_cls initialized with model_params and random_state=seed.
        """
        params = dict(self.model_params)
        params["random_state"] = seed
        return self.model_cls(**params)
    
    def _make_seeds(self):
        """Generate reproducible seeds for observed and permuted model fits.

        Returns
        -------
        tuple
            Three arrays: observed-response model seeds, permuted-response model seeds,
            and response-permutation seeds.
        """
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
            size=(self.n_permutations, self.n_repeats),
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
        """Resolve feature names for the input matrix.

        Parameters
        ----------
        X : array-like
            Original feature matrix, possibly a pandas DataFrame.
        feature_names : sequence of str, optional
            Explicit feature names supplied by the user.
        p : int
            Number of features.

        Returns
        -------
        list of str
            Feature names of length p.

        Raises
        ------
        ValueError
            If feature_names does not have length p.
        """
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
        """Validate and coerce feature and response arrays.

        Parameters
        ----------
        X : array-like
            Feature matrix. A 1D array is reshaped to a single-column matrix.
        y : array-like
            Response vector.

        Returns
        -------
        tuple of np.ndarray
            Validated feature matrix and response vector.

        Raises
        ------
        ValueError
            If X is not 1D or 2D, y is not 1D, or X and y have different numbers of rows.
        """
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