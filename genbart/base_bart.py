"""Shared infrastructure for Bayesian additive regression tree estimators.

This module defines the abstract implementation layer used by concrete BART
models, such as regression BART and probit BART. It owns the common MCMC
state, delegates tree updates to the compiled backfitting engine, and stores
posterior forest draws in a packed representation for efficient prediction.
"""

# pyright: reportMissingModuleSource=false
import numpy as np
from scipy.stats import chi2
from genbart._backend import PackedForest, _BackfittingEngine


class BaseBART:
    """Shared implementation base for GenBART estimators.

    ``BaseBART`` contains the MCMC infrastructure and posterior storage shared by
    :class:`genbart.reg_bart.RegBart` and
    :class:`genbart.probit_bart.ProbitBart`.

    It manages the common BART hyperparameters, compiled backfitting engine,
    posterior forest storage, and variable-importance summaries. Model-specific
    likelihoods, response handling, fitting, and prediction are implemented by
    the concrete estimator subclasses.

    ``BaseBART`` is primarily an implementation base and is not intended to be
    fitted or instantiated directly by users.

    Parameters
    ----------
    m : int, default=200
        Number of trees in the BART ensemble.

    alpha : float, default=0.95
        Base probability parameter in the tree-structure prior.

    beta : float, default=2.0
        Depth penalty in the tree-structure prior.

    k : float, default=2.0
        Shrinkage parameter used by subclasses to determine terminal-node prior
        variance.

    n_burn : int, default=200
        Number of initial MCMC iterations discarded as burn-in.

    n_samples : int, default=1000
        Number of posterior forest draws retained after burn-in.

    move_distribution : tuple of float, default=(0.25, 0.25, 0.40, 0.10)
        Proposal probabilities for ``(grow, prune, change, swap)`` tree moves.

    random_state : int, default=0
        Seed used for reproducible NumPy and backend sampling.

    Notes
    -----
    Users normally interact with :class:`genbart.reg_bart.RegBart` or
    :class:`genbart.probit_bart.ProbitBart`. The main user-facing methods inherited
    from this class are :meth:`get_params`, :meth:`variable_inclusion`, and
    :meth:`variable_inclusion_draws`.
    """
    m: int
    alpha: float
    beta: float
    k: float
    n_burn: int
    n_samples: int
    move_distribution: tuple
    rng: int | None
    random_state: int

    X: np.ndarray
    n: int
    p: int

    sigma2: float
    sigma_mu2: float
    y_work: np.ndarray

    residuals: np.ndarray
    packed_forest: PackedForest
    extreme_values: list[tuple]

    engine: _BackfittingEngine | None

    def __init__(self,
                 m=200,
                 alpha=0.95,
                 beta=2.0,
                 k=2.0,
                 n_burn=200,
                 n_samples=1000,
                 move_distribution=(0.25, 0.25, 0.40, 0.10),
                 random_state=0):
        """Initialize common BART hyperparameters and empty runtime state.

        Args:
            m: Number of trees in each forest draw.
            alpha: Base prior probability that a tree node splits.
            beta: Depth penalty controlling how quickly split probability decays.
            k: Shrinkage parameter used by subclasses when setting terminal-node
                prior variance.
            n_burn: Number of MCMC burn-in iterations to discard.
            n_samples: Number of posterior MCMC iterations to serialize and retain.
            move_distribution: Tuple of proposal probabilities for grow, prune,
                change, and swap moves, respectively.
            random_state: Seed used for reproducible NumPy and backend randomness.

        Notes:
            This initializer does not allocate training data, initialize trees, or set
            likelihood-specific quantities such as ``sigma2`` and ``sigma_mu2``. Those
            steps are performed by subclass ``fit`` methods.
        """
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.move_distribution = move_distribution
        self.rng = np.random.default_rng(seed=random_state)
        self.random_state = random_state

        self.engine = None

        self.packed_forest = None
        self._vi_sum = None
        self._vi_draws = None
        self._logml_vi_sum = None
        self._logml_vi_draws = None

    def get_params(self):
        """Return the estimator's constructor parameters.

        Returns
        -------
        dict
            Dictionary containing the common BART parameters:

            - ``"m"``: number of trees.
            - ``"alpha"``: base tree-splitting prior parameter.
            - ``"beta"``: tree-depth penalty.
            - ``"k"``: terminal-node shrinkage parameter.
            - ``"n_burn"``: number of burn-in iterations.
            - ``"n_samples"``: number of retained posterior draws.
            - ``"move_distribution"``: grow, prune, change, and swap proposal
              probabilities.
            - ``"random_state"``: random seed.

        Notes
        -----
        Concrete estimators may extend this dictionary with model-specific
        parameters. In particular, :class:`genbart.reg_bart.RegBart` also returns
        ``"nu"`` and ``"q"``.

        The returned dictionary contains model configuration rather than fitted
        state. Training data, posterior forests, and fitted variance quantities are
        not included.
        """
        return {
            "m": self.m,
            "alpha": self.alpha,
            "beta": self.beta,
            "k": self.k,
            "n_burn": self.n_burn,
            "n_samples": self.n_samples,
            "move_distribution": self.move_distribution,
            "random_state": self.random_state
        }
    
    def _initialize_fit_state(self):
        self._init_trees()
        self._init_common_arrays()
        self._init_packed_builder()
        self._init_variable_inclusion()


    def _init_trees(self):
        """Initialize feature ranges and the backend forest.

        Computes the observed minimum and maximum for each predictor, constructs the
        compiled backfitting engine, and initializes the live forest as ``m`` root-only
        trees.

        Raises:
            AttributeError: If ``X`` or ``p`` has not been set by the subclass before
                this method is called.
        """
        self.extreme_values = [(self.X[:, var].min(),
                               self.X[:, var].max())
                               for var in range(self.p)]
        self.engine = _BackfittingEngine(self.X, self.m, self.random_state)
        self.engine.initialize_root_forest()

    def _init_common_arrays(self):
        """Initialize shared mutable arrays used by the sampler.

        Copies the model-specific working response into ``residuals`` so the backend
        can maintain partial residuals during successive backfitting sweeps.

        Raises:
            AttributeError: If ``y_work`` has not been set by the subclass before this
                method is called.
        """
        self.residuals = self.y_work.copy()

    def _init_packed_builder(self):
        """Initialize temporary buffers for serialized posterior forest draws.

        The buffers collect node arrays and tree offsets across retained MCMC samples.
        They are later concatenated by ``_finalize_packed_forest`` into a
        ``PackedForest`` object used for vectorized posterior prediction.
        """
        self._packed_variable_chunks = []
        self._packed_value_chunks = []
        self._packed_left_chunks = []
        self._packed_right_chunks = []
        self._packed_mu_chunks = []
        self._packed_tree_offset = [0]
        self._packed_n_trees = 0

    def _init_variable_inclusion(self):
        self._vi_sum = np.zeros(self.p)
        self._vi_draws = np.zeros((self.n_samples, self.p))
        self._logml_vi_sum = np.zeros(self.p)
        self._logml_vi_draws = np.zeros((self.n_samples, self.p))

    def _store_posterior_sample(self, sample: int):
        variable, value, left, right, mu, tree_offset = self._serialize_forest()
        self._append_serialized_forest_block(variable,
                                                value,
                                                left,
                                                right,
                                                mu,
                                                tree_offset)
        self._store_variable_inclusion(sample)

    def _store_variable_inclusion(self, sample: int):
        raw_vi, logml_vi = self.engine.variable_inclusion(self.residuals,
                                                             self.sigma2,
                                                             self.sigma_mu2)
        raw_vi = np.asarray(raw_vi)
        logml_vi = np.asarray(logml_vi)

        if raw_vi.shape != (self.p, ):
            raise RuntimeError(
                f"raw_vi has shape {raw_vi.shape}, expected {(self.p, )}"
            )
        if logml_vi.shape != (self.p, ):
            raise RuntimeError(
                f"logml_vi has shape {logml_vi.shape}, expected {(self.p, )}"
            )
        
        self._vi_draws[sample] = raw_vi
        self._vi_sum += raw_vi
        self._logml_vi_draws[sample] = logml_vi
        self._logml_vi_sum += logml_vi


    def _finalize_packed_forest(self):
        """Build the immutable packed forest from serialized posterior draws.

        Concatenates the accumulated node and offset buffers, validates that the
        expected number of trees was retained, constructs ``PackedForest``, and deletes
        the temporary builder buffers.

        Raises:
            RuntimeError: If the number of serialized trees differs from
                ``n_samples * m``.
            ValueError: If any packed chunk list is empty or cannot be concatenated by
                NumPy.
        """
        expected_trees = self.n_samples * self.m
        if self._packed_n_trees != expected_trees:
            raise RuntimeError(
                f"Packed {self._packed_n_trees} trees, expected {expected_trees}."
            )

        variable = np.concatenate(self._packed_variable_chunks)
        value = np.concatenate(self._packed_value_chunks)
        left = np.concatenate(self._packed_left_chunks)
        right = np.concatenate(self._packed_right_chunks)
        mu = np.concatenate(self._packed_mu_chunks)
        tree_offset = np.asarray(self._packed_tree_offset, dtype=np.int64)

        self.packed_forest = PackedForest(
            variable, value, left, right, mu, tree_offset,
            self.n_samples, self.m, self.p
        )
        del self._packed_variable_chunks
        del self._packed_value_chunks
        del self._packed_left_chunks
        del self._packed_right_chunks
        del self._packed_mu_chunks
        del self._packed_tree_offset
        del self._packed_n_trees

    def variable_inclusion(self, kind: str = "raw"):
        """Return posterior mean variable-importance scores.

        Variable importance is computed separately for every retained posterior
        forest draw and then averaged across draws.

        Parameters
        ----------
        kind : {"raw", "logml"}, default="raw"
            Variable-importance definition.

            ``"raw"`` computes the standard variable inclusion proportion. Each
            split contributes one count to its splitting variable, and the counts are
            normalized by the total number of splits in that posterior forest.

            ``"logml"`` weights each split by its positive log marginal-likelihood
            gain. For an internal node, the gain compares the marginal likelihood of
            retaining the full subtree below that node with the marginal likelihood
            obtained by collapsing the node to a single leaf. Positive gains are
            accumulated for the splitting variable and normalized by the total
            positive gain in that posterior forest.

        Returns
        -------
        np.ndarray
            One-dimensional array of length ``p`` containing the posterior mean
            importance score for each predictor.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        ValueError
            If ``kind`` is not ``"raw"`` or ``"logml"``.

        Notes
        -----
        The normalization is performed separately for each retained posterior draw
        before the scores are averaged across draws.

        If a draw contains no splits, its raw importance vector is all zeros. If a
        draw contains no positive log marginal-likelihood gains, its logML importance
        vector is all zeros.

        Use :meth:`variable_inclusion_draws` to access the individual posterior-draw
        importance vectors.
        """
        if self._vi_sum is None:
            raise RuntimeError("Model not fitted.")

        if kind == "raw":
            return self._vi_sum / self.n_samples

        if kind == "logml":
            return self._logml_vi_sum / self.n_samples

        raise ValueError("kind must be 'raw' or 'logml'.")
    

    def variable_inclusion_draws(self, kind: str = "raw"):
        """Return draw-level variable-importance scores.

        Parameters
        ----------
        kind : {"raw", "logml"}, default="raw"
            Variable-importance definition. See :meth:`variable_inclusion` for the
            definitions of the raw and logML-weighted statistics.

        Returns
        -------
        np.ndarray
            Array with shape ``(n_samples, p)``.

            Each row contains the normalized variable-importance scores for one
            retained posterior forest draw, and each column corresponds to one
            predictor.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        ValueError
            If ``kind`` is not ``"raw"`` or ``"logml"``.

        Notes
        -----
        :meth:`variable_inclusion` returns the column-wise mean of these draw-level
        scores.
        """
        if self._vi_draws is None:
            raise RuntimeError("Model not fitted.")

        if kind == "raw":
            return self._vi_draws.copy()

        if kind == "logml":
            return self._logml_vi_draws.copy()

        raise ValueError("kind must be 'raw' or 'logml'.")


    def _backfitting_sweep(self):
        """Run one backend Bayesian backfitting sweep over the live forest.

        Delegates to the compiled engine to update each tree conditional on the current
        partial residuals, variance parameters, tree-structure prior, and proposal-move
        distribution.

        Raises:
            RuntimeError: If the backend engine has not been initialized.
        """
        if self.engine is None:
            raise RuntimeError("Backfitting engine is not initialized!")
        self.engine.backfitting_sweep(self.residuals,
                                      self.sigma2,
                                      self.sigma_mu2,
                                      self.alpha,
                                      self.beta,
                                      self.move_distribution,
                                      )

    def _serialize_forest(self):
        """Serialize the current live forest from the backend engine.

        Returns:
            A tuple ``(variable, value, left, right, mu, tree_offset)`` describing the
            current forest in packed-node form. The arrays encode split variables,
            split values, child indices, terminal-node means, and per-tree node offsets.

        Raises:
            RuntimeError: If the backend engine has not been initialized.
        """
        if self.engine is None:
            raise RuntimeError("Backfitting engine not initialized.")
        return self.engine.serialize_forest()

    def _append_serialized_forest_block(self,
                                        variable: np.ndarray,
                                        value: np.ndarray,
                                        left: np.ndarray,
                                        right: np.ndarray,
                                        mu: np.ndarray,
                                        tree_offset: np.ndarray,
                                        ):
        """Append one serialized live forest to the packed posterior buffers.

        The input arrays describe a single MCMC draw containing ``m`` trees. This method
        validates shapes and offsets, converts arrays to packed dtypes, shifts child
        indices by the current global node offset, and appends the adjusted arrays to
        the temporary packed-forest builder.

        Args:
            variable: One-dimensional array of split-variable indices, with negative
                values indicating terminal nodes.
            value: One-dimensional array of split thresholds or placeholder values.
            left: One-dimensional array of left-child node indices local to this forest,
                with negative values for absent children.
            right: One-dimensional array of right-child node indices local to this
                forest, with negative values for absent children.
            mu: One-dimensional array of terminal-node parameters or node values.
            tree_offset: One-dimensional array of length ``m + 1`` whose consecutive
                entries delimit each tree's nodes within the serialized arrays.

        Raises:
            RuntimeError: If any serialized array is not one-dimensional, node-array
                lengths do not match, ``tree_offset`` has an invalid length, offsets do
                not start at zero, the final offset does not match the node-array
                length, or any tree contains no nodes.
        """
        variable = np.asarray(variable, dtype=np.int32)
        value = np.asarray(value, dtype=np.float64)
        left = np.asarray(left, dtype=np.int32)
        right = np.asarray(right, dtype=np.int32)
        mu = np.asarray(mu, dtype=np.float64)
        tree_offset = np.asarray(tree_offset, dtype=np.int64)

        n = variable.shape[0]

        if (
            variable.ndim != 1
            or value.ndim != 1
            or left.ndim != 1
            or right.ndim != 1
            or mu.ndim != 1
            or tree_offset.ndim != 1
        ):
            raise RuntimeError("Serialized forest arrays must all be 1D.")

        if not (
            value.shape[0] == n
            and left.shape[0] == n
            and right.shape[0] == n
            and mu.shape[0] == n
        ):
            raise RuntimeError("Serialized node arrays must have equal lengths.")

        if tree_offset.shape[0] != self.m + 1:
            raise RuntimeError("tree_offset must have length m + 1 for one live forest.")

        if tree_offset[0] != 0:
            raise RuntimeError("tree_offset[0] must be 0.")

        if tree_offset[-1] != n:
            raise RuntimeError("tree_offset[-1] must equal node-array length.")

        if np.any(tree_offset[1:] <= tree_offset[:-1]):
            raise RuntimeError("Each live tree must contain at least one node.")

        base = self._packed_tree_offset[-1]

        self._packed_variable_chunks.append(variable)
        self._packed_value_chunks.append(value)
        self._packed_mu_chunks.append(mu)

        self._packed_left_chunks.append(
            np.where(left >= 0, left + base, -1).astype(np.int32, copy=False)
        )
        self._packed_right_chunks.append(
            np.where(right >= 0, right + base, -1).astype(np.int32, copy=False)
        )

        self._packed_tree_offset.extend((tree_offset[1:] + base).tolist())
        self._packed_n_trees += self.m
