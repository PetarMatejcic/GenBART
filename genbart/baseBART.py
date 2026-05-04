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
    """Base class for BART estimators with shared MCMC and forest storage logic.

    The class stores common hyperparameters, initializes the backend forest,
    runs backend backfitting sweeps, serializes posterior tree draws, and exposes
    variable-importance summaries. Subclasses are responsible for defining the
    model-specific likelihood, fitting routine, response transformation, and
    prediction API.

    Attributes:
        m: Number of trees in each live forest.
        alpha: Prior probability scaling parameter for node splitting.
        beta: Prior depth penalty parameter for node splitting.
        k: Shrinkage parameter used by subclasses to set the terminal-node prior.
        n_burn: Number of burn-in MCMC iterations.
        n_samples: Number of posterior forest draws to retain.
        move_distribution: Probabilities for grow, prune, change, and swap moves.
        rng: NumPy random generator seeded from ``random_state``.
        random_state: Seed passed to NumPy and the backend backfitting engine.
        X: Training feature matrix.
        n: Number of training observations.
        p: Number of predictors.
        sigma2: Observation variance used during backfitting.
        sigma_mu2: Terminal-node prior variance used during backfitting.
        y_work: Working response variable used by the model-specific sampler.
        residuals: Current partial residual vector updated during backfitting.
        packed_forest: Packed posterior forest draws used for prediction.
        extreme_values: Per-feature minimum and maximum values from training data.
        engine: Backend object that owns and updates the live forest.
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

    def variable_importance(self):
        """Return posterior average variable-usage frequencies.

        Variable importance is computed as the average, across retained posterior
        forest draws, of each predictor's share of internal splitting rules.

        Returns:
            A one-dimensional NumPy array of length ``p``. Entry ``j`` is the posterior
            mean fraction of splitting rules that used predictor ``j``.

        Notes:
            Subclasses are expected to accumulate ``_vi_sum`` during fitting. This
            method should be called only after a successful fit.
        """
        return self._vi_sum / self.n_samples

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
