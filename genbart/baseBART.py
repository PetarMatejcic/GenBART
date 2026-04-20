# pyright: reportMissingModuleSource=false
import numpy as np
from scipy.stats import chi2
from .tree import Node, Tree, SerializedTree
from genbart._backend import PackedForest, _BackfittingEngine


class BaseBART:
    m: int
    alpha: float
    beta: float
    k: float
    n_burn: int
    n_samples: int
    move_distribution: tuple
    rng: int | None

    X: np.ndarray
    n: int
    p: int

    sigma2: float
    sigma_mu2: float
    y_work: np.ndarray

    training_predictions: np.ndarray
    residuals: np.ndarray
    fitted_sums: np.ndarray
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
                 random_state=None):
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.move_distribution = move_distribution
        self.rng = np.random.default_rng(seed=random_state)

        self.engine = None

        self.packed_forest = None
        self._vi_sum = None

    def _init_trees(self):
        rows_by_var = [np.argsort(self.X[:, var], kind="mergesort") for var in range(self.p)]
        self.extreme_values = [(self.X[x[0], var], self.X[x[-1], var]) for var, x in enumerate(rows_by_var)]
        seed = int(self.rng.integers(0, 2**63 - 1))
        self.engine = _BackfittingEngine(self.X, self.m, seed)
        self.engine.initialize_root_forest()

    def _init_common_arrays(self):
        self.training_predictions = np.zeros((self.m, self.n))
        self.fitted_sums = np.zeros(self.n)
        self.residuals = np.zeros(self.n)

    def _init_packed_builder(self):
        self._packed_variable_chunks = []
        self._packed_value_chunks = []
        self._packed_left_chunks = []
        self._packed_right_chunks = []
        self._packed_mu_chunks = []
        self._packed_tree_offset = [0]
        self._packed_n_trees = 0
        
    def _finalize_packed_forest(self):
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
        return self._vi_sum / self.n_samples
    
    def _draw_tree(self, j: int):
        if self.engine is None:
            raise RuntimeError("Backfitting engine not initialized.")
        return self.engine.draw_tree(j,
                                     self.residuals,
                                     self.sigma2,
                                     self.sigma_mu2,
                                     self.alpha,
                                     self.beta,
                                     self.move_distribution)

    def _draw_mu(self, j: int):
        if self.engine is None:
            raise RuntimeError("Backfitting engine not initialized.")
        self.engine.draw_mu(j,
                            self.residuals,
                            self.sigma2,
                            self.sigma_mu2)
    
    def _append_serialized_tree(self, tree):
        n = tree.variable.shape[0]

        if (
            tree.variable.ndim != 1
            or tree.value.ndim != 1
            or tree.left.ndim != 1
            or tree.right.ndim != 1
            or tree.mu.ndim != 1
        ):
            raise RuntimeError("Serialized tree arrays must all be 1D.")
        if not (
            tree.value.shape[0] == n
            and tree.left.shape[0] == n
            and tree.right.shape[0] == n
            and tree.mu.shape[0] == n
        ):
            raise RuntimeError("Serialized tree arrays must have equal lengths.")
        if n <= 0:
            raise RuntimeError("Serialized tree must contain at least one node.")

        base = self._packed_tree_offset[-1]

        self._packed_variable_chunks.append(tree.variable.astype(np.int32, copy=False))
        self._packed_value_chunks.append(tree.value.astype(np.float32, copy=False))
        self._packed_mu_chunks.append(tree.mu.astype(np.float32, copy=False))

        self._packed_left_chunks.append(
            np.where(tree.left >= 0, tree.left + base, -1).astype(np.int32, copy=False)
        )
        self._packed_right_chunks.append(
            np.where(tree.right >= 0, tree.right + base, -1).astype(np.int32, copy=False)
        )

        self._packed_tree_offset.append(base + n)
        self._packed_n_trees += 1

    def _serialize_tree(self, j: int):
        if self.engine is None:
            raise RuntimeError("Backfitting engine not initialized.")
        variable, value, left, right, mu = self.engine.serialize_tree(j)
        return SerializedTree(variable=np.asarray(variable, dtype=np.int32),
                              value=np.asarray(value, dtype=np.float32),
                              left=np.asarray(left, dtype=np.int32),
                              right=np.asarray(right, dtype=np.int32),
                              mu=np.asarray(mu, dtype=np.float32))
    
    def _partial_residuals(self, j):
        self.residuals = self.y_work - self.fitted_sums + self.training_predictions[j, :]

    def _update_tps_and_fitted_sums_incremental(self, j: int):
        if self.engine is None:
            raise RuntimeError("Backfitting engine not initialized.")
        self.engine.refresh_tree_training_predictions(j,
                                                      self.training_predictions,
                                                      self.fitted_sums)
