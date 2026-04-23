# pyright: reportMissingModuleSource=false
import numpy as np
from scipy.stats import chi2
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
        self.residuals = self.y_work.copy()

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
    
    def _backfitting_sweep(self):
        if self.engine is None:
            raise RuntimeError("Backfitting engin is not initialized!")
        self.engine.backfitting_sweep(self.residuals,
                                      self.sigma2,
                                      self.sigma_mu2,
                                      self.alpha,
                                      self.beta,
                                      self.move_distribution,
                                      )
    
    def _serialize_forest(self):
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
