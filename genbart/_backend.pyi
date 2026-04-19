import numpy as np

class PackedForest:
    def __init__(
        self,
        variable: np.ndarray,
        value: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        mu: np.ndarray,
        tree_offset: np.ndarray,
        n_draws: int,
        m: int,
        p: int,
    ) -> None: ...

    def draw_sums_row(self, x: np.ndarray) -> np.ndarray: ...
    def draw_sums_matrix(self, X: np.ndarray) -> np.ndarray: ...
