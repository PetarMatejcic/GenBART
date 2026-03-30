import numpy as np
from scipy.stats import chi2
from genbart.tree import Node, Tree


class bart:
    m: int
    alpha: float
    beta: float
    k: float
    nu: float
    q: float
    n_burn: int
    n_samples: int
    move_distribution: list
    rng: int | None

    X: np.ndarray
    y: np.ndarray
    n: int
    p: int
    y_shift: float
    y_scale: float

    sigma: float
    sigma_mu: float
    lambda_: float

    trees: list[Tree]
    training_predicitons: np.ndarray
    fitted_sums: np.ndarray
    tree_sample: list[Tree]

    def __init__(self,
                 m=200,
                 alpha=0.95,
                 beta=2.0,
                 k=2.0,
                 nu=3.0,
                 q=0.90,
                 n_burn=200,
                 n_samples=1000,
                 move_distribution = [0.25, 0.4, 0.2, 0.15],
                 random_state=None):
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.nu = nu
        self.q = q
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.move_distribution = move_distribution
        self.rng = np.random.default_rng(seed = random_state)
    
    def _inverse_transform_y(self, v: np.ndarray):
        return (v * self.y_scale) + self.y_shift

    def fit(self, X, y):
        # Transforming and storing data.
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.y_scale = max(self.y) - min(self.y)
        if self.y_scale < 1e-12:
            self.y_scale = 1.0
        self.y_shift = (min(self.y)+max(self.y)) / 2.0
        self.y = (self.y - self.y_shift) / self.y_scale
        self.n, self.p = X.shape

        # Initializing trees.
        self.trees = [Tree(Node.terminal(0.0)) for _ in range(self.m)]

        # Initializing mu_ij|T_j prior ~ N(0, sigma_mu)
        self.sigma_mu = 0.5 / (self.k * np.sqrt(self.m))
        # Initializing sigma prior.
        A_sigma = np.column_stack([self.X, np.ones(self.n)])
        sigma_linalg_beta, _, sigma_linalg_rank, _ = np.linalg.lstsq(A_sigma, self.y)
        rss = np.sum((self.y - A_sigma @ sigma_linalg_beta)**2)
        self.sigma = np.sqrt(rss) / (self.n - self.p)
        self.lambda_ = (self.sigma / self.nu) * chi2.ppf(1 - self.q, df = self.nu)
        
        # Initializing predictions.
        self.training_predicitons = np.zeros((self.n, self.m))
        self.fitted_sums = np.zeros(self.n)