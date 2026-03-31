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
                 move_distribution=[0.25, 0.25, 0.40, 0.10],
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
        self.rng = np.random.default_rng(seed=random_state)

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
        sigma_linalg = np.linalg.lstsq(A_sigma, self.y, rcond=None)
        rss = np.sum((self.y - A_sigma @ sigma_linalg[0])**2)
        self.sigma = np.sqrt(rss) / (self.n - self.p)
        self.lambda_ = (self.sigma / self.nu) * chi2.ppf(1-self.q, df=self.nu)

        # Initializing predictions.
        self.training_predicitons = np.zeros((self.n, self.m))
        self.fitted_sums = np.zeros(self.n)
    
    def _one_mcmc_iteration(self):
        for j in range(self.m):
            self._draw_tree(j)
        self._draw_sigma()
    
    def _draw_tree(self, j: int):
        move = self.rng.choice(["grow", "prune", "change", "swap"],
                               p=self.move_distribution)
        proposed_tree, transition_ratio = self._propose_tree_move(j, move)
        pass

    def _propose_tree_move(self, j: int, move: str):
        if move == "grow":
            possible_paths = self.trees[j].terminal_paths()
            relevant_data = []

            for path in possible_paths:
                path_data = []
                for r in range(self.n):
                    if self.trees[j].is_in_path(path, self.X[r, :]):
                        path_data.append(r)
                relevant_data.append(path_data)
            data_mask = [len(path_data) > 1 for path_data in relevant_data]
            possible_paths = [path for path, m in zip(possible_paths, data_mask) if m]
            relevant_data = [data for data, m in zip(relevant_data, data_mask) if m]
            
            value_counts = []
            for path_data in relevant_data:
                value_count_dict = {}
                for var in range(self.p):
                    value_count_dict[var] = np.unique(self.X[path_data, var])
                value_counts.append(value_count_dict)
            data_mask = [any(len(v) > 1 for v in value_dict.values()) for value_dict in value_counts]
            possible_paths = [path for path, m in zip(possible_paths, data_mask) if m]
            relevant_data = [data for data, m in zip(relevant_data, data_mask) if m]
            value_counts = [v for v, m in zip(value_counts, data_mask) if m]

            path_index = self.rng.choice(len(possible_paths))
            print(type(value_counts[path_index]))
            possible_variables = [k for k, v in value_counts[path_index].items() if len(v) > 1]
            variable = self.rng.choice(possible_variables)
            possible_values = value_counts[path_index][variable]
            value = self.rng.choice(possible_values[:-1])

            proposed_tree = self.trees[j].grow(path=possible_paths[path_index],
                                               variable=variable,
                                               value=value)
            transition_ratio = (self.move_distribution[0]
                                * (1.0 / len(possible_paths))
                                * (1.0 / len(possible_variables))
                                * (1.0 / len(possible_values[:-1])))
        else:
            proposed_tree = 0
            transition_ratio = 0
        return proposed_tree, transition_ratio

    def _draw_sigma(self):
        pass

    def _inverse_transform_y(self, v: np.ndarray):
        return (v * self.y_scale) + self.y_shift
