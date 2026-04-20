# GenBART

GenBART is a Python/C++ implementation of **Bayesian Additive Regression Trees (BART)**.

This project started as part of my master's thesis, with the goal of implementing BART from scratch and understanding the algorithm in detail. At the same time, it is being developed into a general-purpose package that can be extended with additional BART variants over time.

## What is BART?

BART is a Bayesian sum-of-trees model for regression and classification. It models a response variable as a **sum of many small regression trees**. The individual trees are regularized to act as weak learners, and fitting is done with a **Bayesian backfitting MCMC algorithm**. This makes BART a flexible Bayesian nonparametric method, with built-in uncertainty quantification.

## Current scope

GenBART currently includes the core code needed for BART-style models.

The package currently includes:

- regression BART
- probit / classification BART
- tree moves: grow, prune, change, and swap
- a Bayesian backfitting engine implemented in C++
- posterior prediction from stored MCMC draws
- variable-importance summaries
- marginalization utilities

Additional variants and extensions will be added later.

## Structure

The codebase is split between Python and C++.

### Python

- `genbart/baseBART.py` — shared BART logic
- `genbart/reg_bart.py` — regression BART interface
- `genbart/probit_bart.py` — classification BART interface

### C++

- `tree.cpp` — tree operations
- `backfitting_engine.cpp` — backfitting engine
- `packed_forest.cpp` — compact storage of trees and fast posterior prediction

Python handles the user-facing interface and high-level workflow, while C++ handles the more computationally intensive tree and MCMC operations.

## Minimal example

```python
import numpy as np
from genbart.reg_bart import RegBart

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, size=(200, 2))
y = np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.1, size=200)

model = RegBart(
    m=200,
    n_burn=200,
    n_samples=1000,
    random_state=0,
)

model.fit(X, y)

pred = model.predict(X)
print(pred["prediction"][:5])

vi = model.variable_importance()
print(vi)

grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
marg = model.marginalize(variable=0, grid=grid)
```

## Reference

Chipman, H. A., George, E. I., and McCulloch, R. E. (2010).  
**BART: Bayesian Additive Regression Trees.**  
*The Annals of Applied Statistics*, 4(1), 266-298.

## License

