Model Selection
===============

.. currentmodule:: genbart.model_selection

GenBART provides lightweight grid-search cross-validation helpers for choosing
BART hyperparameters.

These functions are intended for straightforward parameter searches. They
evaluate every parameter combination using cross-validation and can optionally
refit the best configuration on the complete dataset.


Regression
----------

Use ``cross_validate_reg_bart`` for continuous-response models.

.. autofunction:: cross_validate_reg_bart


Example
~~~~~~~

.. code-block:: python

   from genbart.model_selection import cross_validate_reg_bart

   result = cross_validate_reg_bart(
       X,
       y,
       param_grid={
           "m": [50, 100],
           "k": [1.0, 2.0],
           "nu": [3.0],
           "q": [0.90],
           "n_burn": [200],
           "n_samples": [500],
       },
       cv=5,
       scoring="rmse",
       random_state=0,
   )

   result["best_params"]
   result["best_score"]

   best_model = result["best_model"]


Binary classification
---------------------

Use ``cross_validate_probit_bart`` for binary-response models.

.. autofunction:: cross_validate_probit_bart


Example
~~~~~~~

.. code-block:: python

   from genbart.model_selection import cross_validate_probit_bart

   result = cross_validate_probit_bart(
       X,
       y,
       param_grid={
           "m": [50, 100],
           "k": [1.0, 2.0],
           "n_burn": [200],
           "n_samples": [500],
       },
       cv=5,
       scoring="auc",
       random_state=0,
   )

   result["best_params"]
   result["best_score"]


Computational cost
------------------

Grid-search cross-validation requires one BART fit for every combination of
parameter configuration and validation fold.

If a grid contains :math:`G` parameter combinations and ``cv`` folds are used,
the search requires

.. math::

   G \times \mathrm{cv}

BART fits, plus one additional fit when ``refit=True``.

Because BART fitting itself uses MCMC, large grids with large ``n_burn`` and
``n_samples`` values can become expensive quickly.