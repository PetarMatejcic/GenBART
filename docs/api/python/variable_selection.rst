Variable selection
==================

BART variable selection uses variable inclusion proportions: the proportion of
splitting rules that use each predictor across posterior tree samples.

The variable-selection workflow is implemented by
:class:`genbart.variable_selection.BartVariableSelector`, but this page focuses
on the workflow rather than the full class reference.

Workflow
--------

A typical workflow has four steps:

1. Configure a BART model for variable selection.
2. Create a selector from the model class or from an existing model instance.
3. Fit the selector on ``X`` and ``y``.
4. Inspect the returned :class:`genbart.variable_selection.VariableSelectionResult`.

Example
-------

.. code-block:: python

   from genbart import RegBart
   from genbart.variable_selection import BartVariableSelector

   selector = BartVariableSelector(
       model_cls=RegBart,
       model_params={
           "m": 20,
           "n_burn": 500,
           "n_samples": 1000,
       },
       n_permutations=100,
       n_repeats=5,
       alpha=0.05,
       method="global_se",
       random_state=0,
   )

   result = selector.fit(X, y)

   result.selected_features()
   result.to_frame()
   result.compare_methods()

Using an existing model
-----------------------

If a BART model implements ``get_params()``, a selector can also be constructed
from an existing model instance.

.. code-block:: python

   model = RegBart(m=20, n_burn=500, n_samples=1000)

   selector = BartVariableSelector.from_model(
       model,
       n_permutations=100,
       n_repeats=5,
       method="global_se",
       random_state=0,
   )

   result = selector.fit(X, y)

Thresholding methods
--------------------

``local``
    Compares each variable against its own permutation-null quantile.

``global_max``
    Computes the maximum null inclusion proportion across variables for each
    response permutation and applies one global threshold to all variables.

``global_se``
    Standardizes each variable's permutation-null inclusion proportions, applies
    a global standardized cutoff, and maps that cutoff back to variable-specific
    thresholds.

Returned result
---------------

``fit`` returns a :class:`genbart.variable_selection.VariableSelectionResult`.

The most useful methods are:

.. code-block:: python

   result.selected_features()
   result.selected_indices()
   result.selected_mask()
   result.thresholds()
   result.to_frame()
   result.summary()
   result.compare_methods()

API reference
-------------

For the full class reference, see:

.. toctree::
   :maxdepth: 1

   bart_variable_selector
   variable_selection_results