Feature Selection
=================

GenBART provides two feature-selection tools:

``BartVariableSelector``
   A BART-native selector based on posterior variable-inclusion proportions.
   It compares observed inclusion values against a permuted-response null
   distribution.

``BartPredictiveSelector``
   A predictive-degradation selector. It permutes one feature at a time and
   measures how much predictive loss increases.

Both selectors are available from ``genbart.feature_selection``:

.. code-block:: python

   from genbart.feature_selection import (
       BartVariableSelector,
       BartPredictiveSelector,
   )


Overview
--------

``BartVariableSelector`` is useful when you want BART-native variable screening
based on posterior split usage.

``BartPredictiveSelector`` is useful when you want predictive importance: how
much model performance worsens when a feature is disrupted.

For exploratory work, it is often useful to compare both selectors.


Permutation-null variable selection
-----------------------------------

``BartVariableSelector`` fits repeated BART models on the observed response,
then repeated BART models on permuted responses. The permuted-response fits
define a null distribution for feature importance.

It supports two importance kinds:

``"raw"``
   Standard split-count variable-inclusion proportions.

``"logml"``
   Log marginal likelihood weighted variable-inclusion proportions.

It supports three thresholding methods:

``"local"``
   Feature-specific permutation-null threshold.

``"global_max"``
   Global threshold based on the maximum null importance across features.

``"global_se"``
   Global standardized threshold mapped back to feature-specific scales.


Predictive-degradation selection
--------------------------------

``BartPredictiveSelector`` measures how much predictive loss increases when
a feature is permuted.

For ``task="classification"``, supported losses are ``"brier"`` and
``"log_loss"``.

For ``task="regression"``, supported losses are ``"mse"`` and ``"rmse"``.

If ``use_posterior_draws=False``, the selector averages
``posterior_sample_draws(X)`` and scores the posterior mean prediction.

If ``use_posterior_draws=True``, the selector scores every posterior sample
draw separately.


Pages
-----

.. toctree::
   :maxdepth: 1

   bart_variable_selector
   bart_predictive_selector
   feature_selection_results