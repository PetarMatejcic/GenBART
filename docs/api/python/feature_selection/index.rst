Feature Selection
=================

GenBART's primary feature-selection workflow is
``BartVariableSelector``, which performs permutation-null selection using BART
variable-importance statistics.

An additional experimental selector, ``BartPredictiveSelector``, is included
for predictive-degradation experiments.


Primary selector
----------------

``BartVariableSelector``
   A BART-native permutation-null selector based on posterior variable
   importance. It supports both standard split-count inclusion proportions and
   GenBART's log marginal-likelihood-weighted importance.


Experimental selector
---------------------

``BartPredictiveSelector``
   An experimental procedure that permutes individual predictors and measures
   the resulting change in predictive loss.

   This API is exploratory and may change or be removed in a future release.

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


Experimental predictive-degradation selection
----------------------------------------------

.. warning::

   ``BartPredictiveSelector`` is experimental. Its methodology and API may
   change or be removed in a future release.

``BartPredictiveSelector`` measures how predictive loss changes when a feature
is permuted in evaluation data.

Classification supports ``"brier"`` and ``"log_loss"`` losses. Regression
supports ``"mse"`` and ``"rmse"``.

See :doc:`bart_predictive_selector` for the current API.


Pages
-----

.. toctree::
   :maxdepth: 1

   bart_variable_selector
   bart_predictive_selector
   feature_selection_results