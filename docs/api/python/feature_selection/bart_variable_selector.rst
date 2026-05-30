BartVariableSelector
====================

.. currentmodule:: genbart.feature_selection

``BartVariableSelector`` performs permutation-null variable selection using
BART posterior variable-inclusion proportions.

The selector fits repeated BART models on the observed response, then repeated
BART models on permuted responses. The observed importance values are compared
against the permutation-null distribution using local, global-max, and global-SE
thresholding rules.


Class
-----

.. autoclass:: BartVariableSelector
   :show-inheritance:


Methods
-------

.. automethod:: BartVariableSelector.from_model

.. automethod:: BartVariableSelector.fit

.. automethod:: BartVariableSelector.selected_features

.. automethod:: BartVariableSelector.selected_indices

.. automethod:: BartVariableSelector.selected_mask

.. automethod:: BartVariableSelector.ranking

.. automethod:: BartVariableSelector.to_frame

.. automethod:: BartVariableSelector.summary

.. automethod:: BartVariableSelector.compare_methods


Example
-------

.. code-block:: python

   from genbart.probit_bart import ProbitBart
   from genbart.feature_selection import BartVariableSelector

   selector = BartVariableSelector(
       ProbitBart,
       model_params={
           "m": 20,
           "n_burn": 100,
           "n_samples": 300,
       },
       n_permutations=20,
       n_repeats=5,
       alpha=0.05,
       method="global_se",
       importance_kind="raw",
       random_state=0,
   )

   result = selector.fit(X, y, feature_names=feature_names)

   result.selected_features()
   result.to_frame()
   result.compare_methods()