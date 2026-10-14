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


Method overview
---------------

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Method
     - Purpose
   * - :meth:`from_model <BartVariableSelector.from_model>`
     - Create a selector from an existing BART model configuration.
   * - :meth:`fit <BartVariableSelector.fit>`
     - Fit observed-response and permutation-null BART models.
   * - :meth:`get_result <BartVariableSelector.get_result>`
     - Retrieve the raw or logML variable-selection result.
   * - :meth:`selected_features <BartVariableSelector.selected_features>`
     - Return names of features selected by a thresholding rule.
   * - :meth:`selected_indices <BartVariableSelector.selected_indices>`
     - Return indices of selected predictors.
   * - :meth:`selected_mask <BartVariableSelector.selected_mask>`
     - Return a Boolean selected-feature mask.
   * - :meth:`ranking <BartVariableSelector.ranking>`
     - Rank predictors by decreasing observed importance.
   * - :meth:`to_frame <BartVariableSelector.to_frame>`
     - Return a row-wise summary of feature-selection results.
   * - :meth:`summary <BartVariableSelector.summary>`
     - Return a compact summary of the fitted result.
   * - :meth:`compare_methods <BartVariableSelector.compare_methods>`
     - Compare selections from all supported thresholding rules.


Construction and fitting
------------------------

.. automethod:: BartVariableSelector.from_model

.. automethod:: BartVariableSelector.fit

.. automethod:: BartVariableSelector.get_result


Inspecting results
------------------

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