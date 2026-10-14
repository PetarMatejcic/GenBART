Feature Selection Results
=========================

.. currentmodule:: genbart.feature_selection

This page documents the result containers returned by the feature-selection
selectors.


Result hierarchy
----------------

Permutation-null selection produces two levels of result objects.

``VariableSelectionResult``
   Contains the complete result for one importance definition, either
   ``"raw"`` or ``"logml"``. It stores observed importance, repeated-fit
   variability, the permutation-null distribution, and all supported
   thresholding methods.

``ThresholdResult``
   Contains the selection produced by one thresholding method such as
   ``"local"``, ``"global_max"``, or ``"global_se"``.

``PredictiveSelectionResult``
   Contains output from the experimental ``BartPredictiveSelector`` API.


ThresholdResult
---------------

``ThresholdResult`` stores the output of one thresholding rule.

.. autoclass:: ThresholdResult
   :show-inheritance:


Methods
~~~~~~~

.. automethod:: ThresholdResult.selected_features

.. automethod:: ThresholdResult.selected_indices

.. automethod:: ThresholdResult.selected_mask

.. automethod:: ThresholdResult.n_selected

.. automethod:: ThresholdResult.threshold_for

.. automethod:: ThresholdResult.importance_for


VariableSelectionResult
-----------------------

``VariableSelectionResult`` is returned by ``BartVariableSelector``.

.. autoclass:: VariableSelectionResult
   :show-inheritance:


Properties
~~~~~~~~~~

.. autoattribute:: VariableSelectionResult.null_mean

.. autoattribute:: VariableSelectionResult.null_sd


Methods
~~~~~~~

.. automethod:: VariableSelectionResult.selected_features

.. automethod:: VariableSelectionResult.selected_indices

.. automethod:: VariableSelectionResult.selected_mask

.. automethod:: VariableSelectionResult.thresholds

.. automethod:: VariableSelectionResult.ranking

.. automethod:: VariableSelectionResult.to_frame

.. automethod:: VariableSelectionResult.summary

.. automethod:: VariableSelectionResult.compare_methods


PredictiveSelectionResult
-------------------------

``PredictiveSelectionResult`` is returned by the experimental
``BartPredictiveSelector`` API.

.. warning::

   This result type is experimental and may change or be removed in a future
   release.

.. autoclass:: PredictiveSelectionResult
   :show-inheritance:


Methods
~~~~~~~

.. automethod:: PredictiveSelectionResult.selected_features

.. automethod:: PredictiveSelectionResult.selected_indices

.. automethod:: PredictiveSelectionResult.selected_mask

.. automethod:: PredictiveSelectionResult.n_selected

.. automethod:: PredictiveSelectionResult.ranking

.. automethod:: PredictiveSelectionResult.to_frame

.. automethod:: PredictiveSelectionResult.summary


Threshold functions
-------------------

The permutation-null selector computes all three thresholding rules from the
same observed and null importance values.

``local_threshold``
   Compares each feature with its own null distribution.

``global_max_threshold``
   Uses one common threshold derived from the largest null importance in each
   response permutation.

``global_se_threshold``
   Uses one common standardized multiplier and converts it into
   feature-specific thresholds using each feature's null mean and standard
   deviation.

.. note::

   The permutation-null distribution is empirical. Larger
   ``n_permutations`` provides better resolution in the tails of the null
   distribution, at the cost of additional BART fits.

.. autofunction:: local_threshold

.. autofunction:: global_max_threshold

.. autofunction:: global_se_threshold

.. autofunction:: build_threshold_results