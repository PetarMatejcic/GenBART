Feature Selection Results
=========================

.. currentmodule:: genbart.feature_selection

This page documents the result containers returned by the feature-selection
selectors.


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

``PredictiveSelectionResult`` is returned by ``BartPredictiveSelector``.

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

.. autofunction:: local_threshold

.. autofunction:: global_max_threshold

.. autofunction:: global_se_threshold

.. autofunction:: build_threshold_results