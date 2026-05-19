Variable selection results
==========================

.. currentmodule:: genbart.variable_selection

This page documents the result objects returned by
:class:`BartVariableSelector`.

VariableSelectionResult
-----------------------

``VariableSelectionResult`` stores the observed variable inclusion proportions,
permutation-null inclusion proportions, and thresholding results returned by
``BartVariableSelector.fit``.

Class
~~~~~

.. autoclass:: VariableSelectionResult
   :show-inheritance:

Properties
~~~~~~~~~~

.. autoproperty:: VariableSelectionResult.null_mean

.. autoproperty:: VariableSelectionResult.null_sd

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

ThresholdResult
---------------

``ThresholdResult`` stores the selected-feature mask and thresholds for one
variable-selection method.

Class
~~~~~

.. autoclass:: ThresholdResult
   :show-inheritance:

Methods
~~~~~~~

.. automethod:: ThresholdResult.selected_features

.. automethod:: ThresholdResult.selected_indices

.. automethod:: ThresholdResult.n_selected

.. automethod:: ThresholdResult.threshold_for

.. automethod:: ThresholdResult.vip_for