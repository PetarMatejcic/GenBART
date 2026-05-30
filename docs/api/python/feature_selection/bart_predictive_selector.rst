BartPredictiveSelector
======================

.. currentmodule:: genbart.feature_selection

``BartPredictiveSelector`` performs predictive-degradation feature selection.

The selector fits repeated BART models, computes a baseline predictive loss,
then permutes each feature column and measures how much the loss increases.


Class
-----

.. autoclass:: BartPredictiveSelector
   :show-inheritance:


Methods
-------

.. automethod:: BartPredictiveSelector.from_model

.. automethod:: BartPredictiveSelector.fit

.. automethod:: BartPredictiveSelector.selected_features

.. automethod:: BartPredictiveSelector.selected_indices

.. automethod:: BartPredictiveSelector.selected_mask

.. automethod:: BartPredictiveSelector.ranking

.. automethod:: BartPredictiveSelector.to_frame

.. automethod:: BartPredictiveSelector.summary


Classification example
----------------------

.. code-block:: python

   from genbart.probit_bart import ProbitBart
   from genbart.feature_selection import BartPredictiveSelector

   selector = BartPredictiveSelector(
       ProbitBart,
       model_params={
           "m": 50,
           "n_burn": 100,
           "n_samples": 300,
       },
       task="classification",
       loss_metric="brier",
       n_repeats=5,
       n_permutations=10,
       selection_probability=0.95,
       min_mean_degradation=0.0,
       use_posterior_draws=False,
       random_state=0,
   )

   result = selector.fit(X, y, feature_names=feature_names)

   result.selected_features()
   result.to_frame()


Regression example
------------------

.. code-block:: python

   from genbart.reg_bart import RegBart
   from genbart.feature_selection import BartPredictiveSelector

   selector = BartPredictiveSelector(
       RegBart,
       model_params={
           "m": 50,
           "n_burn": 100,
           "n_samples": 300,
       },
       task="regression",
       loss_metric="mse",
       n_repeats=5,
       n_permutations=10,
       selection_probability=0.95,
       min_mean_degradation=0.0,
       use_posterior_draws=False,
       random_state=0,
   )

   result = selector.fit(X, y, feature_names=feature_names)