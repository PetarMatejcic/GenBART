ProbitBart
==========

.. currentmodule:: genbart.probit_bart

``ProbitBart`` is the probit BART estimator for binary classification.

Class
-----

.. autoclass:: ProbitBart
   :show-inheritance:


Method overview
---------------

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Method
     - Purpose
   * - :meth:`fit <ProbitBart.fit>`
     - Fit the probit BART classifier and retain posterior forest draws.
   * - :meth:`predict_probs <ProbitBart.predict_probs>`
     - Return posterior positive-class probabilities and credible intervals.
   * - :meth:`predict <ProbitBart.predict>`
     - Convert posterior mean probabilities into binary class predictions.
   * - :meth:`posterior_sample_draws <ProbitBart.posterior_sample_draws>`
     - Access individual posterior draws of positive-class probabilities.
   * - :meth:`partial_dependence <ProbitBart.partial_dependence>`
     - Compute probability-scale partial dependence with posterior uncertainty.
   * - :meth:`evaluate <ProbitBart.evaluate>`
     - Compute classification and probability-prediction metrics.
   * - :meth:`calibration_curve <ProbitBart.calibration_curve>`
     - Summarize probability calibration in bins.
   * - :meth:`variable_inclusion <ProbitBart.variable_inclusion>`
     - Return posterior mean raw or logML-weighted variable importance.
   * - :meth:`variable_inclusion_draws <ProbitBart.variable_inclusion_draws>`
     - Access draw-level variable-importance vectors.
   * - :meth:`get_params <ProbitBart.get_params>`
     - Return the estimator's constructor parameters.


Methods
-------

.. automethod:: ProbitBart.fit

.. automethod:: ProbitBart.predict_probs

.. automethod:: ProbitBart.predict

.. automethod:: ProbitBart.posterior_sample_draws

.. automethod:: ProbitBart.partial_dependence

.. automethod:: ProbitBart.evaluate

.. automethod:: ProbitBart.calibration_curve

.. automethod:: ProbitBart.variable_inclusion

.. automethod:: ProbitBart.variable_inclusion_draws

.. automethod:: ProbitBart.get_params