RegBart
=======

.. currentmodule:: genbart.reg_bart

``RegBart`` is the regression BART estimator for continuous response variables.

Class
-----

.. autoclass:: RegBart
   :show-inheritance:


Method overview
---------------

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Method
     - Purpose
   * - :meth:`fit <RegBart.fit>`
     - Fit the regression BART model and retain posterior forest draws.
   * - :meth:`predict <RegBart.predict>`
     - Return posterior point predictions and credible intervals for the regression function.
   * - :meth:`posterior_sample_draws <RegBart.posterior_sample_draws>`
     - Access individual posterior draws of the regression function.
   * - :meth:`partial_dependence <RegBart.partial_dependence>`
     - Compute one-variable partial dependence with posterior uncertainty.
   * - :meth:`evaluate <RegBart.evaluate>`
     - Compute regression point-prediction metrics.
   * - :meth:`variable_inclusion <RegBart.variable_inclusion>`
     - Return posterior mean raw or logML-weighted variable importance.
   * - :meth:`variable_inclusion_draws <RegBart.variable_inclusion_draws>`
     - Access draw-level variable-importance vectors.
   * - :meth:`get_params <RegBart.get_params>`
     - Return the estimator's constructor parameters.


Methods
-------

.. automethod:: RegBart.fit

.. automethod:: RegBart.predict

.. automethod:: RegBart.posterior_sample_draws

.. automethod:: RegBart.partial_dependence

.. automethod:: RegBart.evaluate

.. automethod:: RegBart.variable_inclusion

.. automethod:: RegBart.variable_inclusion_draws

.. automethod:: RegBart.get_params