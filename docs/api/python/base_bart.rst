BaseBART
========

.. currentmodule:: genbart.base_bart

``BaseBART`` contains shared functionality used by the public estimators
:class:`genbart.reg_bart.RegBart` and
:class:`genbart.probit_bart.ProbitBart`.

Users normally interact with those concrete estimator classes directly.

Class
-----

.. autoclass:: BaseBART
   :show-inheritance:


Method overview
---------------

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Method
     - Purpose
   * - :meth:`get_params <BaseBART.get_params>`
     - Return the estimator's constructor configuration.
   * - :meth:`variable_inclusion <BaseBART.variable_inclusion>`
     - Return posterior mean raw or logML-weighted variable importance.
   * - :meth:`variable_inclusion_draws <BaseBART.variable_inclusion_draws>`
     - Access variable-importance vectors for individual posterior draws.


Methods
-------

.. automethod:: BaseBART.get_params

.. automethod:: BaseBART.variable_inclusion

.. automethod:: BaseBART.variable_inclusion_draws