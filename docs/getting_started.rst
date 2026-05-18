Getting Started
===============

This page shows the basic GenBART workflow for regression and binary
classification.

Regression
----------

Use :class:`genbart.reg_bart.RegBart` for continuous response variables.

.. code-block:: python

   from genbart.reg_bart import RegBart

   model = RegBart(
       m=200,
       n_burn=200,
       n_samples=1000,
       random_state=0,
   )

   model.fit(X_train, y_train)

   pred = model.predict(X_test)

The returned prediction object is a dictionary containing posterior point
predictions and, by default, pointwise credible intervals.

.. code-block:: python

   y_hat = pred["prediction"]
   lower = pred["conf_int_low"]
   upper = pred["conf_int_high"]


Binary classification
---------------------

Use :class:`genbart.probit_bart.ProbitBart` for binary response variables.

.. code-block:: python

   from genbart.probit_bart import ProbitBart

   model = ProbitBart(
       m=200,
       n_burn=200,
       n_samples=1000,
       random_state=0,
   )

   model.fit(X_train, y_train)

   probs = model.predict_probs(X_test)
   labels = model.predict(X_test)

The probability prediction object contains posterior mean probabilities and
credible intervals.

.. code-block:: python

   p_hat = probs["probs"]
   lower = probs["conf_int_low"]
   upper = probs["conf_int_high"]


Variable importance
-------------------

Both regression and classification models expose variable inclusion summaries
through ``variable_importance``.

.. code-block:: python

   importance = model.variable_importance()

The returned array has one value per predictor.