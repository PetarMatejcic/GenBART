Getting Started
===============

This page shows the basic GenBART workflow for regression and binary
classification.

For a more detailed discussion of posterior draws, interpretation, variable
selection, and model selection, see :doc:`user_guide/workflows`.


Installation
------------

From the repository root, install GenBART in editable mode with:

.. code-block:: bash

   pip install -e .

GenBART includes a compiled C++ backend, so installation requires a working
Python and C++ build environment.


Regression
----------

Use :class:`genbart.reg_bart.RegBart` for continuous response variables.

.. code-block:: python

   from genbart import RegBart

   model = RegBart(
       m=200,
       n_burn=200,
       n_samples=1000,
       random_state=0,
   )

   model.fit(X_train, y_train)

   pred = model.predict(X_test)

``predict`` returns posterior point predictions and, by default, pointwise
credible intervals:

.. code-block:: python

   y_hat = pred["prediction"]
   lower = pred["conf_int_low"]
   upper = pred["conf_int_high"]

These intervals describe posterior uncertainty in the conditional mean
response. They do not include observation noise and therefore are not
posterior predictive intervals for future observed responses.

Basic regression metrics are available through:

.. code-block:: python

   metrics = model.evaluate(X_test, y_test)


Binary classification
---------------------

Use :class:`genbart.probit_bart.ProbitBart` for binary responses encoded as
0 and 1.

.. code-block:: python

   from genbart import ProbitBart

   model = ProbitBart(
       m=200,
       n_burn=200,
       n_samples=1000,
       random_state=0,
   )

   model.fit(X_train, y_train)

   probs = model.predict_probs(X_test)
   labels = model.predict(X_test)

The probability result contains posterior mean positive-class probabilities
and credible intervals:

.. code-block:: python

   p_hat = probs["probs"]
   lower = probs["conf_int_low"]
   upper = probs["conf_int_high"]

These intervals describe posterior uncertainty in
:math:`P(Y=1 \mid X=x)`.

Classification metrics are available through:

.. code-block:: python

   metrics = model.evaluate(X_test, y_test)


Variable importance
-------------------

Both estimators expose BART variable-importance summaries through
:meth:`genbart.base_bart.BaseBART.variable_inclusion`.

Standard variable inclusion proportions are returned by default:

.. code-block:: python

   raw_importance = model.variable_inclusion()

GenBART also provides log marginal-likelihood-weighted importance:

.. code-block:: python

   logml_importance = model.variable_inclusion(kind="logml")

``"raw"`` measures the proportion of tree splits using each predictor.

``"logml"`` weights split occurrences by their positive log
marginal-likelihood contribution before normalization.

Draw-level importance values are available through
:meth:`genbart.base_bart.BaseBART.variable_inclusion_draws`.


Partial dependence
------------------

One-variable partial dependence summarizes the fitted relationship with a
predictor while averaging over the observed training values of the remaining
predictors.

.. code-block:: python

   pd = model.partial_dependence(variable=0)

   grid = pd["grid"]
   effect = pd["prediction"]
   lower = pd["conf_int_low"]
   upper = pd["conf_int_high"]

For regression, the result is on the response scale. For probit BART, it is
on the positive-class probability scale.


Where to go next
----------------

The :doc:`user_guide/workflows` page explains how posterior draws,
uncertainty summaries, variable importance, feature selection, and
cross-validation fit together.

See :doc:`user_guide/parameters` for interpretation of the main BART prior
and MCMC parameters.

The :doc:`api/python/index` contains the complete Python API reference.