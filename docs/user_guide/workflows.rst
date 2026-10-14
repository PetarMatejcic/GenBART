Working with GenBART
====================

This page describes how the main GenBART tools fit together after a model has
been constructed.

For details on individual arguments and return values, use the
:doc:`../api/python/index`.


Choosing an estimator
---------------------

Use :class:`genbart.reg_bart.RegBart` when the response is continuous.

Use :class:`genbart.probit_bart.ProbitBart` when the response is binary and
encoded as 0 and 1.

Both estimators share the same basic workflow:

.. code-block:: python

   model.fit(X_train, y_train)

   prediction = model.predict(X_test)

The fitted model stores ``n_samples`` posterior forest draws. Prediction,
credible intervals, partial dependence, and variable importance are all
derived from these retained posterior samples.


Point predictions and posterior uncertainty
-------------------------------------------

The high-level prediction methods summarize the retained posterior draws.

For regression:

.. code-block:: python

   pred = model.predict(
       X_test,
       central_measure="mean",
       level=0.90,
   )

For probit classification:

.. code-block:: python

   pred = model.predict_probs(
       X_test,
       level=0.90,
   )

Regression credible intervals quantify posterior uncertainty in the fitted
conditional mean function

.. math::

   f(x) = E[Y \mid X=x].

They do not include a new draw of the Gaussian observation noise and should
therefore not be interpreted as posterior predictive intervals for future
observed responses.

For probit BART, credible intervals quantify posterior uncertainty in the
positive-class probability

.. math::

   P(Y=1 \mid X=x).


Using posterior draws directly
------------------------------

Use ``posterior_sample_draws`` when the built-in summaries are not sufficient.

.. code-block:: python

   draws = model.posterior_sample_draws(X_test)

The returned array has shape

.. code-block:: text

   (n_samples, n_observations)

For ``RegBart``, these are posterior draws of the regression function on the
original response scale.

For ``ProbitBart``, these are posterior draws of the positive-class
probability.

This makes it possible to construct custom posterior summaries without
refitting the model.


Evaluating predictive performance
---------------------------------

Both estimators provide an ``evaluate`` method.

.. code-block:: python

   metrics = model.evaluate(X_test, y_test)

For regression, the reported metrics evaluate posterior point predictions.

For classification, the result includes both probability-based metrics and
threshold-based classification metrics.

Evaluation metrics summarize predictive performance; they do not summarize
posterior uncertainty. Use the prediction methods or posterior draws when
uncertainty is required.


Partial dependence
------------------

``partial_dependence`` computes classical one-variable partial dependence.

For a predictor :math:`x_j`, GenBART approximates

.. math::

   \operatorname{PD}(x_j)
   =
   \frac{1}{n}
   \sum_{i=1}^{n}
   f(x_j, x_{i,-j}).

For each grid value, the selected predictor is replaced for every training
observation while the observed values of all remaining predictors are
preserved.

.. code-block:: python

   pd = model.partial_dependence(
       variable=0,
       grid_samples=100,
       level=0.90,
   )

   grid = pd["grid"]
   effect = pd["prediction"]

Posterior averaging is performed separately for each retained forest before
the final partial-dependence summary and credible interval are calculated.

For regression the result is on the response scale. For probit BART it is on
the positive-class probability scale.


Variable importance
-------------------

GenBART provides two split-based variable-importance definitions.

Standard variable inclusion proportions count how often each predictor is
used in splitting rules:

.. code-block:: python

   raw = model.variable_inclusion(kind="raw")

Log marginal-likelihood-weighted importance additionally considers the
structural contribution of each split:

.. code-block:: python

   logml = model.variable_inclusion(kind="logml")

For an internal node, the logML statistic compares the marginal likelihood of
retaining the subtree below that node with the marginal likelihood obtained by
collapsing the node to a terminal node.

Only positive gains contribute to the importance statistic.

Both statistics are normalized separately within each posterior forest and
then averaged over posterior draws.

Use

.. code-block:: python

   draws = model.variable_inclusion_draws(kind="logml")

when draw-level importance values are required.


Variable importance versus variable selection
---------------------------------------------

Variable importance ranks or describes predictors. It does not by itself
define a selected subset.

For permutation-null variable selection, use
:class:`genbart.feature_selection.BartVariableSelector`.

.. code-block:: python

   from genbart import BartVariableSelector, RegBart

   selector = BartVariableSelector(
       model_cls=RegBart,
       model_params={
           "m": 20,
           "n_burn": 200,
           "n_samples": 500,
       },
       n_permutations=20,
       n_repeats=5,
       method="global_se",
       importance_kind="raw",
       random_state=0,
   )

   result = selector.fit(X, y)

   selected = result.selected_features()

The selector fits repeated BART models on the observed response and compares
their importance values with importance obtained after permuting the response.

Both raw and logML importance are computed during the same selector run. The
alternative result can therefore be inspected without refitting:

.. code-block:: python

   raw_result = selector.get_result("raw")
   logml_result = selector.get_result("logml")

Three thresholding rules are available: ``"local"``, ``"global_max"``, and
``"global_se"``. See the
:doc:`../api/python/feature_selection/index` reference for their precise
definitions.


Hyperparameter selection
------------------------

For straightforward grid-search cross-validation, GenBART provides

``cross_validate_reg_bart``
   Regression grid search using K-fold cross-validation.

``cross_validate_probit_bart``
   Binary-classification grid search using stratified K-fold
   cross-validation.

For example:

.. code-block:: python

   from genbart import cross_validate_reg_bart

   search = cross_validate_reg_bart(
       X,
       y,
       param_grid={
           "m": [50, 100],
           "k": [1.0, 2.0],
           "nu": [3.0],
           "q": [0.90],
           "n_burn": [200],
           "n_samples": [500],
       },
       cv=5,
       scoring="rmse",
       random_state=0,
   )

   best_model = search["best_model"]

Grid search can require many BART fits. Use a focused parameter grid when
MCMC settings are expensive.

See :doc:`parameters` for interpretation of the individual BART
hyperparameters.


Recommended workflow
--------------------

For a typical analysis:

#. Choose ``RegBart`` or ``ProbitBart`` according to the response type.
#. Fit the model and inspect predictive performance on held-out data.
#. Use posterior draws or credible intervals when uncertainty matters.
#. Use partial dependence to inspect fitted marginal relationships.
#. Examine raw and logML variable importance for feature relevance.
#. Use ``BartVariableSelector`` when an explicit selected feature set is
   required.
#. Use cross-validation when comparing BART hyperparameter configurations.

These steps serve different purposes. Prediction, interpretation, variable
importance, feature selection, and hyperparameter selection should not be
treated as interchangeable summaries of the fitted model.