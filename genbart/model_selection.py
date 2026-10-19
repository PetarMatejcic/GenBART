"""Cross-validation utilities for selecting GenBART hyperparameters.

The module provides lightweight grid-search helpers for regression and probit
BART. Candidate parameter combinations are evaluated by cross-validation and
the best configuration can optionally be refitted on the complete dataset.

For more specialized resampling schemes or search strategies, users can build
custom workflows around :class:`genbart.reg_bart.RegBart` and
:class:`genbart.probit_bart.ProbitBart`.
"""


import numpy as np

from sklearn.model_selection import ParameterGrid, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, root_mean_squared_error, roc_auc_score, accuracy_score

from genbart.reg_bart import RegBart
from genbart.probit_bart import ProbitBart


def cross_validate_reg_bart(
    X,
    y,
    param_grid=None,
    cv=5,
    scoring="rmse",
    refit=True,
    random_state=0,
):
    """Select regression BART hyperparameters by grid-search cross-validation.

    Every combination in ``param_grid`` is evaluated using shuffled K-fold
    cross-validation. A fresh :class:`RegBart` model is fitted on each training
    fold and scored on the corresponding validation fold.

    The configuration with the lowest mean validation loss is selected.

    Parameters
    ----------
    X : array-like
        Predictor data.

    y : array-like
        Continuous response values.

    param_grid : dict, optional
        Mapping from ``RegBart`` constructor parameter names to candidate values.

        Candidate combinations are generated using
        :class:`sklearn.model_selection.ParameterGrid`.

        If omitted, the following grid is used:

        - ``m``: ``[50, 100, 200]``
        - ``k``: ``[1.0, 2.0, 3.0]``
        - ``nu``: ``[3.0]``
        - ``q``: ``[0.90, 0.99]``
        - ``n_burn``: ``[200]``
        - ``n_samples``: ``[1000]``

    cv : int, default=5
        Number of K-fold cross-validation splits.

    scoring : {"rmse", "mse"}, default="rmse"
        Validation loss used to compare parameter combinations.

        Lower values are better for both supported metrics.

    refit : bool, default=True
        Whether to fit a new ``RegBart`` model with the best parameters on the
        complete ``X`` and ``y`` dataset after cross-validation.

    random_state : int, default=0
        Seed controlling fold shuffling and the seeds assigned to models fitted
        within each fold.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"best_params"``: parameter dictionary with the lowest mean validation
        loss.
        - ``"best_score"``: mean validation loss for that configuration.
        - ``"best_model"``: model refitted on all observations when
        ``refit=True``; otherwise ``None``.
        - ``"cv_results"``: list containing one dictionary for every parameter
        combination.

        Each entry in ``"cv_results"`` contains:

        - ``"params"``: evaluated parameter combination.
        - ``"mean_score"``: mean validation loss across folds.
        - ``"std_score"``: standard deviation of validation loss across folds.
        - ``"fold_scores"``: individual validation-fold scores.

    Raises
    ------
    ValueError
        If ``scoring`` is not ``"rmse"`` or ``"mse"``.

    Notes
    -----
    The default parameter grid requires many BART fits and can therefore be
    computationally expensive. For exploratory tuning, it is often useful to
    start with a smaller grid or fewer retained posterior samples.

    The cross-validation score evaluates posterior point predictions. Posterior
    credible intervals are not used during model selection.
    """
    if param_grid is None:
        param_grid = {
            "m": [50, 100, 200],
            "k": [1.0, 2.0, 3.0],
            "nu": [3.0],
            "q": [0.90, 0.99],
            "n_burn": [200],
            "n_samples": [1000],
        }

    X = np.asarray(X)
    y = np.asarray(y)

    splitter = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    results = []

    for params in ParameterGrid(param_grid):
        fold_scores = []

        for fold_id, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            model = RegBart(
                **params,
                random_state=random_state + fold_id,
            )
            model.fit(X[train_idx], y[train_idx])

            pred = model.predict(X[val_idx], conf_int=False)["prediction"]

            if scoring == "rmse":
                score = root_mean_squared_error(y[val_idx], pred)
            elif scoring == "mse":
                score = mean_squared_error(y[val_idx], pred)
            else:
                raise ValueError("scoring must be 'rmse' or 'mse'")

            fold_scores.append(score)

        results.append({
            "params": params,
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
        })

    best_result = min(results, key=lambda r: r["mean_score"])

    best_model = None
    if refit:
        best_model = RegBart(
            **best_result["params"],
            random_state=random_state,
        )
        best_model.fit(X, y)

    return {
        "best_params": best_result["params"],
        "best_score": best_result["mean_score"],
        "best_model": best_model,
        "cv_results": results,
    }


def cross_validate_probit_bart(
    X,
    y,
    param_grid=None,
    cv=5,
    scoring="auc",
    refit=True,
    random_state=0,
):
    """Select probit BART hyperparameters by grid-search cross-validation.

    Every combination in ``param_grid`` is evaluated using shuffled stratified
    K-fold cross-validation. Stratification preserves the binary class balance as
    far as possible across folds.

    The configuration with the highest mean validation score is selected.

    Parameters
    ----------
    X : array-like
        Predictor data.

    y : array-like
        Binary response values encoded as 0 and 1.

    param_grid : dict, optional
        Mapping from ``ProbitBart`` constructor parameter names to candidate
        values.

        Candidate combinations are generated using
        :class:`sklearn.model_selection.ParameterGrid`.

        If omitted, the following grid is used:

        - ``m``: ``[50, 100, 200]``
        - ``k``: ``[1.0, 2.0, 3.0]``
        - ``n_burn``: ``[200]``
        - ``n_samples``: ``[1000]``

    cv : int, default=5
        Number of stratified cross-validation folds.

    scoring : {"auc", "accuracy"}, default="auc"
        Validation metric used to compare parameter combinations.

        ``"auc"`` evaluates posterior mean positive-class probabilities.

        ``"accuracy"`` converts posterior mean probabilities to class labels using
        a fixed threshold of ``0.5``.

        Higher values are better for both supported metrics.

    refit : bool, default=True
        Whether to fit a new ``ProbitBart`` model with the best parameters on the
        complete dataset after cross-validation.

    random_state : int, default=0
        Seed controlling fold shuffling and the seeds assigned to models fitted
        within each fold.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"best_params"``: parameter dictionary with the highest mean validation
        score.
        - ``"best_score"``: mean validation score for that configuration.
        - ``"best_model"``: model refitted on all observations when
        ``refit=True``; otherwise ``None``.
        - ``"cv_results"``: list containing one dictionary for every parameter
        combination.

        Each entry in ``"cv_results"`` contains:

        - ``"params"``: evaluated parameter combination.
        - ``"mean_score"``: mean validation score across folds.
        - ``"std_score"``: standard deviation across folds.
        - ``"fold_scores"``: individual validation-fold scores.

    Raises
    ------
    ValueError
        If ``scoring`` is not ``"auc"`` or ``"accuracy"``.

    Notes
    -----
    AUC is computed from posterior mean class probabilities and is independent of
    a classification threshold.

    Accuracy uses a fixed probability threshold of ``0.5``.

    Because stratified cross-validation requires observations from each class in
    the folds, ``cv`` should be chosen appropriately for the available class
    counts.
    """
    if param_grid is None:
        param_grid = {
            "m": [50, 100, 200],
            "k": [1.0, 2.0, 3.0],
            "n_burn": [200],
            "n_samples": [1000],
        }

    X = np.asarray(X)
    y = np.asarray(y)

    splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    results = []

    for params in ParameterGrid(param_grid):
        fold_scores = []

        for fold_id, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            model = ProbitBart(
                **params,
                random_state=random_state + fold_id,
            )
            model.fit(X[train_idx], y[train_idx])

            probs = model.predict_probs(X[val_idx])["probs"]

            if scoring == "auc":
                score = roc_auc_score(y[val_idx], probs)
            elif scoring == "accuracy":
                pred = probs >= 0.5
                score = accuracy_score(y[val_idx], pred)
            else:
                raise ValueError("scoring must be 'auc' or 'accuracy'")

            fold_scores.append(score)

        results.append({
            "params": params,
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
        })

    best_result = max(results, key=lambda r: r["mean_score"])

    best_model = None
    if refit:
        best_model = ProbitBart(
            **best_result["params"],
            random_state=random_state,
        )
        best_model.fit(X, y)

    return {
        "best_params": best_result["params"],
        "best_score": best_result["mean_score"],
        "best_model": best_model,
        "cv_results": results,
    }