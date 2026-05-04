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
                score = mean_squared_error(y[val_idx], pred, squared=True)
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