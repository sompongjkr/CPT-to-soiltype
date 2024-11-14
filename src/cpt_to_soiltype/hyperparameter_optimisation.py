from pprint import pformat
from typing import Any

import optuna
import pandas as pd
from rich.console import Console
from sklearn.metrics import balanced_accuracy_score

from cpt_to_soiltype.train_eval_funcs import xgb_native_pipeline


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    oversample_level: int,
    undersample_level: int,
) -> float:

    console = Console()

    # Defining the hyperparameters to be optimised
    model_params = {
        "objective": "multi:softmax",
        "device": "gpu",
        "random_state": 42,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-3, 10.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
    }

    console.print(f"\nSuggested hyperparameters: \n{pformat(trial.params)}")

    # Call the pipeline function
    y_pred = xgb_native_pipeline(
        X_train,
        X_test,
        y_train,
        y_test,
        model_params,
        oversample_level,
        undersample_level,
    )

    # Evaluate the performance using balanced accuracy (or you can use any other metric)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    console.print(f"Balanced accuracy: {balanced_accuracy}")

    return balanced_accuracy


# Run Optuna optimisation
def run_optimisation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    oversample_level: int,
    undersample_level: int,
    n_trials: int = 100,
    study_name: str = "xgboost_hyperparameter_optimisation",
) -> Any:
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction="maximize", study_name=study_name, sampler=sampler
    )
    study.optimize(
        lambda trial: objective(
            trial, X_train, X_test, y_train, y_test, oversample_level, undersample_level
        ),
        n_trials=n_trials,
    )
    return study
