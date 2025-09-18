from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from rich.pretty import pprint
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cpt_to_soiltype.plotting import plot_confusion_matrix


def load_data(
    train_data_path: Path,
    test_data_path: Path,
    target_column: str,
    features_columns: list,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_data = pd.read_csv(Path(train_data_path))
    test_data = pd.read_csv(Path(test_data_path))

    X_train = train_data[features_columns]
    y_train = train_data[target_column]
    X_test = test_data[features_columns]
    y_test = test_data[target_column]

    return X_train, X_test, y_train, y_test


def xgb_native_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_params: dict,
    oversample_level: int,
    undersample_level: int,
    model_save_path: str = "models/xgb_model.json",  # Path to save the model
    save_model: bool = False,
) -> pd.Series:
    # Ensure labels are numeric (but keep original values for mapping back)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    undersample_dict = {
        cls: undersample_level
        for cls in y_train.value_counts().index
        if y_train.value_counts()[cls] > undersample_level
    }
    oversample_dict = {
        cls: oversample_level
        for cls in y_train.value_counts().index
        if y_train.value_counts()[cls] < oversample_level
    }

    # Apply RandomUnderSampler to reduce majority classes
    undersampler = RandomUnderSampler(
        sampling_strategy=undersample_dict, random_state=42
    )

    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

    # Apply SMOTE to oversample minority classes
    smote = SMOTE(sampling_strategy=oversample_dict, random_state=42)
    X_train_final, y_train_final = smote.fit_resample(
        X_train_resampled, y_train_resampled
    )

    # Map labels to contiguous range [0, num_class) required by XGBoost
    unique_labels = sorted(y_train_final.unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    y_train_mapped = y_train_final.map(label_to_idx).astype(int)
    # Map test labels using same mapping; keep original y_test for output comparison
    y_test_mapped = y_test.map(label_to_idx).astype(int)

    # Convert the balanced training data to DMatrix (with GPU support)
    dtrain = xgb.DMatrix(X_train_final, label=y_train_mapped)
    dtest = xgb.DMatrix(X_test, label=y_test_mapped)

    # Set default parameters for XGBoost if not provided
    params = {
        "num_class": len(unique_labels),  # Number of classes
    }
    model_params.update(params)

    # Train the XGBoost model
    xgb_model = xgb.train(model_params, dtrain, num_boost_round=100)

    # optionally save the model
    if save_model:
        xgb_model.save_model(model_save_path)
        pprint(f"Model saved at {model_save_path}")

    # Make predictions (output is class indices for multi:softmax); map back to original labels
    y_pred_indices = xgb_model.predict(dtest)
    # Ensure integer indices for mapping
    y_pred_indices = pd.Series(y_pred_indices).astype(int)
    y_pred = y_pred_indices.map(idx_to_label).to_numpy()
    return y_pred


def train_predict(
    model_name: str,
    model_params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    undersample_level: int,
    oversample_level: int,
    save_model: bool = False,
) -> Pipeline:
    undersample_dict = {
        cls: undersample_level
        for cls in y_train.value_counts().index
        if y_train.value_counts()[cls] > undersample_level
    }
    oversample_dict = {
        cls: oversample_level
        for cls in y_train.value_counts().index
        if y_train.value_counts()[cls] < oversample_level
    }

    # Select classifier based on model_name
    match model_name:
        case "knn":
            classifier = KNeighborsClassifier(**model_params)
        case "xgboost":
            classifier = XGBClassifier(**model_params)
        case "xgboost_native":
            classifier = None
        case "dummy":
            classifier = DummyClassifier(**model_params)
        case "logistic_regression":
            classifier = LogisticRegression(**model_params)
        case "extra_trees":
            classifier = ExtraTreesClassifier(**model_params)
        case "random_forest":
            classifier = RandomForestClassifier(**model_params)
        case _:
            raise NotImplementedError(f"Model {model_name} is not implemented yet.")

    if model_name == "xgboost_native":
        y_pred = xgb_native_pipeline(
            X_train,
            X_test,
            y_train,
            y_test,
            model_params,
            oversample_level,
            undersample_level,
            save_model=save_model,
        )
    else:
        pipeline = make_pipeline(
            StandardScaler(),
            RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42),
            SMOTE(sampling_strategy=oversample_dict, random_state=42),
            classifier,
        )
        # changes the shape of y_train to (n_samples, )
        y_train = y_train.values.ravel()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

    return y_pred


def evaluate_model(
    y_test: pd.Series,
    y_pred: pd.Series,
    class_mapping: dict,
) -> tuple[dict[str, float], dict[str, plt.Figure]]:
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_recall = recall_score(y_test, y_pred, average="macro")
    macro_precision = precision_score(y_test, y_pred, average="macro")
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Store metrics in dictionary
    metrics = {
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
    }

    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(y_test, y_pred, class_mapping)
    artifacts = {"confusion_matrix": cm_fig}

    # Print classification report
    pprint("Classification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, artifacts


def log_mlflow_metrics_and_model(
    mlflow_path: Path,
    experiment_name: str,
    metrics: dict,
    artifacts: dict,
    model_name: str,
    model_params: dict,
    undersample_level: int,
    oversample_level: int,
    hydra_cfg_dir: Path,
) -> None:
    # Setting MLflow experiment and tracking URI
    mlflow.set_tracking_uri(mlflow_path)
    mlflow.set_experiment(experiment_name=experiment_name)

    # Logging to MLflow
    with mlflow.start_run():
        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model details
        model_details = {
            "model_name": model_name,
            "scaler": "StandardScaler",
            "undersample_level": undersample_level,
            "oversample_level": oversample_level,
        }
        mlflow.log_params(model_details)
        mlflow.log_params(model_params)

        # Log confusion matrix, and eventual other figures as artifact
        for name, fig in artifacts.items():
            mlflow.log_figure(fig, f"{name}.png")

        # Log Hydra config files as artifacts
        hydra_configs = [f for f in hydra_cfg_dir.iterdir() if f.suffix == ".yaml"]

        # Log paths of Hydra config files as MLflow parameters
        hydra_cfg_paths = []
        for config_file in hydra_configs:
            mlflow.log_artifact(str(config_file), artifact_path="hydra_configs")
            hydra_cfg_paths.append(str(config_file))

        # Log paths as MLflow parameters
        hydra_cfg_path_str = ", ".join(hydra_cfg_paths)
        mlflow.log_param("hydra_config_paths", hydra_cfg_path_str)


def log_mlflow_optimisation_trial(
    mlflow_path: Path,
    experiment_name: str,
    trial_number: int,
    metric_name: str,
    metric_value: float,
    model_name: str,
    model_params: dict,
    undersample_level: int,
    oversample_level: int,
) -> None:
    """Log a single optimisation trial to MLflow without artifacts.

    Parameters
    ----------
    mlflow_path : Path
        Tracking URI path for MLflow.
    experiment_name : str
        MLflow experiment name (e.g., "optimisation").
    trial_number : int
        The Optuna trial number.
    metric_name : str
        Name of the metric being optimized (e.g., "balanced_accuracy").
    metric_value : float
        The value of the metric for this trial.
    model_name : str
        The model name (e.g., "xgboost_native").
    model_params : dict
        The hyperparameters evaluated in this trial.
    undersample_level : int
        Undersampling level used in the pipeline.
    oversample_level : int
        Oversampling level used in the pipeline.
    """
    mlflow.set_tracking_uri(str(mlflow_path))
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=f"trial_{trial_number}"):
        mlflow.log_metric(metric_name, metric_value)
        mlflow.log_param("trial_number", trial_number)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("undersample_level", undersample_level)
        mlflow.log_param("oversample_level", oversample_level)
        # Log the hyperparameters evaluated in the trial
        if model_params:
            mlflow.log_params(model_params)
