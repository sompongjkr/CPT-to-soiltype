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
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
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

    # Convert labels to integers to ensure consistency
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

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

    # Adjust labels to start from 0 (required for XGBoost with multiclass)
    y_train = y_train_final - 1
    y_test = y_test - 1

    # Convert the balanced training data to DMatrix (with GPU support)
    dtrain = xgb.DMatrix(X_train_final, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set default parameters for XGBoost if not provided
    params = {
        "num_class": len(y_train.unique()),  # Number of classes
    }
    model_params.update(params)

    # Train the XGBoost model
    xgb_model = xgb.train(model_params, dtrain, num_boost_round=100)

    # optionally save the model
    if save_model:
        xgb_model.save_model(model_save_path)
        pprint(f"Model saved at {model_save_path}")

    # Make predictions (output is directly class labels)
    y_pred = xgb_model.predict(dtest)
    y_pred = y_pred + 1
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
