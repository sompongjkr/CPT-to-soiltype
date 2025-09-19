from pathlib import Path
from typing import Any

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


def determine_num_classes(y: pd.Series) -> int:
    """Robustly determine number of unique classes in a label vector."""
    try:
        return int(pd.Series(y).nunique())
    except Exception:
        # Fallback if pandas isn't imported here; compute via set
        return len(set(y.tolist() if hasattr(y, "tolist") else y))


def _find_project_root() -> Path:
    """Find repository root by looking for pyproject.toml upwards from CWD."""
    project_root = Path.cwd()
    for p in [project_root] + list(project_root.parents):
        if (p / "pyproject.toml").exists():
            return p
    return project_root


def _with_cls_suffix(path: Path, num_classes: int) -> Path:
    stem, suffix = path.stem, path.suffix
    return path.with_name(f"{stem}_{num_classes}cls{suffix}")


def build_model_save_paths(
    json_path_str: str, ubj_path_str: str, num_classes: int
) -> list[str]:
    """Construct fully-qualified model save paths (JSON and UBJ) with class suffix.

    Ensures parent directories exist and paths are anchored to the repo root
    if provided as relative.
    """
    root = _find_project_root()

    def _resolve(path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (root / p)

    json_p = _with_cls_suffix(_resolve(json_path_str), num_classes)
    ubj_p = _with_cls_suffix(_resolve(ubj_path_str), num_classes)

    for p in (json_p, ubj_p):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    return [str(json_p), str(ubj_p)]


def print_saved_model_summary(
    paths: list[str], num_classes: int, *, console: Any | None = None
) -> None:
    """Print a concise summary of saved model files."""
    msg = f"Saved trained model for {num_classes} classes to:\n" + "\n".join(
        [f"  - {p}" for p in paths]
    )
    try:
        if console is not None:
            console.print(msg, style="success")
        else:
            print(msg)
    except Exception:
        print(msg)


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


def transform_labels(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    labels_to_exclude: list[int] | None = None,
    label_groups: list[list[int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Exclude specific labels and/or group labels into new classes.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices for train and test splits.
    y_train, y_test : pd.Series
        Label vectors for train and test splits (numeric labels expected).
    labels_to_exclude : list | None, default None
        Labels to drop from both train and test. Rows with these labels will be removed.
    label_groups : list[tuple] | None, default None
        A list of tuples, where each tuple contains labels to be merged into a new label.
        Example: [(0, 2), (3, 5, 6)] will merge 0 and 2 into a new class, and 3, 5, 6 into another new class.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        The transformed datasets with indices reset for consistency.
    """
    # Work on copies to avoid side-effects
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    y_train_out = y_train.copy()
    y_test_out = y_test.copy()

    # 1) Exclude selected labels (apply consistently to X and y)
    if labels_to_exclude:
        mask_train = ~y_train_out.isin(labels_to_exclude)
        mask_test = ~y_test_out.isin(labels_to_exclude)

        X_train_out = X_train_out.loc[mask_train].reset_index(drop=True)
        y_train_out = y_train_out.loc[mask_train].reset_index(drop=True)
        X_test_out = X_test_out.loc[mask_test].reset_index(drop=True)
        y_test_out = y_test_out.loc[mask_test].reset_index(drop=True)

    # 2) Group labels into new classes
    if label_groups:
        # Determine a starting point for new labels; assume numeric labels
        combined_labels = pd.concat([y_train_out, y_test_out], ignore_index=True)
        combined_numeric = pd.to_numeric(combined_labels, errors="coerce")
        max_label = (
            int(pd.Series(combined_numeric).max())
            if not combined_numeric.isna().all()
            else -1
        )

        label_mapping: dict = {}
        for idx, group in enumerate(label_groups, start=1):
            new_label = max_label + idx
            for original_label in group:
                label_mapping[original_label] = new_label

        if label_mapping:
            y_train_out = y_train_out.replace(label_mapping)
            y_test_out = y_test_out.replace(label_mapping)

        # Ensure resulting labels are numeric dtype where possible without deprecated behavior
        def _safe_to_numeric(series: pd.Series) -> pd.Series:
            try:
                return pd.to_numeric(series)
            except Exception:
                return series

        y_train_out = _safe_to_numeric(y_train_out)
        y_test_out = _safe_to_numeric(y_test_out)

    return X_train_out, X_test_out, y_train_out, y_test_out


def compute_group_id_mapping(
    y_train: pd.Series, y_test: pd.Series, label_groups: list[list[int]]
) -> dict[int, int]:
    """Compute a mapping from original labels (in groups) to new group IDs.

    New IDs are assigned deterministically as max(existing_label)+1, +2, ...
    in the order of label_groups.
    """
    combined_labels = pd.concat([y_train, y_test], ignore_index=True)
    combined_numeric = pd.to_numeric(combined_labels, errors="coerce")
    max_label = (
        int(pd.Series(combined_numeric).max())
        if not combined_numeric.isna().all()
        else -1
    )

    mapping: dict[int, int] = {}
    for idx, group in enumerate(label_groups, start=1):
        new_label = max_label + idx
        for original_label in group:
            mapping[original_label] = new_label
    return mapping


def augment_class_mapping_with_groups(
    class_mapping: dict[int, str],
    group_id_mapping: dict[int, int],
    label_group_names: list[str] | None,
) -> dict[int, str]:
    """Return a new class_mapping including names for the grouped classes.

    If label_group_names is provided, it will be assigned to the new group IDs
    in the order those IDs were created; otherwise numeric strings are used.
    """
    updated = dict(class_mapping) if class_mapping is not None else {}

    # Determine unique new IDs in stable order by sorting by value then key
    new_ids_ordered = []
    for _, new_id in sorted(group_id_mapping.items(), key=lambda kv: (kv[1], kv[0])):
        if new_id not in new_ids_ordered:
            new_ids_ordered.append(new_id)

    for i, new_id in enumerate(new_ids_ordered):
        if label_group_names and i < len(label_group_names):
            updated[new_id] = label_group_names[i]
        else:
            updated[new_id] = str(new_id)
    return updated


def xgb_native_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_params: dict,
    oversample_level: int,
    undersample_level: int,
    model_save_path: str = "models/xgb_model.json",  # Backward-compat single path
    save_model: bool = False,
    model_save_paths: list[str] | None = None,  # New: optionally save to multiple paths
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
        # Always ensure at least the legacy single path is saved
        try:
            xgb_model.save_model(model_save_path)
            pprint(f"Model saved at {model_save_path}")
        except Exception as e:
            pprint(f"Warning: failed saving model to {model_save_path}: {e}")

        # And save to any additional provided paths (e.g., UBJ)
        if model_save_paths:
            for p in model_save_paths:
                try:
                    xgb_model.save_model(p)
                    pprint(f"Model saved at {p}")
                except Exception as e:
                    pprint(f"Warning: failed saving model to {p}: {e}")

    # Make predictions (output is class indices for multi:softmax); map back to original labels
    y_pred_indices = xgb_model.predict(dtest)
    # Ensure integer indices for mapping
    y_pred_indices = pd.Series(y_pred_indices).astype(int)
    y_pred = y_pred_indices.map(idx_to_label).to_numpy()
    return y_pred


def train_eval(
    model_name: str,
    model_params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    undersample_level: int,
    oversample_level: int,
    save_model: bool = False,
    *,
    model_save_path: str | None = None,
    model_save_paths: list[str] | None = None,
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
            model_save_path=(
                model_save_path if model_save_path else "models/xgb_model.json"
            ),
            model_save_paths=model_save_paths,
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
    num_classes: int,
    class_mapping: dict[int, str] | None = None,
    model_artifact_paths: list[str] | None = None,
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
            "num_classes": num_classes,
        }
        mlflow.log_params(model_details)
        mlflow.log_params(model_params)

        # Log confusion matrix, and eventual other figures as artifact
        for name, fig in artifacts.items():
            mlflow.log_figure(fig, f"{name}.png")

        # Optionally log class mapping (IDs to names) for transparency/comparability
        if class_mapping:
            # Log a few entries as params (compact), and the full mapping as JSON artifact
            preview_items = list(class_mapping.items())[:10]
            for k, v in preview_items:
                mlflow.log_param(f"class_{k}", v)
            # Log the full mapping directly as an artifact in the run store
            mlflow.log_dict(
                {int(k): v for k, v in class_mapping.items()},
                artifact_file="metadata/class_mapping.json",
            )

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

        # Optionally log local model files as MLflow artifacts
        if model_artifact_paths:
            for p in model_artifact_paths:
                try:
                    mlflow.log_artifact(p, artifact_path="model")
                except Exception:
                    # Best-effort: skip if cannot log
                    pass


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
