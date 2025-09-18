import os
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig  # Import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.traceback import install

from cpt_to_soiltype.schema_config import Config
from cpt_to_soiltype.train_eval_funcs import (
    augment_class_mapping_with_groups,
    compute_group_id_mapping,
    evaluate_model,
    load_data,
    log_mlflow_metrics_and_model,
    train_eval,
    transform_labels,
)
from cpt_to_soiltype.utility import get_custom_console


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    To rerun an experiment, use the following command:

    python <script_name>.py \
    --config-path <path_to_config> \
    --config-name <config_name> \
    +hydra=<hydra_config> \
    +overrides=<overrides_config>

    Example:
    python train.py \
    --config-path ./experiments/hydra_outputs/2024-11-12/14-25-36 \
    --config-name config \
    +hydra=@hydra.yaml \
    +overrides=@overrides.yaml
    """
    pcfg = Config(**OmegaConf.to_object(cfg))
    console = get_custom_console()
    console.print(pcfg)
    pcfg.mlflow.experiment_name = "train_test"

    # Load data
    console.print("Loading training and testing data...", style="info")
    X_train, X_test, y_train, y_test = load_data(
        pcfg.dataset.path_model_ready_train,
        pcfg.dataset.path_model_ready_test,
        pcfg.experiment.label,
        pcfg.experiment.features,
    )

    # If grouping is enabled, compute new group IDs BEFORE transforming labels so names match IDs
    pre_group_id_map = None
    if pcfg.experiment.label_groups:
        pre_group_id_map = compute_group_id_mapping(
            y_train, y_test, pcfg.experiment.label_groups
        )

    # Optionally transform labels: exclude rare classes and/or group classes via config
    X_train, X_test, y_train, y_test = transform_labels(
        X_train,
        X_test,
        y_train,
        y_test,
        labels_to_exclude=pcfg.experiment.labels_to_exclude,
        label_groups=pcfg.experiment.label_groups,
    )

    # Train model and predict output
    console.print("Train and evaluate...", style="info")
    y_pred = train_eval(
        pcfg.model.name,
        pcfg.model.params,
        X_train,
        X_test,
        y_train,
        y_test,
        pcfg.experiment.undersample_level,
        pcfg.experiment.oversample_level,
        pcfg.experiment.save_model,
    )

    # Evaluate model
    console.print("Evaluating model...", style="info")
    # If we grouped labels, augment class mapping with names for the new groups
    class_mapping = pcfg.experiment.soil_classification
    if pcfg.experiment.label_groups and pre_group_id_map is not None:
        class_mapping = augment_class_mapping_with_groups(
            class_mapping, pre_group_id_map, pcfg.experiment.label_group_names
        )

    metrics, artifacts = evaluate_model(y_test, y_pred, class_mapping)

    # log results and config to mlflow
    console.print("Logging data to MLflow...", style="info")
    hydra_cfg_dir = Path(HydraConfig.get().run.dir) / ".hydra"
    # Number of classes used for training (post-transform)
    try:
        num_classes = int(pd.Series(y_train).nunique())
    except Exception:
        # Fallback if pandas isn't imported here; compute via set
        num_classes = len(
            set(y_train.tolist() if hasattr(y_train, "tolist") else y_train)
        )
    log_mlflow_metrics_and_model(
        pcfg.mlflow.path,
        pcfg.mlflow.experiment_name,
        metrics,
        artifacts,
        pcfg.model.name,
        pcfg.model.params,
        pcfg.experiment.undersample_level,
        pcfg.experiment.oversample_level,
        hydra_cfg_dir,
        num_classes,
        class_mapping,
    )


if __name__ == "__main__":
    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"
    console = Console()
    main()
