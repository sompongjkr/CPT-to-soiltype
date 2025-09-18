from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from cpt_to_soiltype.hyperparameter_optimisation import run_optimisation
from cpt_to_soiltype.preprocess_funcs import get_dataset, split_drillhole_data
from cpt_to_soiltype.schema_config import Config
from cpt_to_soiltype.train_eval_funcs import log_mlflow_optimisation_trial
from cpt_to_soiltype.utility import get_custom_console


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pcfg = Config(**OmegaConf.to_object(cfg))
    console = get_custom_console()
    console.print(pcfg)
    pcfg.mlflow.experiment_name = "train_test"

    console.print(
        f"Running hyperparameter optimisation of XGBoost model, with {pcfg.optuna.n_trials} trials",
        style="info",
    )

    # Load data - IMPORTANT: only use the model_ready TRAIN set here.
    # We'll split it into TRAIN and VALIDATION for hyperparameter optimisation.
    # The held-out TEST set is reserved exclusively for scripts/train.py.
    console.print("Loading model_ready training dataset...", style="info")
    df_train_full = get_dataset(pcfg.dataset.path_model_ready_train)

    console.print(
        "Splitting TRAIN into TRAIN and VALIDATION (no TEST used here)...",
        style="info",
    )
    df_train, df_val = split_drillhole_data(
        df_train_full, id_column="ID", train_fraction=pcfg.experiment.train_fraction
    )

    # Clear naming for readability
    X_train = df_train[pcfg.experiment.features]
    y_train = df_train[pcfg.experiment.label]

    X_val = df_val[pcfg.experiment.features]
    y_val = df_val[pcfg.experiment.label]

    # optimally run using cross validation
    study = run_optimisation(
        X_train,
        X_val,  # validation set used as Optuna evaluation set
        y_train,
        y_val,
        pcfg.experiment.oversample_level,
        pcfg.experiment.undersample_level,
        n_trials=pcfg.optuna.n_trials,
    )

    console.rule("Study statistics")
    console.print("Number of finished trials: ", len(study.trials))

    console.print("Best trial:")
    trial = study.best_trial

    console.print("Trial number: \t", trial.number)
    console.print("Balanced accuracy: \t", trial.value)

    console.print("Params: ")
    for key, value in trial.params.items():
        console.print(f"{key}: {value}")

    # Save best parameters, per-trial metrics, and an optimisation history plot
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    optim_experiment_name = f"optimisation_{timestamp}"

    # Use original project root to avoid Hydra CWD surprises
    base_dir = Path(get_original_cwd())

    # 1) Save best params YAML
    save_dir = base_dir / "experiments" / "hyperparameters"
    save_dir.mkdir(parents=True, exist_ok=True)
    yaml_filename = save_dir / f"best_hyperparameters_xgboost_{timestamp}.yaml"
    with open(yaml_filename, "w") as yaml_file:
        yaml.dump(trial.params, yaml_file)
    console.print(f"Best hyperparameters saved to {yaml_filename}", style="success")

    # 2) Save metrics from all trials to CSV
    trials_completed = [t for t in study.trials if t.value is not None]
    trials_completed.sort(key=lambda t: t.number)
    metrics_df = pd.DataFrame(
        {
            "trial_number": [t.number for t in trials_completed],
            "objective_value": [t.value for t in trials_completed],
        }
    )
    csv_path = save_dir / f"optimisation_metrics_{timestamp}.csv"
    metrics_df.to_csv(csv_path, index=False)
    console.print(
        f"Saved trial metrics ({len(metrics_df)} rows) to {csv_path}",
        style="success",
    )

    # 3) Plot optimisation history and save PNG
    plots_dir = base_dir / "plots" / "optimisation"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plots_dir / f"optimisation_history_{timestamp}.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        metrics_df["trial_number"],
        metrics_df["objective_value"],
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=4,
        color="#1f77b4",
    )
    plt.title(
        f"Optuna optimisation history - Balanced accuracy ({len(metrics_df)} trials)"
    )
    plt.xlabel("Trial number")
    plt.ylabel("Balanced accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    console.print(f"Saved optimisation history plot to {fig_path}", style="success")

    # 4) Log each trial to MLflow under a unique experiment name per run
    console.print(
        f"Logging trial metrics to MLflow (experiment: {optim_experiment_name})...",
        style="info",
    )
    for t in trials_completed:
        log_mlflow_optimisation_trial(
            mlflow_path=Path(pcfg.mlflow.path),
            experiment_name=optim_experiment_name,
            trial_number=t.number,
            metric_name="balanced_accuracy",
            metric_value=float(t.value),
            model_name=pcfg.model.name,
            model_params=t.params,
            undersample_level=pcfg.experiment.undersample_level,
            oversample_level=pcfg.experiment.oversample_level,
        )
    console.print("Finished logging optimisation trials to MLflow", style="success")


if __name__ == "__main__":
    main()
