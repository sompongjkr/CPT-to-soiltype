from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.traceback import install

from cpt_to_soiltype.feature_selection import (
    save_selected_features,
    select_features_with_featurewiz,
)
from cpt_to_soiltype.schema_config import Config
from cpt_to_soiltype.utility import get_custom_console


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pcfg = Config(**OmegaConf.to_object(cfg))
    console = get_custom_console()
    console.print("Starting Feature Selection with Featurewiz", style="info")

    # Inputs
    train_csv = Path(pcfg.dataset.path_model_ready_train)
    test_csv = Path(pcfg.dataset.path_model_ready_test)
    target_column = pcfg.experiment.label
    feature_columns = pcfg.experiment.features

    # Tunables (could be exposed via Hydra overrides)
    corr_limit: float = 0.7
    verbose: int = 1
    feature_engg: str | None = None
    category_encoders: str | None = None

    selected = select_features_with_featurewiz(
        train_csv=train_csv,
        test_csv=test_csv,
        target_column=target_column,
        feature_columns=feature_columns,
        corr_limit=corr_limit,
        verbose=verbose,
        feature_engg=feature_engg,
        category_encoders=category_encoders,
    )
    console.print(f"Selected {len(selected)} features:", style="success")
    console.print(selected)

    # Save to experiments/selected_features
    # Decide output directory: project requirement is experiments/selected_features
    # Hydra changes working dirs, so anchor at project root using cwd parents
    project_root = Path.cwd()
    # If invoked under Hydra, cwd is the run dir; project root should be original working dir's parent if 'outputs' in path.
    # To be robust, detect repository root by presence of pyproject.toml
    for p in [project_root] + list(project_root.parents):
        if (p / "pyproject.toml").exists():
            project_root = p
            break
    out_dir = project_root / "experiments" / "selected_features"
    saved = save_selected_features(
        selected_features=selected,
        out_dir=out_dir,
        corr_limit=corr_limit,
        label=target_column,
        extra_tag="featurewiz",
    )
    console.print(f"Saved selected features to: {saved}", style="info")


if __name__ == "__main__":
    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
