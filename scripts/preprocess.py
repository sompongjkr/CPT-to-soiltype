import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from cpt_to_soiltype.preprocess_funcs import preprocess_data, split_drillhole_data
from cpt_to_soiltype.schema_config import Config
from cpt_to_soiltype.utility import info_dataset


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pcfg = Config(**OmegaConf.to_object(cfg))
    console = Console()
    console.print(pcfg)

    # DATASET USED IN SUPERVISED LEARNING
    ##################################################################

    df: pd.DataFrame = preprocess_data(
        path_file=pcfg.dataset.path_raw_dataset,
        features=pcfg.experiment.features,
        site_features=pcfg.experiment.site_info,
        labels=pcfg.experiment.label,
        outlier_feature=pcfg.preprocess.outlier_feature,
        remove_duplicates=pcfg.preprocess.remove_duplicates,
        remove_outliers_hard=pcfg.preprocess.remove_outliers_hard,
        remove_outliers_uni=pcfg.preprocess.remove_outliers_uni,
        remove_outliers_multi=pcfg.preprocess.remove_outliers_multi,
        univariate_threshold=pcfg.preprocess.univariate_threshold,
        multivariate_threshold=pcfg.preprocess.multivariate_threshold,
    )

    df.to_csv(pcfg.dataset.path_model_ready_dataset, index=False)

    train_df, test_df = split_drillhole_data(
        df,
        id_column="ID",
        train_fraction=pcfg.experiment.train_fraction,
        random_state=pcfg.experiment.seed,
    )
    train_df.to_csv(pcfg.dataset.path_model_ready_train, index=False)
    test_df.to_csv(pcfg.dataset.path_model_ready_test, index=False)
    info_dataset(df, label="Oberhollenzer_classes", df_train=train_df, df_test=test_df)

    # DATASET WITH FEATURES ONLY (NO LABELS) - FOR UNSUPERVISED LEARNING
    ##################################################################

    df_features_only: pd.DataFrame = preprocess_data(
        path_file=pcfg.dataset.path_raw_dataset,
        features=pcfg.experiment.features,
        outlier_feature=pcfg.preprocess.outlier_feature,
        remove_duplicates=pcfg.preprocess.remove_duplicates,
        remove_outliers_hard=pcfg.preprocess.remove_outliers_hard,
        remove_outliers_uni=pcfg.preprocess.remove_outliers_uni,
        remove_outliers_multi=pcfg.preprocess.remove_outliers_multi,
        univariate_threshold=pcfg.preprocess.univariate_threshold,
        multivariate_threshold=pcfg.preprocess.multivariate_threshold,
    )

    df_features_only.to_csv(pcfg.dataset.path_model_ready_features, index=False)
    info_dataset(df_features_only, label=pcfg.experiment.label, only_features=True)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
