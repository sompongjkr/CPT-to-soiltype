from cpt_to_soiltype.preprocess_funcs import get_dataset, choose_features, drop_na, drop_duplicates, remove_outliers_hardcoded, remove_outliers_multivariate, remove_outliers_univariate, split_drillhole_data
from cpt_to_soiltype.utility import info_dataset
from cpt_to_soiltype.schema_config import Config
import pandas as pd
from rich.console import Console
from rich.pretty import pprint
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def preprocess_data(cfg: DictConfig) -> pd.DataFrame:
    """Preprocess dataset."""
    pcfg = Config(**OmegaConf.to_object(cfg))

    df = get_dataset(pcfg.dataset.path_raw_dataset)
    pprint("Dataset loaded")
    df = choose_features(df, features=pcfg.experiment.features + pcfg.experiment.site_info + [pcfg.experiment.label])
    df = drop_na(df)
    pprint("NA values dropped")
    if pcfg.preprocess.remove_duplicates:
        df = drop_duplicates(df, pcfg.experiment.features)
        pprint("Duplicates dropped")
    if pcfg.preprocess.remove_outliers_hard:
        df = remove_outliers_hardcoded(df)
        pprint("Hardcoded outliers removed")
    if pcfg.preprocess.remove_outliers_uni:
        df = remove_outliers_univariate(df, pcfg.preprocess.outlier_feature, threshold=pcfg.preprocess.univariate_threshold)
        pprint("Univariate outliers removed")
    if pcfg.preprocess.remove_outliers_multi:
        df = remove_outliers_multivariate(df, pcfg.experiment.features, confidence_threshold=pcfg.preprocess.multivariate_threshold)
        pprint("Multivariate outliers removed")
    return df


if __name__ == '__main__':

    console = Console()

    # PREPROCESSING
    ################################
    df: pd.DataFrame = preprocess_data()
    console.print(df.head())
    df.to_csv('data/model_ready/dataset_total.csv', index=False)

    train_df, test_df = split_drillhole_data(df, id_column="ID", train_fraction=0.75)
    train_df.to_csv('data/model_ready/dataset_train.csv', index=False)
    test_df.to_csv('data/model_ready/dataset_test.csv', index=False)
    info_dataset(df, train_df, test_df, label="Oberhollenzer_classes")
