from cpt_to_soiltype.preprocess_funcs import get_dataset, choose_features, drop_na, drop_duplicates, remove_outliers_hardcoded, remove_outliers_multivariate, remove_outliers_univariate, split_drillhole_data
from cpt_to_soiltype.utility import info_dataset
import pandas as pd
from rich.console import Console
from rich.pretty import pprint


def preprocess_data(path_file: str, features: list, site_features: list, labels: list, outlier_feature: str, remove_duplicates: bool, remove_outliers_hard: bool, remove_outliers_uni: bool, remove_outliers_multi: bool, uni_threshold: int=3, multi_threshold=0.95) -> pd.DataFrame:
    """Preprocess dataset."""
    df = get_dataset(path_file)
    pprint("Dataset loaded")
    df = choose_features(df, features=features+site_features+labels)
    df = drop_na(df)
    pprint("NA values dropped")
    if remove_duplicates:
        df = drop_duplicates(df, features)
        pprint("Duplicates dropped")
    if remove_outliers_hard:
        df = remove_outliers_hardcoded(df)
        pprint("Hardcoded outliers removed")
    if remove_outliers_uni:
        df = remove_outliers_univariate(df, outlier_feature, threshold=uni_threshold)
        pprint("Univariate outliers removed")
    if remove_outliers_multi:
        df = remove_outliers_multivariate(df, features, confidence_threshold=multi_threshold)
        pprint("Multivariate outliers removed")
    return df


if __name__ == '__main__':

    # INPUTS
    ###########################################
    path_file = 'data/raw/mmc1.csv'
    FEATURES = ['Depth (m)', 'qc (MPa)', 'fs (kPa)','Rf (%)', 'σ,v (kPa)', 'u0 (kPa)',"σ',v (kPa)", 'Qtn (-)', 'Fr (%)']
    SITE_INFO = ['ID', 'test_type', 'basin_valley']
    LABELS_O = ['Oberhollenzer_classes']
    features = FEATURES + SITE_INFO + LABELS_O
    outlier_feature = 'qc (MPa)'
    remove_duplicates = True
    remove_outliers_hard = True
    remove_outliers_uni = False
    remove_outliers_multi = True
    uni_threshold = 3
    multi_threshold = 0.95
    label = "Oberhollenzer_classes"

    console = Console()

    # PREPROCESSING
    ################################
    df = preprocess_data(path_file, FEATURES, SITE_INFO, LABELS_O,outlier_feature, remove_duplicates, remove_outliers_hard, remove_outliers_uni, remove_outliers_multi, uni_threshold, multi_threshold)
    console.print(df.head())
    df.to_csv('data/model_ready/dataset_total.csv', index=False)

    train_df, test_df = split_drillhole_data(df, id_column="ID", train_fraction=0.75)
    train_df.to_csv('data/model_ready/dataset_train.csv', index=False)
    test_df.to_csv('data/model_ready/dataset_test.csv', index=False)
    info_dataset(df, train_df, test_df, label=label)
