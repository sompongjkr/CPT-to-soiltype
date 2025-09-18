from pathlib import Path

import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.mad import MAD
from rich.pretty import pprint
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from cpt_to_soiltype.utility import track_sample_num


def get_dataset(path_file: Path) -> pd.DataFrame:
    """Read dataset."""
    df = pd.read_csv(path_file, header=0, sep=",")
    return df


@track_sample_num
def choose_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Choose features for dataset."""
    df = df[features]
    return df


@track_sample_num
def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NA values."""
    df = df.dropna()
    return df


@track_sample_num
def drop_duplicates(df: pd.DataFrame, duplicate_features: list[str]) -> pd.DataFrame:
    """Drop duplicated rows."""
    df = df.drop_duplicates(subset=duplicate_features)
    return df


@track_sample_num
def remove_outliers_hardcoded(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers based on hardcoded values."""
    df = df[df["qc (MPa)"] > 0]
    df = df[df["u0 (kPa)"] >= 0]
    df = df[df["Qtn (-)"] > 0]
    df = df[(df["fs (kPa)"] < 1200) & (df["fs (kPa)"] > 0)]
    # df = df[(df['Rf (%)'] < 10) & (df['Rf (%)'] > 0)]
    df = df[df["Rf (%)"] > 0]
    df = df[(df["Fr (%)"] < 10) & (df["Fr (%)"] > 0)]
    # skip samples with label 3.0
    # df = df[df['Oberhollenzer_classes'] != 3.0] # due to low sample size
    # df = df[df["Oberhollenzer_classes"] != 0.0]  # due to low sample size
    return df


@track_sample_num
def remove_outliers_univariate(
    df: pd.DataFrame, feature: str, threshold: float
) -> pd.DataFrame:
    """
    Removes outliers from a dataframe based on the MAD (Median Absolute Deviation) method.
    """
    # Initialize the MAD model with the provided threshold
    mad = MAD(threshold=threshold)

    # Fit the model on the specified feature
    mad.fit(df[[feature]])

    # Predict outliers (1 for outlier, 0 for inlier)
    outliers = mad.predict(df[[feature]])

    # Filter the DataFrame to exclude outliers
    df_no_outliers = df[outliers == 0]

    return df_no_outliers


@track_sample_num
def remove_outliers_multivariate(
    df: pd.DataFrame, features: list[str], confidence_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame using the Isolation Forest model.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list[str]): List of feature column names to consider for outlier detection.
        confidence_threshold (float): The threshold for outlier confidence. Defaults to 0.95.

    Returns:
        pd.DataFrame: A DataFrame excluding detected outliers.
    """
    # Initialize and fit the Isolation Forest model
    iforest = IForest(n_estimators=100)
    iforest.fit(df[features])

    # Get the outlier probabilities
    probs = iforest.predict_proba(df[features])[:, 1]

    # Create a mask for outliers based on the confidence threshold
    is_outlier = probs > confidence_threshold

    # Display results
    outliers = df[is_outlier]
    num_outliers = len(outliers)
    print(f"Number of outliers with Isolation Forest: {num_outliers}")
    print(f"Percentage of outliers: {num_outliers / len(df):.4f}")
    print("Outlier samples:\n", outliers)

    # Return DataFrame excluding outliers
    return df[~is_outlier]


def preprocess_data(
    path_file: str,
    features: list[str],
    site_features: list[str] | None = None,
    labels: str | None = None,
    *,
    # Defaults aligned with scripts/config/main.yaml
    outlier_feature: str = "qc (MPa)",
    remove_duplicates: bool = True,
    remove_outliers_hard: bool = True,
    remove_outliers_uni: bool = False,
    remove_outliers_multi: bool = False,
    univariate_threshold: int = 3,
    multivariate_threshold: float = 0.5,
) -> pd.DataFrame:
    """Preprocess dataset.

    Required:
        path_file: Path to CSV dataset.
        features: Feature column names to keep for modeling and duplicate checks.

    Optional:
        site_features: Columns with site/drillhole metadata to retain. If None, none are added.
        labels: Target column name to include. If None, label is not included in the output DataFrame.

    Other parameters default to values from Hydra config (scripts/config/main.yaml).
    """
    df = get_dataset(path_file)
    pprint("Dataset loaded")
    # Build selected columns while allowing None for site_features/labels
    selected_cols: list[str] = []
    if site_features:
        selected_cols.extend(site_features)
    selected_cols.extend(features)
    if labels:
        selected_cols.append(labels)

    df = choose_features(df, features=selected_cols)
    df = drop_na(df)
    pprint("NA values dropped")
    if remove_duplicates:
        df = drop_duplicates(df, features)
        pprint("Duplicates dropped")
    if remove_outliers_hard:
        df = remove_outliers_hardcoded(df)
        pprint("Hardcoded outliers removed")
    if remove_outliers_uni:
        # Only attempt univariate outlier removal if the feature exists
        if outlier_feature in df.columns:
            df = remove_outliers_univariate(
                df, outlier_feature, threshold=univariate_threshold
            )
            pprint("Univariate outliers removed")
        else:
            pprint(
                f"Skip univariate outlier removal: '{outlier_feature}' not in columns"
            )
    if remove_outliers_multi:
        df = remove_outliers_multivariate(
            df, features, confidence_threshold=multivariate_threshold
        )
        pprint("Multivariate outliers removed")
    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing")
    return df


def split_drillhole_data(
    df: pd.DataFrame,
    id_column: str,
    train_fraction: float = 0.75,
    random_state: int = 42,
):
    """
    Splits the DataFrame into training and testing sets, keeping drillhole data intact.

    Args:
        df (pd.DataFrame): The input DataFrame containing drillhole data.
        id_column (str): The column name that holds the drillhole ID.
        train_fraction (float): The fraction of data to use for training. Defaults to 0.75.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
    """
    gss = GroupShuffleSplit(
        n_splits=1, train_size=train_fraction, random_state=random_state
    )
    groups = df[id_column]
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def split_drillhole_data_manual_implementation(
    df: pd.DataFrame, id_column: str, train_fraction: float = 0.75
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and testing sets, keeping drillhole data intact.

    Args:
        df (pd.DataFrame): The input DataFrame containing drillhole data.
        id_column (str): The column name that holds the drillhole ID.
        train_fraction (float): The fraction of data to use for training. Defaults to 0.75.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
    """
    # Get the unique drillhole IDs
    unique_ids = df[id_column].unique()

    # Randomly sample a fraction of IDs for the training set
    train_ids, _ = train_test_split(
        unique_ids, train_size=train_fraction, random_state=42
    )

    # Split the dataset based on the sampled IDs
    train_df = df[df[id_column].isin(train_ids)]
    test_df = df[~df[id_column].isin(train_ids)]

    return train_df, test_df
