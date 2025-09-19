from collections.abc import Callable

import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.theme import Theme


def load_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    # Load the dataset from the specified path
    df = pd.read_csv(file_path)

    # Clean column names to avoid issues with whitespace or special characters
    df.columns = df.columns.str.strip()

    # Define selected features
    SELECTED_FEATURES = [
        "Depth (m)",
        "qc (MPa)",
        "fs (kPa)",
        "Rf (%)",
        "σ,v (kPa)",
        "u0 (kPa)",
        "σ',v (kPa)",
        "Qtn (-)",
        "Fr (%)",
    ]

    # Extract the features
    features = df[SELECTED_FEATURES].copy()

    # Extract labels if the column exists
    labels = None
    if "Oberhollenzer_classes" in df.columns:
        labels = df["Oberhollenzer_classes"].copy()

    return df, features, labels


# Loading the saved model for future use
def load_xgb_model(model_path: str) -> xgb.Booster:
    """Load an XGBoost Booster from json or ubj.

    Tries the provided path first; if it fails and the extension is .json or .ubj,
    tries the alternate extension in the same folder.
    """
    loaded_model = xgb.Booster()
    try:
        loaded_model.load_model(model_path)
        return loaded_model
    except Exception:
        # Attempt fallback between .json and .ubj
        if model_path.endswith(".json"):
            alt = model_path[:-5] + ".ubj"
        elif model_path.endswith(".ubj"):
            alt = model_path[:-4] + ".json"
        else:
            raise
        loaded_model.load_model(alt)
        return loaded_model


def track_sample_num(func: Callable) -> Callable:
    """Tracking number of samples of a dataframe before and after processing."""
    console = Console()

    def df_processing(*args: int, **kwargs: int) -> pd.DataFrame:
        res = func(*args, **kwargs)
        for data in args:
            if isinstance(data, pd.DataFrame):
                console.print("--------------------------------")
                console.print(
                    f"Number of samples before processing with {func.__name__} function"
                    f" (rows,cols): {data.shape}"
                )
                console.print(
                    f"Number of samples after processing with {func.__name__} function"
                    f" (rows,cols): {res.shape}"
                )
                console.print("--------------------------------")
        return res

    return df_processing


def info_dataset(
    df_main: pd.DataFrame,
    label: str,
    *,
    df_test: pd.DataFrame | None = None,
    df_train: pd.DataFrame | None = None,
    only_features: bool = False,
) -> None:
    """Print info about dataset."""
    console = Console()
    console.rule()
    console.print("\nFirst five rows:")
    console.print(df_main.head())
    console.rule()
    console.print(df_main.info())
    console.rule()
    console.print(
        f"\nA fantastic dataset of {df_main.shape[0]} samples is built :smiley:"
    )
    if not only_features:
        console.print(f"Num samples trainset: {df_train.shape[0]}")
        console.print(f"Num samples testset: {df_test.shape[0]}")
        console.rule()
        # value counts train
        console.print("\nValue counts trainset:")
        console.print(df_train[label].value_counts())
        # value counts test
        console.print("\nValue counts testset:")
        console.print(df_test[label].value_counts())


def get_custom_console() -> Console:
    """
    Returns a custom console with a custom theme.

    Returns:
        Console: Rich Console object with custom theme.

    """
    custom_theme = Theme(
        {
            "info": "bold green",
            "warning": "yellow",
            "danger": "bold red",
            "error": "bold magenta",
            "success": "bold blue",
        }
    )
    return Console(theme=custom_theme)
