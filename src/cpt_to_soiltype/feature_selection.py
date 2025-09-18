from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console

from cpt_to_soiltype.utility import get_custom_console


def _load_and_filter(
    train_csv: str | Path,
    test_csv: str | Path,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs and keep only selected features + target.

    Parameters
    ----------
    train_csv : str | Path
        Path to training CSV.
    test_csv : str | Path
        Path to testing CSV.
    target_column : str
        Name of target column in the CSVs.
    feature_columns : list[str] | None
        Optional list of feature columns to retain. If None, keeps all columns
        except the target.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Filtered train and test DataFrames containing feature_columns + target.
    """
    train_df = pd.read_csv(Path(train_csv))
    test_df = pd.read_csv(Path(test_csv))

    # Ensure consistent column names (strip whitespace)
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # Derive features if not provided
    if feature_columns is None:
        feature_columns = [c for c in train_df.columns if c != target_column]

    # Ensure target exists
    if target_column not in train_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in training data")
    if target_column not in test_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in testing data")

    # Keep only requested features that actually exist
    existing_features = [c for c in feature_columns if c in train_df.columns]

    # Compose final column order
    cols = existing_features + [target_column]
    train_df = train_df[cols].copy()
    test_df = test_df[cols].copy()
    return train_df, test_df


def select_features_with_featurewiz(
    train_csv: str | Path,
    test_csv: str | Path,
    target_column: str,
    *,
    feature_columns: list[str] | None = None,
    corr_limit: float = 0.7,
    verbose: int = 1,
    feature_engg: str | None = None,
    category_encoders: str | None = None,
) -> list[str]:
    """Run Featurewiz on provided train/test CSVs and return selected features.

    This tries the class-based API first, then falls back to the function API
    for compatibility with different Featurewiz versions.
    """
    console: Console = get_custom_console()
    train_df, test_df = _load_and_filter(
        train_csv, test_csv, target_column, feature_columns
    )

    # Separate X, y for class API; keep full frames for function API
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Defaults for optional strings
    feature_engg = feature_engg or ""
    category_encoders = category_encoders or ""

    # Try class API (FeatureWiz)
    try:
        from featurewiz import FeatureWiz

        fw = FeatureWiz(
            corr_limit=corr_limit,
            verbose=verbose,
            feature_engg=feature_engg,  # type: ignore[arg-type]
            category_encoders=category_encoders,  # type: ignore[arg-type]
        )
        _ = fw.fit_transform(X_train, y_train)
        selected = list(getattr(fw, "features", []))
        if selected:
            console.print(
                f"Featurewiz (class API) selected {len(selected)} features.",
                style="info",
            )
            return selected
    except Exception as e:  # noqa: BLE001 - show fallback path as info
        console.print(
            f"FeatureWiz class API failed or unavailable, falling back. Reason: {e}",
            style="warning",
        )

    # Fallback to function API
    try:
        from featurewiz import featurewiz as fw_function

        selected, _trained = fw_function(
            train_df,
            target_column,
            corr_limit=corr_limit,
            verbose=verbose,
            test_data=test_df,
            feature_engg=feature_engg,
            category_encoders=category_encoders,
        )
        selected = list(selected)
        console.print(
            f"Featurewiz (function API) selected {len(selected)} features.",
            style="info",
        )
        return selected
    except Exception as e:  # noqa: BLE001
        console.print(f"Featurewiz function API failed. Reason: {e}", style="danger")
        raise


def save_selected_features(
    selected_features: list[str],
    out_dir: str | Path,
    *,
    corr_limit: float,
    label: str,
    extra_tag: str | None = None,
) -> Path:
    """Save selected features to a timestamped CSV in out_dir.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tag = f"_{extra_tag}" if extra_tag else ""
    fname = f"selected_features_{ts}_n{len(selected_features)}_corr{corr_limit:.2f}{tag}.csv"
    file_path = out_path / fname

    df = pd.DataFrame({"feature": selected_features})
    df.to_csv(file_path, index=False)

    # Also write a lightweight metadata sidecar for traceability
    meta = pd.DataFrame(
        {
            "label": [label],
            "corr_limit": [corr_limit],
            "timestamp": [ts],
            "num_features": [len(selected_features)],
            "extra_tag": [extra_tag or ""],
        }
    )
    meta.to_csv(file_path.with_suffix(".meta.csv"), index=False)
    return file_path
