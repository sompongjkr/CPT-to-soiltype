from typing import Any

from pydantic import BaseModel, Field


# Model Configurations
class ModelConfig(BaseModel):
    name: str = Field("knn", description="Name of the model")
    params: dict[str, Any] = Field(..., description="Parameters for the model")


# Experiment Configuration
class ExperimentConfig(BaseModel):
    save_model: bool = Field(False, description="Whether to save the model")
    seed: int | None = Field(
        42, description="Random seed for reproducibility, can be None"
    )
    train_fraction: float = Field(
        0.75, description="Fraction of data used for training"
    )
    features: list[str] = Field(
        ..., description="List of features to be used in the experiment"
    )
    site_info: list[str] = Field(..., description="List of site information fields")
    label: str = Field(..., description="Label for the dataset")
    undersample_level: int = Field(..., description="Level for undersampling")
    oversample_level: int = Field(..., description="Level for oversampling")
    soil_classification: dict[int, str] = Field(
        ..., description="Soil classification dictionary"
    )


# MLflow Configuration
class MLflowConfig(BaseModel):
    path: str = Field(..., description="Path to MLflow experiments directory")
    experiment_name: str | None = Field(
        ..., description="Name of the MLflow experiment"
    )


class OptunaConfig(BaseModel):
    n_trials: int = Field(..., description="Number of trials for Optuna optimization")
    path_results: str = Field(
        ..., description="Path to save the results of experiments"
    )


# Preprocessing Configuration
class PreprocessConfig(BaseModel):
    outlier_feature: str = Field(
        ..., description="Feature to use for outlier detection"
    )
    remove_duplicates: bool = Field(
        True, description="Whether to remove duplicate entries"
    )
    remove_outliers_hard: bool = Field(
        True, description="Whether to remove hard outliers"
    )
    remove_outliers_uni: bool = Field(
        False, description="Whether to remove univariate outliers"
    )
    remove_outliers_multi: bool = Field(
        False, description="Whether to remove multivariate outliers"
    )
    univariate_threshold: float = Field(
        3.0, description="Threshold for univariate outlier detection"
    )
    multivariate_threshold: float = Field(
        0.5, description="Threshold for multivariate outlier detection"
    )


# Dataset Configuration
class DatasetConfig(BaseModel):
    path_raw: str = Field(..., description="Path to raw data directory")
    path_intermediate: str = Field(
        ..., description="Path to intermediate data directory"
    )
    path_model_ready: str = Field(..., description="Path to model-ready data directory")
    path_raw_dataset: str = Field(..., description="Path to the raw dataset CSV file")
    path_model_ready_train: str = Field(
        ..., description="Path to the training dataset CSV file"
    )
    path_model_ready_test: str = Field(
        ..., description="Path to the testing dataset CSV file"
    )


# Main Configuration
class Config(BaseModel):
    experiment: ExperimentConfig = Field(
        ..., description="Experiment-specific configuration"
    )
    preprocess: PreprocessConfig = Field(
        ..., description="Data preprocessing configuration"
    )
    dataset: DatasetConfig = Field(..., description="Dataset paths configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    mlflow: MLflowConfig = Field(..., description="MLflow configuration")
    optuna: OptunaConfig = Field(..., description="Optuna configuration")
