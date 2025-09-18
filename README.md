# CPT to soil type

CPT to soil type is a machine learning tool to predict the granular soil type from cone penetration test (CPT) data. The following soil types are supported:

- 1: gravel
- 2: fine grained organic soils
- 3: coarse grained organic soils
- 4: sand to gravel
- 5: sand
- 6: silt to fine sand
- 7: clay to silt

The developed model uses an XGBoost classifier, trained on the Oberhollenzer dataset: [https://doi.org/10.1016/j.dib.2020.106618](https://doi.org/10.1016/j.dib.2020.106618). The dataset can be downloaded from [here](https://www.tugraz.at/en/institutes/ibg/research/computational-geotechnics-group/database).

The code is developed for use in an applied machine learning course at the Norwegian Geotechnical Institute (NGI): [https://www.ngi.no/](https://www.ngi.no/). For more information about geotechnical data and applied machine learning, check out this NGI course: [Introduction to Applied Machine Learning - Using Geotechnical Data](https://www.ngi.no/en/events/ngi-code-academy/applied-machine-learning-using-geotechnical-data/)

The implementation of this end-to-end machine learning project focus on reproducability, trustworthiness and readability. The project structure aims to be a good starting point for developing machine learning models for geotechnical engineering applications.

Some of the key features of the project are:

- The projects is developed as a classic software project with functionality structured as a **python package** in the `src` directory and entry points in the `scripts` directory.
- Use of the **uv** package manager for dependency management and Python versions.
- In several topics the code is demonstrated in **jupyter notebooks** for educational purposes, then later refactored into the main codebase.
- Use of the **Hydra** configuration framework for easy configuration of the model and training parameters.
- Use of **mlflow** for tracking experiments and model parameters.
- Use of **ydata-profiling** for data Exploratory Data Analysis (EDA).
- Use of the **XGBoost** library (optionally with gpu support) for training the machine learning model.
- Use of the **scikit-learn** library for data preprocessing ml-algorithms (other than xgboost) and evaluation of the model.
- Use **pyOD** for outlier detection.
- Use **imblearn** for handling imbalanced datasets.
- Use **pydantic** for data validation.
- Use of **optuna** for hyperparameter optimization.
- Use of the **Streamlit** library for developing interactive web applications.
- Use of the **Ruff** and **isort** code formatters for code formatting.

## Installation

1. **Clone the repository**:
    ```sh
    git clone <repo url>
    cd CPT-to-soiltype
    ```
2. **Install uv**: Follow the instructions at [https://uv.dev/](https://uv.dev/) to install uv.

3. **Install dependencies**:
    ```sh
    uv sync
    ```


## Usage

### Prepare the data

1. Download the Oberhollenzer dataset from [here](https://www.tugraz.at/en/institutes/ibg/research/computational-geotechnics-group/database) and place the file `CPT_PremstallerGeotechnik_revised.csv` in the `data/raw/` directory.

2. Preprocess the data:

```sh
uv run python scripts/preprocess.py
```

### Optimise hyperparameters (optional)

```sh
uv run python scripts/optimise_hyperparameters.py
```

### Select features (optional)

```sh
uv run python scripts/select_features.py
```

### Train and evaluate a model

```sh
uv run python scripts/train.py
```

Use hydra configuration options with train.py to specify the model and training parameters. For example, to train a model with a KNN classifier, run:

```sh
uv run python scripts/train.py model=knn
```

See options with:

```sh
uv run python scripts/train.py --help
```

Other entry-point scripts are:

```sh
uv run python scripts/preprocess.py
# hyperparameter optimization is only implemented for the XGBoost model
uv run python scripts/optimise_hyperparameters.py
```

### Start the MLflow server

```sh
cd experiments
uv run mlflow ui
```

Open the web interface at your local machine.


### Running the Streamlit Application

Before starting the Streamlit app (`Main.py`), make sure you have completed the following steps:

1. **Install dependencies**
    ```sh
    uv sync
    ```

2. **Preprocess the data**
    ```sh
    uv run python scripts/preprocess.py
    ```
    This will generate model-ready data in `data/model_ready/`.

3. **Train the model**
    ```sh
    uv run python scripts/train.py
    ```
    This will save the trained model to `models/xgb_model.json`.

    Optionally, optimize hyperparameters:
    ```sh
    uv run python scripts/optimise_hyperparameters.py
    ```

Then you can run the Streamlit app:

4. **Run the Streamlit app**
    ```sh
    uv run streamlit run Main.py --server.port 8503
    ```

## Deployment on Streamlit Community Cloud

These steps ensure the app has the necessary data and model files to function correctly.

### Export requirements.txt for Streamlit Cloud

Streamlit Community Cloud expects a `requirements.txt` at the repository root. You can export one from your uv-managed project with pinned (locked) versions and without development-only dependencies:

```sh
uv export --no-dev --no-hashes -o requirements.txt
```

Notes:
- `--no-dev` excludes tools like ruff, isort, mypy from the deployment image.
- `--no-hashes` avoids requiring `pip --require-hashes` on the Streamlit platform.
- If you want to enforce the exact versions from `uv.lock`, you can add `--frozen`.

### Deploy the app

Follow the instructions at [https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app) to deploy the app.


## Contact

The project is developed by Tom F. Hansen and Sjur Beyer.

For any questions or suggestions, please open an issue or contact us at [tom.frode.hansen@ngi.no](mailto:tom.frode.hansen@ngi.no).
