# CPT to soil type

CPT to soil type is a machine learning tool to predict the granular soil type from cone penetration test (CPT) data. The following soil types are supported:

- 1: gravel
- 2: fine grained organic soils
- 3: coarse grained organic soils
- 4: sand to gravel
- 5: sand
- 6: silt to fine sand
- 7: clay to silt

The developed model uses an XGBoost classifier, trained on the Oberhollenzer dataset: [https://doi.org/10.1016/j.dib.2020.106618](https://doi.org/10.1016/j.dib.2020.106618).

The code is developed for use in an applied machine learning course at the Norwegian Geotechnical Institute (NGI): [https://www.ngi.no/](https://www.ngi.no/). For more information about geotechnical data and applied machine learning, check out this NGI course: [Introduction to Applied Machine Learning - Using Geotechnical Data](https://www.ngi.no/en/events/ngi-code-academy/introduction-to-applied-machine-learning---using-geotechnical-data-pilot-course/)

The implementation of this end-to-end machine learning project focus on reproducability, trustworthiness and readability. The project structure aims to be a good starting point for developing machine learning models for geotechnical engineering applications.

Some of the key features of the project are:

- The projects is developed as a classic software project with functionality structured as a **python package** in the `src` directory and entry points in the `scripts` directory.
- Use of the **Poetry** package manager for dependency management.
- Use of **pyenv** for managing Python versions.
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
- Use of the **Black** and **isort** code formatters for code formatting.

## Installation

1. **Clone the repository**:
    ```sh
    git clone <repo url>
    cd CPT-to-soiltype
    ```

2. **Set the correct Python version**:

    ```sh
    # inspect the .python-version file to see which version is required
    $pversion=cat .\.python-version
    pyenv install $pversion
    pyenv local $pversion
    ```

3. **Install dependencies**:
    ```sh
    poetry install
    ```

4. **Activate the virtual environment**:
    ```sh
    poetry shell
    ```

## Usage

### Train and evaluate a model

```sh
python scripts/train.py
```

Use hydra configuration options with train.py to specify the model and training parameters. For example, to train a model with a KNN classifier, run:

```sh
python scripts/train.py model=knn
```

See options with:

```sh
python scripts/train.py --help
```

Other entry-point scripts are:

```sh
python scripts/preprocess.py
# hyperparameter optimization is only implemented for the XGBoost model
python scripts/optimise_hyperparameters.py
```

### Start the MLflow server

```sh
cd experiments
mlflow ui
```

Open the web interface at your local machine.

### Start the Streamlit application

```sh
# use port 8503 to avoid conflicts with mlflow
streamlit run Main.py --server.port 8503
```

## Contact

The project is developed by Tom F. Hansen.

For any questions or suggestions, please open an issue or contact us at [tom.frode.hansen@ngi.no](mailto:tom.frode.hansen@ngi.no).