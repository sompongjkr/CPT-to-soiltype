# CPT to soil type

CPT to soil type is a machine learning tool to predict the soiltype from the cone penetration test (CPT) data.

The code is developed for use in the applied machine learning course at the Norwegian Geotechnical Institute (NGI): [https://www.ngi.no/](https://www.ngi.no/)

## Installation

1. **Clone the repository**:
    ```sh
    git clone <repo url>
    cd CPT-to-soiltype
    ```

2. **Install dependencies**:
    ```sh
    poetry install
    ```

3. **Activate the virtual environment**:
    ```sh
    poetry shell
    ```


## Usage

### Train and evaluate a model

```sh
python train.py
```

Use hydra configuration options with train.py to specify the model and training parameters. For example, to train a model with a KNN classifier, run:

```sh
python train.py model=knn
```

See options with:
    
```sh
python train.py --help
```



### Start the Streamlit application

```sh
streamlit run Main.py
```

## Contact

For any questions or suggestions, please open an issue or contact us at [tom.frode.hansen@ngi.no](mailto:tom.frode.hansen@ngi.no).