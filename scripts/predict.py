import xgboost as xgb

from cpt_to_soiltype.plotting import plot_predictions
from cpt_to_soiltype.utility import load_data, load_xgb_model

if __name__ == "__main__":

    df, features, labels = load_data("data/predict_example/test_hole_859.csv")

    loaded_model = load_xgb_model("models/xgb_model.json")
    dnew = xgb.DMatrix(data=features)
    y_pred_loaded = loaded_model.predict(dnew)

    # Plotting the subplots
    plot_predictions(df, y_pred_loaded, labels, show_labels=True, streamlit=False)
