import pandas as pd

# Load the dataset
df = pd.read_csv("./data/model_ready/dataset_test.csv")

# Specify the drillhole ID to filter (for example, ID = 5)
drillhole_id = 861

# Filter the rows where the 'ID' column matches the specified drillhole ID
filtered_df = df[df["ID"] == drillhole_id]

# Specify the features you want to keep (for example, 'Feature1', 'Feature2', 'Feature3')

selected_features = [
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
labels = ["Oberhollenzer_classes"]

# Select only the specified features
filtered_features_df = filtered_df[selected_features + labels]

# Save the filtered data to a new CSV file
filename = f"./data/predict_example/test_hole_{drillhole_id}.csv"
filtered_features_df.to_csv(filename, index=False)

print("Drillhole data saved to:", filename)
