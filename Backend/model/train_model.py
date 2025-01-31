import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("../data/53ft_van_synthetic_data.csv")


# Encode Load Type as numerical values
df["Load Type Encoded"] = df["Load Type"].astype("category").cat.codes

# Define features (X) and target (y)
X = df[["Carton Count", "Load Type Encoded", "Estimated Cargo Volume"]]
y = df["Fill Percentage"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_filename = "trailer_fill_model_53ft.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

print(f"Model trained and saved as '{model_filename}'")
