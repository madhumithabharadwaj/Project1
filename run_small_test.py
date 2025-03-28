import numpy as np
import pandas as pd
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

# Load small_test.csv
data = pd.read_csv("LassoHomotopy/tests/small_test.csv")

# Assume last column is target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Fit the model
model = LassoHomotopyModel()
results = model.fit(X, y)

# Print coefficients and prediction
print("Coefficients:", results.coef_)

# Predict on training data
predictions = results.predict(X)
print("Predictions:", predictions)
