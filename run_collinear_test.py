import numpy as np
import pandas as pd
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

# Load collinear_data.csv
data = pd.read_csv("LassoHomotopy/tests/collinear_data.csv")

# Assume last column is target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Fit the model with higher tolerance
model = LassoHomotopyModel()
results = model.fit(X, y, tol=0.2)

# Print coefficients
print("Coefficients (expecting sparsity):", results.coef_)

# Check how many coefficients are zero (or near zero)
num_zero = np.sum(np.abs(results.coef_) < 1e-2)


print(f"Number of zero coefficients: {num_zero} out of {len(results.coef_)}")
