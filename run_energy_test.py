import pandas as pd
import numpy as np
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

# Load Excel file
df = pd.read_excel("ENB2012_data.xlsx")

# Features: first 8 columns
X = df.iloc[:, :8].values

# Target: Heating Load (column 8)
y = df.iloc[:, 8].values  # Use column 8 for Heating Load

# Fit LassoHomotopy model
model = LassoHomotopyModel()
results = model.fit(X, y, tol=0.1)  # Higher tol for sparsity

# Print coefficients
print("Coefficients (expecting sparsity):", results.coef_)

# Count near-zero coefficients
num_zero = np.sum(np.abs(results.coef_) < 1e-2)
print(f"Number of zero coefficients: {num_zero} out of {len(results.coef_)}")

# Predict on training data
predictions = results.predict(X)
print("Sample predictions:", predictions[:10])
