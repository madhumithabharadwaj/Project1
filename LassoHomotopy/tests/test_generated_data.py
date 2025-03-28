import sys
import os
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, r2_score
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

# Add project root to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

@pytest.mark.generated
def test_generated_data_lasso():
    # Load the synthetic dataset
    df = pd.read_csv("LassoHomotopy/tests/generated_test.csv")
    X = df.drop("y", axis=1).values
    y = df["y"].values

    # Fit model
    model = LassoHomotopyModel()
    results = model.fit(X, y, tol=0.1)
    coef = results.coef_
    y_pred = results.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    num_zero = np.sum(np.abs(coef) < 0.05)  # stricter threshold

    # Print
    print("\n Synthetic Data Lasso Test Results:")
    print("True Coefficients: [2, -3, 0, 0, 5]")
    print("Learned Coefficients:", np.round(coef, 2))
    print(f" MSE: {mse:.2f}")
    print(f" R²: {r2:.4f}")
    print(f" Near-zero coefficients: {num_zero} / {len(coef)}")

    # Assertions
    assert r2 > 0.99, f"R² too low: {r2}"
    assert mse < 5, f"MSE too high: {mse}"
    assert num_zero >= 2, f"Expected at least 2 near-zero coefficients, got {num_zero}"
