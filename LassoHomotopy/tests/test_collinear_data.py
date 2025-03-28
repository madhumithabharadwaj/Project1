import sys
import os
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error, r2_score

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

@pytest.mark.collinear
def test_collinear_data():
    df = pd.read_csv("LassoHomotopy/tests/collinear_data.csv")
    X = df.drop("target", axis=1).values
    y = df["target"].values


    model = LassoHomotopyModel()
    results = model.fit(X, y, tol=0.1)
    y_pred = results.predict(X)
    coef = results.coef_

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    num_zero = np.sum(np.abs(coef) < 1e-2)

    print("\n Collinear Data Lasso Test Results:")
    print(f" MSE: {mse:.2f}")
    print(f" R²: {r2:.4f}")
    print(f" Near-zero coefficients: {num_zero} / {len(coef)}")

    assert r2 > 0.8, f"R² too low: {r2}"
    assert num_zero >= 1, f"Expected sparsity but got {num_zero} near-zero coefficients"
