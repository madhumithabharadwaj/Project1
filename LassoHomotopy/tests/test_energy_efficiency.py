import sys
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score
)

# Add the project root to the path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

@pytest.mark.energy
def test_energy_efficiency_metrics():
    # Load dataset
    df = pd.read_excel("ENB2012_data.xlsx")
    X = df.iloc[:, :8].values
    y = df.iloc[:, 8].values  # Heating Load as target

    # Fit LASSO Homotopy model
    model = LassoHomotopyModel()
    results = model.fit(X, y, tol=0.1)  # higher tol to encourage sparsity

    # Predictions
    y_pred = results.predict(X)
    coef = results.coef_

    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    med_ae = median_absolute_error(y, y_pred)
    evs = explained_variance_score(y, y_pred)
    num_zero = np.sum(np.abs(coef) < 1e-2)

    # Assertions
    assert mse < 50, f"MSE too high: {mse}"
    assert r2 > 0.8, f"R² too low: {r2}"
    assert num_zero >= 1, f"Expected at least 1 near-zero coefficient, got {num_zero}"

    # Print for report
    print("\n Energy Efficiency Lasso Test Results:")
    print(f" Mean Squared Error (MSE): {mse:.2f}")
    print(f" R-squared (R²): {r2:.4f}")
    print(f" Mean Absolute Error (MAE): {mae:.2f}")
    print(f" Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f" Median Absolute Error: {med_ae:.2f}")
    print(f" Explained Variance Score: {evs:.4f}")
    print(f" Number of near-zero coefficients: {num_zero} out of {len(coef)}\n")
