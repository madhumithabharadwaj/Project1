import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

import numpy as np
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

# Sample data
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
y = np.array([1, 2, 3])

# Initialize model and fit
model = LassoHomotopyModel()
results = model.fit(X, y)

# Print stored coefficients
print("Initial coefficients:", results.coef_)
