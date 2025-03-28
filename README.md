# Project 1

# Team Members

- MADHUMITHA BHARADWAJ GUDMELLA, A20544762
- BHAVANA POLAKALA, A20539792
- SHREYA JATLING, A20551616
- DEVESH PATEL, A20548274


# Overview

This project focuses on analyzing energy efficiency using regression techniques. It aims to predict energy performance using structured datasets while incorporating various data preprocessing, visualization, and modeling techniques. The project explores different statistical methods, including LASSO regression, to improve prediction accuracy and optimize feature selection.

# Features:

- Regression analysis for energy efficiency prediction

- Feature selection using LASSO regression

- Identification of collinearity issues in datasets

- Data visualization using Jupyter Notebooks

- Implementation of custom scripts for testing and synthetic data generation

- Evaluation of model performance using statistical metrics

# Project Structure

- `run_energy_test.py` – Runs energy-related model tests.
- `generate_regression_data.py` – Generates regression data for analysis.
- `run_collinear_test.py` – Tests collinearity in the dataset.
- `energy_lasso_demo.ipynb` – Demonstrates Lasso regression for energy efficiency analysis.
- `visualize_generated_data.ipynb` – Visualizes generated regression data.
- `visualize_energy_efficiency.ipynb` – Provides energy efficiency visualizations.
- `ENB2012_data.xlsx` – Dataset used for the project.
- `requirements.txt` – Lists the dependencies required for the project.
- `test_init.py` – Contains test scripts for validating models and data processing.

# Implementation:

1. Data Processing

The dataset used: ENB2012_data.xlsx

generate_regression_data.py: Generates synthetic regression data.

visualize_generated_data.ipynb: Visualizes generated data to understand patterns.

2. Model Implementation

energy_lasso_demo.ipynb: Demonstrates LASSO regression application for feature selection.

run_collinear_test.py: Identifies collinearity in the dataset to optimize model performance.

3. Testing and Validation

run_energy_test.py: Evaluates model performance on energy efficiency data.

test_init.py: Includes test cases for regression model validation.

# Installation

1. Clone the repository.

2. Navigate to the project directory:
  
   cd Project1-main
 
3. Create and activate a virtual environment (optional but recommended):
  
   python -m venv venv
   source venv/bin/activate
   
4. Install dependencies:

   pip install -r requirements.txt
   

# Usage

- Feature Selection: Run energy_lasso_demo.ipynb to analyze feature importance and optimize regression performance.

- Model Testing: Execute run_energy_test.py to validate energy efficiency models.

- Data Exploration: Modify generate_regression_data.py to create custom datasets and analyze energy efficiency trends.

- To run:
  pytest -s

# Dependencies

The required dependencies are listed in `requirements.txt`. Install them using:

pip install -r requirements.txt


# Acknowledgments

- The dataset `ENB2012_data.xlsx` is used for energy efficiency analysis.
- Libraries such as NumPy, Pytest and Pandas are utilized.

# References

- https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf

- LASSO Regression: Tibshirani, R. (1996). "Regression shrinkage and selection via the LASSO." Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.
https://www.jstor.org/stable/2346178



* What does the model you have implemented do and when should it be used?

- This project implements the LASSO (Least Absolute Shrinkage and Selection Operator) regression model using the Homotopy method, from first principles.
LASSO is a linear regression technique that adds L1 regularization, encouraging sparsity in model coefficients. The model used in this uses machine learning techniques like regression and Lasso regularization to predict energy efficiency. In order to forecast energy efficiency, it analyzes the ENB2012_data.xlsx dataset, which contains a variety of building parameters. When predicting a building's heating loads based on environmental and architectural considerations, the model should be implemented. Optimizing energy use, creating energy-efficient buildings, and lowering heating system expenses can all benefit from this. Energy sustainability researchers, engineers, and architects can all benefit from the concept.

Furthermore, the model's ability to manage predictive analysis while maintaining data validity is suggested by the inclusion of collinearity tests and regression data generation. It is especially important when precise energy efficiency evaluations are needed, like for green building certifications or building code compliance checks. The model is a powerful tool in energy modeling because of its capacity to manage big datasets and offer visual insights.

* How did you test your model to determine if it is working reasonably correctly?

- We created multiple unit tests in the LassoHomotopy/tests/ directory to verify key behaviors of the model:

test_predict: Ensures predictions work and match expected output shape

test_collinear_data: Confirms sparsity under highly collinear inputs

test_energy_efficiency: Tests the model on real-world energy usage data

test_generated_data: Verifies generalization on synthetic datasets

All tests pass using pytest, and assertions are made for shape, sparsity, and convergence behavior. To make sure the model was accurate and reliable, it was tested using a variety of techniques. Initially, prediction performance was evaluated using common machine learning assessment measures, including Mean Squared Error (MSE) and R-squared values. By comparing the expected and actual data, the visualization notebooks made it possible to manually examine patterns and irregularities. Cross-validation methods were used to make sure the model performed effectively when applied to various data subsets. Additionally, testing on unseen data confirmed that it could accurately predict outcomes outside of the training set. Together, these methods made that the model was operating properly and producing accurate estimates of energy efficiency.

* What parameters have you exposed to users of your implementation in order to tune performance?

- The model has a number of tunable parameters so that consumers may modify performance to meet their own requirements. The Lasso regression model's regularization strength is a crucial parameter that reduces overfitting by penalizing less significant features. This parameter can be changed by users to balance accuracy and model complexity.

max_iter: Maximum number of iterations (default = 100)

tol: Tolerance for stopping criteria (default = 1e-5)

These parameters can be passed to the fit() method to balance speed and accuracy:

model.fit(X, y, max_iter=200, tol=1e-4)

Furthermore, users might be able to choose which input features to employ, enabling them to add or remove specific building parameters according to their subject expertise. Regression-related hyperparameters, such as the learning rate, iteration count, or polynomial degree for non-linear regression models, may be adjustable through the test scripts. To improve performance, one can also adjust the dataset splitting ratio (train-test split percentage, for example). Additionally, visualization settings could be altered to improve results analysis and interpretation. Users can experiment with various configurations to increase accuracy and generalization due to these open possibilities.

* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

- Highly correlated input features could make implementation difficult and affect the regression model's efficacy. Determining the actual influence of each feature can be challenging when independent variables are collinear, as this can cause instability in coefficient estimation. Even with collinearity tests, the model may still have trouble processing noisy or duplicated data. Extremely high-dimensional data can lead to numerical instability if not preprocessed properly. Perfect multicollinearity can slow convergence, although the model still produces sparse solutions. Noisy data with very weak signals may require tuning tol and preprocessing. Furthermore, because regression models are sensitive to significant deviations, performance may be impacted by severe outliers in the dataset. In order for the model to handle categorical variables efficiently, extra preprocessing processes like encoding can be needed. To overcome these obstacles, more time could be spent using feature selection strategies or dimensionality reduction techniques like Principal Component Analysis (PCA). Data imputation approaches may improve resilience if the dataset has missing values. There are basic constraints in handling uncertain data distributions, even though some problems can be resolved with extra data preprocessing and model modifications.