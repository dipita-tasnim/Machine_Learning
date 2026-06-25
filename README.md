# Food Waste Prediction

A machine learning project that predicts daily food waste (in kilograms) for food-service establishments based on operational, environmental, and staffing data. By understanding the key drivers of waste, kitchens can plan smarter, cut costs, and reduce their environmental footprint.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Given roughly 900 daily records describing how many meals were served, the kitchen conditions, and staffing, this project trains and compares three regression models to predict the amount of food waste produced. The pipeline covers everything from raw data cleaning to model evaluation, visualization, and final predictions.

Three models are trained and compared:

| Model | Library | Notes |
|-------|---------|-------|
| Linear Regression | scikit-learn | Simple, interpretable baseline |
| Decision Tree | scikit-learn | Captures non-linear relationships |
| Neural Network (MLP) | scikit-learn | Two hidden layers `(100, 50)`, up to 1000 iterations |

The best-performing model (lowest RMSE on the held-out test set) is automatically selected to generate the final predictions.

---

## Project Structure

```
Machine_Learning/
│
├── myProject.py                 # Main script, full pipeline (run locally)
├── colab.py                     # Google Colab version of the pipeline
├── requirements.txt             # Python dependencies
│
├── food_waste.csv               # Training dataset (with target)
├── food_waste_test.csv          # Test dataset (features only)
├── food_waste_submission.csv    # Ground-truth values for the test set
├── my_predictions.csv           # Generated predictions (output)
│
└── *.png                        # Generated charts (correlations, model comparisons, etc.)
```

---

## Dataset

The dataset contains roughly 911 daily records with the following features:

### Quantitative Features
| Feature | Description |
|---------|-------------|
| `meals_served` | Number of meals served that day |
| `kitchen_staff` | Number of kitchen staff on duty |
| `temperature_C` | Ambient temperature in Celsius |
| `humidity_percent` | Humidity percentage |
| `day_of_week` | Day of the week (0 to 6) |
| `special_event` | Whether a special event occurred (0/1) |
| `past_waste_kg` | Food waste recorded the previous day |
| `food_waste_kg` | Target, food waste produced (kg) |

### Categorical Features
| Feature | Categories |
|---------|-----------|
| `staff_experience` | Beginner / Intermediate / Expert |
| `waste_category` | Meat / Dairy / Vegetables / Grains |

Note: The raw data contains real-world messiness such as inconsistent casing (e.g. `MeAt`, `intermediate`), missing values, and `nan` strings, all handled during preprocessing.

---

## Pipeline

The script runs end-to-end through the following stages:

1. Data Loading. Robust CSV loading that handles both tab- and comma-separated files.
2. Exploratory Analysis. Correlation matrix and feature relationship visualizations.
3. Preprocessing
   - Standardize text casing and fill missing values with `unknown`
   - Encode categorical variables with `LabelEncoder`
   - Scale numeric features with `StandardScaler`
4. Model Training. Train Linear Regression, Decision Tree, and Neural Network.
5. Evaluation. Report R2, Adjusted R2, MSE, RMSE, MAE, and Explained Variance for each model.
6. Model Selection. Pick the model with the lowest RMSE on the test set.
7. Prediction and Export. Save final predictions to `my_predictions.csv`.

---

## Evaluation Metrics

Each model is scored on:

- R2 Score. Proportion of variance explained (1.0 = perfect fit)
- Adjusted R2. R2 adjusted for the number of features
- MSE / RMSE. Mean / Root Mean Squared Error (kg)
- MAE. Mean Absolute Error (kg)
- Explained Variance Score

The pipeline also generates residual plots, actual-vs-predicted scatter plots, and error distribution histograms for visual diagnosis.

---

## Generated Visualizations

Running the script produces a set of charts, including:

| File | Description |
|------|-------------|
| `correlation_matrix.png` | Heatmap of feature correlations |
| `lr_visualizations.png` | Linear Regression diagnostics |
| `dt_visualizations.png` | Decision Tree diagnostics |
| `nn_visualizations.png` | Neural Network diagnostics |
| `model_comparison_visualizations.png` | R2, RMSE, and residual comparison across models |
| `actual_vs_predicted.png` | Final model: actual vs predicted waste |

---

## Getting Started

### Prerequisites
- Python 3.8 or higher

### Installation

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd Machine_Learning
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run locally:
```bash
python myProject.py
```

This trains all three models, prints detailed metrics to the console, saves all visualizations, and writes predictions to `my_predictions.csv`.

Run on Google Colab:

Upload `train.csv`, `test.csv`, and `submission.csv` to `/content/sample_data/`, then run `colab.py`. (The Colab version uses inline plots instead of saving PNG files.)

---

## Dependencies

```
pandas==2.0.3
numpy==1.24.3
seaborn==0.12.2
matplotlib==3.7.1
scikit-learn==1.3.0
```

---

## Key Findings

- Meals served and past waste are the strongest predictors of food waste.
- Environmental factors (temperature, humidity) have a comparatively weak impact.
- Proper preprocessing (case normalization, missing-value handling, scaling) was essential to get clean, comparable model results.

---

## Future Improvements

- Collect more data to improve model accuracy
- Add features such as menu type and seasonality
- Try advanced models like XGBoost or Random Forest
- Add cross-validation for more robust evaluation
- Explore ensemble methods combining multiple models

---

## Project Report

[Project Report](https://docs.google.com/document/d/1tlXQ02P1SwYLxzZVeruEJKkpTD8BtKVnLQKk8V2cbFY/edit?tab=t.0)
