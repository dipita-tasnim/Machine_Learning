# Food Waste Prediction Project

This project aims to predict food waste in kilograms based on various factors such as meals served, kitchen staff, environmental conditions, and operational parameters.

## Project Structure

- `myProject.py`: Main Python script containing the analysis and modeling code
- `requirements.txt`: List of Python dependencies
- `food_waste.csv`: Training dataset
- `food_waste_test.csv`: Test dataset
- `food_waste_submission.csv`: Submission format file
- `correlation_matrix.png`: Generated correlation matrix visualization
- `model_comparison.png`: Generated model performance comparison visualization

## Features

The dataset includes the following features:

### Quantitative Features
- meals_served: Number of meals served
- kitchen_staff: Number of kitchen staff
- temperature_C: Temperature in Celsius
- humidity_percent: Humidity percentage
- day_of_week: Day of the week (0-6)
- special_event: Binary indicator (0/1)
- past_waste_kg: Previous waste in kg
- food_waste_kg: Target variable (waste in kg)

### Categorical Features
- staff_experience: Experience level (Beginner/Intermediate/Expert)
- waste_category: Type of waste (Meat/Dairy/Vegetables/Grains)


## Analysis Steps

1. Data Loading and Initial Exploration
2. Correlation Analysis
3. Data Preprocessing
   - Missing Value Handling
   - Categorical Variable Encoding
   - Feature Scaling
4. Model Training and Evaluation
   - Linear Regression
   - Decision Tree
   - Neural Network
5. Results Visualization and Analysis

## Results

The project implements multiple regression models to predict food waste. The performance of each model is evaluated using R2 score and Mean Squared Error metrics. Visualizations are generated to compare model performance and understand feature correlations.

**Project Report(https://docs.google.com/document/d/1tlXQ02P1SwYLxzZVeruEJKkpTD8BtKVnLQKk8V2cbFY/edit?tab=t.0)
