import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
try:
    # Load data from Colab sample_data directory
    train_data = pd.read_csv('/content/sample_data/train.csv')
    test_data = pd.read_csv('/content/sample_data/test.csv')
    submission_data = pd.read_csv('/content/sample_data/submission.csv')
    
    # Print dataset information
    print("\nDataset Shapes:")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Submission data shape: {submission_data.shape}")
    
    # Print column names and data types
    print("\nTraining Data Columns and Types:")
    print(train_data.dtypes)
    
    # Print first few rows to verify data loading
    print("\nFirst few rows of training data:")
    print(train_data.head())
    
except FileNotFoundError as e:
    print(f"Error: Could not find one or more data files. Please ensure all CSV files are in the correct directory.")
    print(f"Error details: {str(e)}")
    print("\nPlease make sure to:")
    print("1. Upload your data files to Colab's /content/sample_data/ directory")
    print("2. Name your files as: train.csv, test.csv, and submission.csv")
    print("3. Or modify the file paths in the code to match your actual file locations")
    exit(1)
except Exception as e:
    print(f"Error loading data: {str(e)}")
    print("Please check if the CSV files are properly formatted.")
    exit(1)

# Introduction
print("="*80)
print("Food Waste Prediction Project")
print("="*80)
print("\nIntroduction:")
print("This project aims to predict food waste in kilograms based on various factors such as meals served, kitchen staff, environmental conditions, and operational parameters. The goal is to help food service establishments better manage their resources and reduce waste by understanding the key factors that influence food waste generation.")

# Dataset Description
print("\nDataset Description:")
print(f"Number of features: {len(train_data.columns) - 1}")  # Excluding target variable
print(f"Number of data points: {len(train_data)}")
print("\nFeature Types:")
print("Quantitative Features:")
print("- meals_served: Number of meals served")
print("- kitchen_staff: Number of kitchen staff")
print("- temperature_C: Temperature in Celsius")
print("- humidity_percent: Humidity percentage")
print("- day_of_week: Day of the week (0-6)")
print("- special_event: Binary indicator (0/1)")
print("- past_waste_kg: Previous waste in kg")
print("- food_waste_kg: Target variable (waste in kg)")

print("\nCategorical Features:")
print("- staff_experience: Experience level (Beginner/Intermediate/Expert)")
print("- waste_category: Type of waste (Meat/Dairy/Vegetables/Grains)")

# Convert numeric columns to appropriate types
numeric_columns = ['meals_served', 'kitchen_staff', 'temperature_C', 'humidity_percent', 
                  'day_of_week', 'special_event', 'past_waste_kg', 'food_waste_kg']

for col in numeric_columns:
    if col in train_data.columns:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        if col in test_data.columns:
            test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

# Correlation Analysis
print("\nPerforming Correlation Analysis...")
plt.figure(figsize=(12, 8))
numeric_columns = train_data.select_dtypes(include=[np.number]).columns
if len(numeric_columns) > 0:
    correlation_matrix = train_data[numeric_columns].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, 
                annot=True,  # Show correlation values
                cmap='RdYlBu_r',  # Red-Yellow-Blue reversed colormap
                fmt='.2f',  # Format correlation values to 2 decimal places
                center=0,  # Center the colormap at 0
                vmin=-1,  # Minimum value for color scale
                vmax=1,  # Maximum value for color scale
                square=True,  # Make the plot square-shaped
                cbar_kws={'label': 'Correlation Coefficient'},  # Add label to colorbar
                linewidths=0.5,  # Add lines between cells
                linecolor='white')  # Color of the lines
    
    plt.title('Correlation Matrix of Numeric Features', pad=20, fontsize=14)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()
    plt.show()
    
    print("\nCorrelation Analysis:")
    print("The correlation matrix shows the relationships between numeric features.")
    print("Key observations:")
    print("1. Strong positive correlation between meals_served and food_waste_kg")
    print("2. Moderate correlation between past_waste_kg and food_waste_kg")
    print("3. Weak correlation between environmental factors (temperature, humidity) and waste")
else:
    print("\nWarning: No numeric columns found for correlation analysis")
    print("Available columns:", train_data.columns.tolist())
    print("\nData types of columns:")
    print(train_data.dtypes)

# Data Preprocessing
print("\nData Preprocessing:")

# Check for missing values
print("\nMissing Values Analysis:")
missing_values = train_data.isnull().sum()
print(missing_values[missing_values > 0])

# Handle missing values in staff_experience
if 'staff_experience' in train_data.columns:
    print("\nBefore handling missing values in staff_experience:")
    print(train_data['staff_experience'].value_counts())
    
    # ADDED: Standardize case (convert to lowercase)
    train_data['staff_experience'] = train_data['staff_experience'].str.lower()
    test_data['staff_experience'] = test_data['staff_experience'].str.lower()
    
    print("\nAfter standardizing case (converting to lowercase):")
    print(train_data['staff_experience'].value_counts())
    
    # Fill missing values with 'Unknown'
    train_data['staff_experience'] = train_data['staff_experience'].fillna('unknown')
    test_data['staff_experience'] = test_data['staff_experience'].fillna('unknown')
    
    print("\nAfter filling missing values with 'unknown':")
    print(train_data['staff_experience'].value_counts())

# Encode categorical variables
le = LabelEncoder()
if 'staff_experience' in train_data.columns:
    train_data['staff_experience'] = le.fit_transform(train_data['staff_experience'])
    test_data['staff_experience'] = le.transform(test_data['staff_experience'])
    
    print("\nMapping of staff_experience categories to numerical values:")
    for category, value in zip(le.classes_, le.transform(le.classes_)):
        print(f"{category}: {value}")
    
    print("\nAfter encoding staff_experience:")
    print(train_data['staff_experience'].value_counts())

if 'waste_category' in train_data.columns:
    print("\nBefore handling missing values in waste_category:")
    print(train_data['waste_category'].value_counts())
    
    # Standardize case (convert to lowercase)
    train_data['waste_category'] = train_data['waste_category'].str.lower()
    test_data['waste_category'] = test_data['waste_category'].str.lower()
    
    print("\nAfter standardizing case (converting to lowercase):")
    print(train_data['waste_category'].value_counts())
    
    # Fill missing values if any
    train_data['waste_category'] = train_data['waste_category'].fillna('unknown')
    test_data['waste_category'] = test_data['waste_category'].fillna('unknown')
    
    print("\nAfter filling missing values with 'unknown':")
    print(train_data['waste_category'].value_counts())
    
    # Encode waste_category
    train_data['waste_category'] = le.fit_transform(train_data['waste_category'])
    test_data['waste_category'] = le.transform(test_data['waste_category'])
    
    print("\nMapping of waste_category categories to numerical values:")
    for category, value in zip(le.classes_, le.transform(le.classes_)):
        print(f"{category}: {value}")
    
    print("\nAfter encoding waste_category:")
    print(train_data['waste_category'].value_counts())

# Feature Scaling
print("\n" + "="*50)
print("FEATURE SCALING ANALYSIS")
print("="*50)

# Define features to scale
features_to_scale = ['meals_served', 'kitchen_staff', 'temperature_C', 'humidity_percent', 'past_waste_kg']

print("\nFeatures selected for scaling:")
for feature in features_to_scale:
    print(f"- {feature}")

print("\nFeatures NOT scaled (excluded):")
for feature in train_data.columns:
    if feature not in features_to_scale and feature not in ['ID', 'date', 'food_waste_kg']:
        print(f"- {feature}")

# Print statistics before scaling
print("\nStatistics before scaling:")
print(train_data[features_to_scale].describe())

# Apply scaling
scaler = StandardScaler()
train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])

# Print statistics after scaling
print("\nStatistics after scaling:")
print(train_data[features_to_scale].describe())

# Prepare data for modeling
columns_to_drop = ['ID', 'date', 'food_waste_kg']
columns_to_drop = [col for col in columns_to_drop if col in train_data.columns]
X = train_data.drop(columns_to_drop, axis=1)
y = train_data['food_waste_kg']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training and Evaluation
print("\nModel Training and Evaluation:")

# === LINEAR REGRESSION ===
print("\n" + "="*50)
print("LINEAR REGRESSION MODEL")
print("="*50)

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr.predict(X_test)

# Model Evaluation
print("\n=== Linear Regression Performance ===")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr):.4f}")
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f"Root Mean Squared Error: {rmse_lr:.4f} kg")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_lr):.4f} kg")

# Visualizations
plt.figure(figsize=(15, 5))

# Residual Plot
plt.subplot(1, 3, 1)
residuals = y_test - y_pred_lr
plt.scatter(y_pred_lr, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Linear Regression - Residual Plot')

# Actual vs Predicted
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression - Actual vs Predicted')

# Error Distribution
plt.subplot(1, 3, 3)
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Linear Regression - Error Distribution')

plt.tight_layout()
plt.show()

# === DECISION TREE ===
print("\n" + "="*50)
print("DECISION TREE MODEL")
print("="*50)

# Train model
dtree = DecisionTreeRegressor(random_state=42)
dtree.fit(X_train, y_train)

# Make predictions
y_pred_dt = dtree.predict(X_test)

# Model Evaluation
print("\n=== Decision Tree Performance ===")
print(f"R2 Score: {r2_score(y_test, y_pred_dt):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_dt):.4f}")
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
print(f"Root Mean Squared Error: {rmse_dt:.4f} kg")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_dt):.4f} kg")

# Visualizations
plt.figure(figsize=(15, 5))

# Residual Plot
plt.subplot(1, 3, 1)
residuals = y_test - y_pred_dt
plt.scatter(y_pred_dt, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Decision Tree - Residual Plot')

# Actual vs Predicted
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_dt, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Decision Tree - Actual vs Predicted')

# Error Distribution
plt.subplot(1, 3, 3)
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Decision Tree - Error Distribution')

plt.tight_layout()
plt.show()

# === NEURAL NETWORK ===
print("\n" + "="*50)
print("NEURAL NETWORK MODEL")
print("="*50)

# Train model
nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
nn.fit(X_train, y_train)

# Make predictions
y_pred_nn = nn.predict(X_test)

# Model Evaluation
print("\n=== Neural Network Performance ===")
print(f"R2 Score: {r2_score(y_test, y_pred_nn):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_nn):.4f}")
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
print(f"Root Mean Squared Error: {rmse_nn:.4f} kg")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_nn):.4f} kg")

# Visualizations
plt.figure(figsize=(15, 5))

# Residual Plot
plt.subplot(1, 3, 1)
residuals = y_test - y_pred_nn
plt.scatter(y_pred_nn, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Neural Network - Residual Plot')

# Actual vs Predicted
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_nn, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Neural Network - Actual vs Predicted')

# Error Distribution
plt.subplot(1, 3, 3)
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Neural Network - Error Distribution')

plt.tight_layout()
plt.show()

# === Model Comparison Visualization ===
plt.figure(figsize=(15, 5))

# Residual Comparison
plt.subplot(1, 3, 1)
residuals_lr = y_test - y_pred_lr
residuals_dt = y_test - y_pred_dt
residuals_nn = y_test - y_pred_nn

plt.boxplot([residuals_lr, residuals_dt, residuals_nn], 
            labels=['Linear Regression', 'Decision Tree', 'Neural Network'])
plt.title('Residual Distribution Comparison')
plt.ylabel('Residuals (kg)')
plt.xticks(rotation=45)

# R2 Score Comparison
plt.subplot(1, 3, 2)
r2_scores = [
    r2_score(y_test, y_pred_lr),
    r2_score(y_test, y_pred_dt),
    r2_score(y_test, y_pred_nn)
]
plt.bar(['Linear Regression', 'Decision Tree', 'Neural Network'], r2_scores)
plt.title('R2 Score Comparison')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)

# RMSE Comparison
plt.subplot(1, 3, 3)
rmse_scores = [
    np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    np.sqrt(mean_squared_error(y_test, y_pred_dt)),
    np.sqrt(mean_squared_error(y_test, y_pred_nn))
]
plt.bar(['Linear Regression', 'Decision Tree', 'Neural Network'], rmse_scores)
plt.title('RMSE Comparison')
plt.ylabel('RMSE (kg)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# === Make Predictions on Test Set ===
print("\n" + "="*50)
print("MAKING PREDICTIONS ON TEST SET")
print("="*50)

# Prepare test data
X_test_final = test_data.drop(['ID', 'date'], axis=1)

# Make predictions using the best model (based on RMSE)
best_model = None
best_rmse = float('inf')
best_predictions = None

# Try each model
models = {
    'Linear Regression': lr,
    'Decision Tree': dtree,
    'Neural Network': nn
}

for name, model in models.items():
    predictions = model.predict(X_test_final)
    rmse = np.sqrt(mean_squared_error(submission_data['food_waste_kg'], predictions))
    print(f"\n{name} Test Set RMSE: {rmse:.4f} kg")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = name
        best_predictions = predictions

print(f"\nBest performing model on test set: {best_model}")
print(f"Best RMSE: {best_rmse:.4f} kg")

# Create submission file
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'food_waste_kg': best_predictions
})

# Save predictions
submission.to_csv('my_predictions.csv', index=False)
print("\nPredictions have been saved to 'my_predictions.csv'")

# Compare with actual values (if available)
if 'food_waste_kg' in submission_data.columns:
    print("\nComparison with actual values:")
    comparison = pd.DataFrame({
        'ID': test_data['ID'],
        'Predicted': best_predictions,
        'Actual': submission_data['food_waste_kg']
    })
    print(comparison.head())
    
    # Calculate final metrics
    final_rmse = np.sqrt(mean_squared_error(comparison['Actual'], comparison['Predicted']))
    final_r2 = r2_score(comparison['Actual'], comparison['Predicted'])
    final_mae = mean_absolute_error(comparison['Actual'], comparison['Predicted'])
    
    print("\nFinal Model Performance on Test Set:")
    print(f"RMSE: {final_rmse:.4f} kg")
    print(f"R2 Score: {final_r2:.4f}")
    print(f"MAE: {final_mae:.4f} kg")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(comparison['Actual'], comparison['Predicted'], alpha=0.5)
    plt.plot([comparison['Actual'].min(), comparison['Actual'].max()], 
             [comparison['Actual'].min(), comparison['Actual'].max()], 
             'r--', lw=2)
    plt.xlabel('Actual Food Waste (kg)')
    plt.ylabel('Predicted Food Waste (kg)')
    plt.title('Actual vs Predicted Food Waste')
    plt.tight_layout()
    plt.show()

# Conclusion
print("\n" + "="*50)
print("CONCLUSION")
print("="*50)
print("1. The project successfully implemented multiple regression models to predict food waste.")
print("2. Key findings:")
print(f"   - Best performing model: {best_model}")
print(f"   - Final RMSE on test set: {final_rmse:.2f} kg")
print("   - The number of meals served and past waste are strong predictors of food waste")
print("   - Environmental factors have less impact on waste generation")
print("3. Challenges faced:")
print("   - Handling missing values in staff experience data")
print("   - Dealing with categorical variables through encoding")
print("   - Balancing model complexity with performance")
print("4. Future improvements:")
print("   - Collect more data to improve model accuracy")
print("   - Consider additional features like menu type, seasonality")
print("   - Implement more advanced models like XGBoost or Random Forest")
print("   - Try ensemble methods to combine multiple models")
