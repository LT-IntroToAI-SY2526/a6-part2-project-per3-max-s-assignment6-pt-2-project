"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Max Carmona
- 
- 
- 

Dataset: Micheal Jordan Career Statistics
Predicting: Points Per Game (PPG)
Features: Minutes, rebounds, Assists, Steals, Field Goal %
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'mj_stats.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic infor
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)

    # Load the CSV file
    data = pd.read_csv(filename)
    
    # Print 15 rows
    print("\nJordan's 15 seasons:")
    print(data.head(15))
    
    # Print the shape
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Print summary statistics
    print("\nBasic statistics:")
    print(data.describe())
    
    # Check for missing values
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Print column names
    print(f"\nColumn names: {list(data.columns)}")
    
    return data


def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MJ Features vs Points Per Game', fontsize=16, fontweight='bold')
    
    # Plot 1: Minutes vs Points
    axes[0, 0].scatter(data['Minutes'], data['Points'], color='red', alpha=0.6)
    axes[0, 0].set_xlabel('Minutes Per Game')
    axes[0, 0].set_ylabel('Points Per Game')
    axes[0, 0].set_title('Minutes vs Points')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Rebounds vs Points
    axes[0, 1].scatter(data['Rebounds'], data['Points'], color='blue', alpha=0.6)
    axes[0, 1].set_xlabel('Rebounds Per Game')
    axes[0, 1].set_ylabel('Points Per Game')
    axes[0, 1].set_title('Rebounds vs Points')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Assists vs Points
    axes[0, 2].scatter(data['Assists'], data['Points'], color='green', alpha=0.6)
    axes[0, 2].set_xlabel('Assists Per Game')
    axes[0, 2].set_ylabel('Points Per Game')
    axes[0, 2].set_title('Assists vs Points')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Steals vs Points
    axes[1, 0].scatter(data['Steals'], data['Points'], color='orange', alpha=0.6)
    axes[1, 0].set_xlabel('Steals Per Game')
    axes[1, 0].set_ylabel('Points Per Game')
    axes[1, 0].set_title('Steals vs Points')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: FG% vs Points
    axes[1, 1].scatter(data['FG_Percent'], data['Points'], color='purple', alpha=0.6)
    axes[1, 1].set_xlabel('Field Goal %')
    axes[1, 1].set_ylabel('Points Per Game')
    axes[1, 1].set_title('FG% vs Points')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('mj_feature_plots.png', dpi=300, bbox_inches='tight')
    print("\nFeature plots saved as 'mj_feature_plots.png'")
    


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    # Create list of feature columns
    feature_columns = ['Minutes', 'Rebounds', 'Assists', 'Steals', 'FG_Percent']
    
    # Separate features and target
    X = data[feature_columns]
    y = data['Points']
    
    # Print shapes
    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print sizes
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
     # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get feature names
    feature_names = X_train.columns
    
    # Print intercept
    print(f"\nIntercept: {model.intercept_:.2f}")
    
    # Print coefficients
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    
    # Print full equation
    print(f"\nEquation:")
    equation = f"Points = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f}*{name}"
        else:
            if coef >= 0:
                equation += f" + {coef:.2f}*{name}"
            else:
                equation += f" + ({coef:.2f})*{name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)
    
    # Feature importance
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    
    return model

    

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate R² score
    r2 = r2_score(y_test, predictions)
    
    # Calculate MSE and RMSE
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # Print results
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of variation in points")
    
    print(f"\nRoot Mean Squared Error: {rmse:.2f}")
    print(f"  → On average, predictions are off by {rmse:.2f} points")
    
    # Comparison table
    print(f"\n=== Prediction Examples ===")
    print(f"{'Actual PPG':<15} {'Predicted PPG':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)
    
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error)/actual)*100
        print(f"{actual:>13.2f}   {predicted:>15.2f}   {error:>10.2f}   {pct_error:>6.2f}%")
    
    return predictions


def make_prediction(model):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
     # Create sample input (prime MJ stats)
    print("\nPredicting PPG for a hypothetical season:")
    sample_minutes = 38.0
    sample_rebounds = 6.5
    sample_assists = 6.0
    sample_steals = 2.5
    sample_fg = 52.0
    
    # Create DataFrame
    sample = pd.DataFrame([[sample_minutes, sample_rebounds, sample_assists, sample_steals, sample_fg]], 
                         columns=feature_names)
    
    # Make prediction
    predicted_points = model.predict(sample)[0]
    
    # Print results
    print(f"\nInput stats:")
    print(f"  Minutes: {sample_minutes}")
    print(f"  Rebounds: {sample_rebounds}")
    print(f"  Assists: {sample_assists}")
    print(f"  Steals: {sample_steals}")
    print(f"  FG%: {sample_fg}")
    
    print(f"\nPredicted Points Per Game: {predicted_points:.2f}")
    
    return predicted_points


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)

    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)

    plt.show()
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    

