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
    Load your dataset and print basic information
    
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
    
    # Print first 5 rows
    print("\nFirst 5 seasons:")
    print(data.head())
    
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
    plt.show()


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
    
    # Your code here
    
    pass


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
    
    # Your code here
    
    pass


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
    
    # Your code here
    
    pass


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
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    pass


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
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

