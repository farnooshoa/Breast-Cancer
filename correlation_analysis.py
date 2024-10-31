# correlation_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df, features):
    """Plots a heatmap of correlations between selected features."""
    # Selecting only relevant features and applying one-hot encoding to categorical columns
    corr_df = df[features].dropna()  # Drop rows with NaNs for a cleaner correlation matrix
    corr_df = pd.get_dummies(corr_df, drop_first=True)  # One-hot encode categorical features

    # Calculating the correlation matrix for numeric columns
    corr_matrix = corr_df.corr()

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Selected Features")
    plt.show()
