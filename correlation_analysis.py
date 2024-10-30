# correlation_analysis.py
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_with_mutation(df, features):
    """Plots the correlation between mutation count and specified clinical features."""
    # Checking correlation with numeric features using scatter plots
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature, y='Mutation Count', hue='ER Status')
        plt.title(f"Correlation between Mutation Count and {feature}")
        plt.xlabel(feature)
        plt.ylabel("Mutation Count")
        plt.legend(title="ER Status")
        plt.show()

    # Heatmap of correlations for numerical features
    corr_matrix = df[features + ['Mutation Count']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix including Mutation Count")
    plt.show()
