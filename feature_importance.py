# feature_importance.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=10):
    """Plots the top N most important features from the trained model."""
    # Extracting feature importances and sorting them in descending order
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]  # Get indices of top N features
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Plotting the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_importances[::-1], align="center")
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importances")
    plt.show()

    # Display the top features in a dataframe for further inspection
    feature_df = pd.DataFrame({'Feature': top_features, 'Importance': top_importances})
    print(feature_df)
