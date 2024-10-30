# eda_by_subtypes.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compare_subtypes(df, feature, by="ER Status"):
    """Compares a clinical feature across cancer subtypes."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=by, y=feature)
    plt.title(f"{feature} by {by}")
    plt.xlabel(by)
    plt.ylabel(feature)
    plt.show()

def analyze_treatment_effects(df, treatment="Chemotherapy", outcome="Relapse Free Status"):
    """Analyzes survival or relapse rates based on treatment type across subtypes."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=treatment, hue=outcome)
    plt.title(f"{outcome} by {treatment}")
    plt.xlabel(treatment)
    plt.ylabel("Count")
    plt.legend(title=outcome)
    plt.show()

    # Display proportions within each treatment group
    treatment_outcome = pd.crosstab(df[treatment], df[outcome], normalize='index')
    print(f"Proportion of {outcome} by {treatment}:\n", treatment_outcome)
