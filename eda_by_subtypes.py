# eda_by_subtypes.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compare_subtypes(df, feature, by="ER Status"):
    """Compares a clinical feature across cancer subtypes using boxplots."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=by, y=feature)
    plt.title(f"{feature} by {by}")
    plt.xlabel(by)
    plt.ylabel(feature)


def analyze_treatment_effects(df, treatment="Chemotherapy", outcome="Relapse Free Status"):
    """Analyzes survival or relapse rates based on treatment type across subtypes."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=treatment, hue=outcome)
    plt.title(f"{outcome} by {treatment}")
    plt.xlabel(treatment)
    plt.ylabel("Count")
    plt.legend(title=outcome)


    # Display proportions within each treatment group
    treatment_outcome = pd.crosstab(df[treatment], df[outcome], normalize='index')
    print(f"Proportion of {outcome} by {treatment}:\n", treatment_outcome)

def subtype_feature_comparison(df, features, subtypes=["ER Status", "PR Status", "HER2 status measured by SNP6"]):
    """Compares multiple clinical features across different cancer subtypes."""
    for feature in features:
        for subtype in subtypes:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=subtype, y=feature)
            plt.title(f"{feature} by {subtype}")
            plt.xlabel(subtype)
            plt.ylabel(feature)
    
# Show all plots together
plt.show()