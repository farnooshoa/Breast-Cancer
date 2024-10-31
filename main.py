# main.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, clean_data, encode_features, balance_classes
from eda_by_subtypes import compare_subtypes, analyze_treatment_effects, subtype_feature_comparison
from model_training import train_model, evaluate_model
from feature_importance import plot_feature_importance
from sklearn.model_selection import train_test_split
from correlation_analysis import plot_correlation_matrix


# Step 1: Load and Clean Data
df = load_data("Breast_Cancer_METABRIC.csv")  # Replace with your dataset path
df_cleaned = clean_data(df)

# Step 2: Plot Distribution of Target Variable Before Balancing
plt.figure(figsize=(8, 6))
sns.countplot(x='Relapse Free Status', data=df_cleaned)
plt.title("Distribution of Relapse Status Before Balancing")
plt.xlabel("Relapse Free Status")
plt.ylabel("Count")

# Step 3: Encode Features and Prepare for Balancing
X, y = encode_features(df_cleaned)

# Step 4: Apply SMOTE to Balance Classes
X_balanced, y_balanced = balance_classes(X, y)

# Step 5: Plot Distribution of Target Variable After Balancing
plt.figure(figsize=(8, 6))
sns.countplot(x=y_balanced)
plt.title("Distribution of Relapse Status After Balancing")
plt.xlabel("Relapse Free Status")
plt.ylabel("Count")

# Show all plots together
plt.show()

# Exploratory Data Analysis by Cancer Subtypes
# Compare mutation count by ER status
compare_subtypes(df_cleaned, feature="Mutation Count", by="ER Status")

# Compare tumor size by PR status
compare_subtypes(df_cleaned, feature="Tumor Size", by="PR Status")

# Compare multiple clinical features across subtypes (ER, PR, HER2 status)
subtype_feature_comparison(df_cleaned, features=["Mutation Count", "Tumor Size", "Neoplasm Histologic Grade"])

# Analyze treatment effects on relapse status by chemotherapy
analyze_treatment_effects(df_cleaned, treatment="Chemotherapy", outcome="Relapse Free Status")
# Step 3: Encode Features and Prepare for Modeling
X, y = encode_features(df_cleaned)

print("Unique values in y:", y.value_counts())


# Step 5: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = train_model(X_train, y_train)

# Step 7: Evaluate the Model
evaluate_model(model, X_test, y_test)


# Step 7: Plot and Display Top N Feature Importances
plot_feature_importance(model, feature_names=X.columns, top_n=10)  # Change top_n as needed

features = ["Mutation Count", "Tumor Size", "Neoplasm Histologic Grade", "ER Status", "PR Status", "HER2 Status"]
plot_correlation_matrix(df_cleaned, features)