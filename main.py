# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, clean_data, encode_features, balance_classes
from model_training import train_model, evaluate_model
from survival_analysis import plot_kaplan_meier, fit_cox_model
from feature_importance import plot_feature_importance
from eda_by_subtypes import compare_subtypes, analyze_treatment_effects

# Step 1: Load and Clean Data
df = load_data("data/Breast_Cancer_METABRIC.csv")
df_cleaned = clean_data(df)

# Step 2: Exploratory Data Analysis by Cancer Subtypes
# Compare mutation count by ER status
compare_subtypes(df_cleaned, feature="Mutation Count", by="ER Status")

# Compare tumor size by PR status
compare_subtypes(df_cleaned, feature="Tumor Size", by="PR Status")

# Analyze treatment effects on relapse status by chemotherapy
analyze_treatment_effects(df_cleaned, treatment="Chemotherapy", outcome="Relapse Free Status")

# Step 3: Encode Features and Prepare for Modeling
X, y = encode_features(df_cleaned)

# Step 4: Handle Class Imbalance
X_balanced, y_balanced = balance_classes(X, y)

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Step 6: Train and Evaluate the Model
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

# Step 7: Feature Importance Analysis
plot_feature_importance(model, X.columns, top_n=10)

# Step 8: Survival Analysis
# Kaplan-Meier Survival Curves by ER Status and Chemotherapy Status
plot_kaplan_meier(df_cleaned, group_by="ER Status")
plot_kaplan_meier(df_cleaned, group_by="Chemotherapy")

# Cox Proportional-Hazards Model
fit_cox_model(df_cleaned)
