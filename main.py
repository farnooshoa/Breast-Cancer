import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 1. Load the dataset
df = pd.read_csv('Breast_Cancer_METABRIC.csv')

# 2. Handle missing values

# Calculate missing percentages
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Drop columns with more than 50% missing values
columns_to_drop = missing_percentage[missing_percentage > 50].index
df.drop(columns=columns_to_drop, inplace=True)

# Impute missing values in numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Impute missing values in categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# 3. Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 4. Define features and target
y = df["Patient's Vital Status"]
X = df.drop(columns=["Patient ID", "Patient's Vital Status"])

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate the model
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

y_pred = model.predict(X_test)

# For multi-class, use "ovr" (One-vs-Rest) strategy with predict_proba
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

# Print the results
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Feature importance
import matplotlib.pyplot as plt

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()
