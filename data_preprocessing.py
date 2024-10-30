# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Handles missing values and cleans the dataset."""
    # Fill numerical columns with the median
    num_imputer = SimpleImputer(strategy='median')
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    # Fill categorical columns with the most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df

def encode_features(df):
    """Encodes categorical features and prepares the feature matrix and target variable."""
    # Encode the target variable
    df['Relapse Free Status'] = df['Relapse Free Status'].apply(lambda x: 1 if x == 'Recurred' else 0)
    
    # Select relevant features and encode categorical variables
    features = ['Age at Diagnosis', 'Chemotherapy', 'ER Status', 'PR Status', 
                'HER2 status measured by SNP6', 'Neoplasm Histologic Grade', 
                'Mutation Count', 'Tumor Size', 'Radio Therapy']
    X = df[features]
    X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables
    y = df['Relapse Free Status']
    
    return X, y

def balance_classes(X, y):
    """Balances classes using SMOTE."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
