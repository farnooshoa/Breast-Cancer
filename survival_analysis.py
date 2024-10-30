# survival_analysis.py
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

def plot_kaplan_meier(df, group_by):
    """Plots Kaplan-Meier survival curves for different groups."""
    kmf = KaplanMeierFitter()
    
    # Plot survival curves for each group within the specified column
    plt.figure(figsize=(10, 6))
    
    for group in df[group_by].unique():
        mask = df[group_by] == group
        kmf.fit(df[mask]['Overall Survival (Months)'], 
                event_observed=df[mask]['Overall Survival Status'] == 'Deceased')
        kmf.plot(label=str(group))
    
    plt.title(f"Kaplan-Meier Survival Curves by {group_by}")
    plt.xlabel("Months")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.show()

def fit_cox_model(df):
    """Fits a Cox Proportional-Hazards Model to identify factors affecting survival."""
    # Convert the 'Overall Survival Status' to 1 for deceased and 0 for living
    df['Overall Survival Status'] = df['Overall Survival Status'].apply(lambda x: 1 if x == 'Deceased' else 0)

    # Selecting relevant features and encoding categorical ones
    survival_data = df[['Overall Survival (Months)', 'Overall Survival Status', 
                        'Age at Diagnosis', 'Chemotherapy', 'ER Status', 
                        'PR Status', 'HER2 status measured by SNP6', 
                        'Neoplasm Histologic Grade', 'Mutation Count']]
    survival_data = pd.get_dummies(survival_data, drop_first=True)

    # Fitting the Cox model
    cph = CoxPHFitter()
    cph.fit(survival_data, duration_col='Overall Survival (Months)', event_col='Overall Survival Status')
    cph.print_summary()  # Display model summary
    return cph
