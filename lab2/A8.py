import pandas as pd
import numpy as np

def impute_thyroid_data(file_path):
    # Load thyroid data
    df = pd.read_excel(file_path, sheet_name=2)

    # Replace '?' with NaN
    df = df.replace('?', np.nan)

    # Numeric and categorical columns
    numeric_cols = ['age','TSH','T3','TT4','T4U','FTI','TBG']
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Convert numeric columns properly
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Mean imputation for age
    df['age'] = df['age'].fillna(df['age'].mean())

    # Median imputation for hormone values (outliers present)
    hormone_cols = ['TSH','T3','TT4','T4U','FTI','TBG']
    df[hormone_cols] = df[hormone_cols].fillna(df[hormone_cols].median())

    # Mode imputation for categorical attributes
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def main():
    df = impute_thyroid_data("Lab Session Data.xlsx")

    print("Missing values after imputation:")
    print(df.isnull().sum())

main()
