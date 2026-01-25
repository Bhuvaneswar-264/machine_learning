import pandas as pd

def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

def main():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name=2)
    numeric_df = df.select_dtypes(include="number")

    normalized_df = normalize(numeric_df)
    print("Normalized Data:")
    print(normalized_df.head())

main()
