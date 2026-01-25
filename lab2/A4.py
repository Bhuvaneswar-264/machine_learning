import pandas as pd

def explore_data(df):
    summary = df.describe(include="all")
    missing = df.isnull().sum()
    return summary, missing

def main():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name=2)

    summary, missing = explore_data(df)

    print("Data Summary:")
    print(summary)

    print("\nMissing Values:")
    print(missing)

main()
