import pandas as pd

def classify_customers(payments):
    return ["RICH" if p > 200 else "POOR" for p in payments]

def main():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name=0)
    payments = df["Payment (Rs)"].values

    labels = classify_customers(payments)
    print("Customer Classification:")
    print(labels)

main()
