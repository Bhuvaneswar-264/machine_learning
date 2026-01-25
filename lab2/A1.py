import pandas as pd
import numpy as np

def load_purchase_data(file_path):
    df = pd.read_excel(file_path, sheet_name=0) 
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values
    return X, y

def get_rank(X):
    return np.linalg.matrix_rank(X)

def get_costs(X, y):
    return np.linalg.pinv(X).dot(y)

def main():
    X, y = load_purchase_data("Lab Session Data.xlsx")

    print("Dimensionality:", X.shape[1])
    print("Number of Vectors:", X.shape[0])
    print("Rank of Matrix:", get_rank(X))
    print("Product Costs:", get_costs(X, y))

main()
