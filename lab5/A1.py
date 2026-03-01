import numpy as np
import pandas as pd

def load_dataset(filepath):
    print("Loading dataset from:", filepath)
    data = pd.read_csv(filepath)
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.fillna(numeric_data.mean())
    print("Dataset loaded successfully!")
    return numeric_data.values