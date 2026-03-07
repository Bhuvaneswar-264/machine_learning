import pandas as pd

df = pd.read_csv("BERT_embeddings (4).csv")

def binning(data, bins=4, method="width"):
    if method == "width":
        return pd.cut(data, bins=bins, labels=False)
    if method == "frequency":
        return pd.qcut(data, q=bins, labels=False, duplicates='drop')

df['0_binned'] = binning(df['0'], 4, "width")
print(df[['0','0_binned']].head())