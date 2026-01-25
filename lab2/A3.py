import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def manual_mean(data):
    return sum(data) / len(data)

def manual_variance(data):
    mean = manual_mean(data)
    return sum((x - mean) ** 2 for x in data) / len(data)

def main():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name=1)

    price = df.iloc[:, 3].dropna().values
    chg = df.iloc[:, 8].dropna().values
    days = df.iloc[:, 1].values

    print("Numpy Mean:", np.mean(price))
    print("Numpy Variance:", np.var(price))
    print("Manual Mean:", manual_mean(price))
    print("Manual Variance:", manual_variance(price))

    loss_prob = len(list(filter(lambda x: x < 0, chg))) / len(chg)
    print("Probability of Loss:", loss_prob)

    wed_profit = sum(
        1 for i in range(len(chg))
        if days[i] == "Wednesday" and chg[i] > 0
    )
    wed_total = sum(1 for d in days if d == "Wednesday")

    if wed_total == 0:
        print("No Wednesday data available in dataset")
    else:
        print("Profit Probability on Wednesday:", wed_profit / wed_total)

    plt.scatter(days, chg)
    plt.xlabel("Day")
    plt.ylabel("Change %")
    plt.show()

main()
