import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    data = np.random.rand(20, 20)
    sns.heatmap(data, annot=True)
    plt.show()

main()
