import numpy as np

def jaccard(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    return f11 / (f11 + f10 + f01)

def smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    return (f11 + f00) / (f11 + f00 + f10 + f01)

def main():
    v1 = np.array([1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 1])

    print("Jaccard Coefficient:", jaccard(v1, v2))
    print("Simple Matching Coefficient:", smc(v1, v2))

main()
