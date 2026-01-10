
def matrix_power(A, m):
    def multiply(X, Y):
        result = [[0]*len(Y[0]) for _ in range(len(X))]
        for i in range(len(X)):
            for j in range(len(Y[0])):
                for k in range(len(Y)):
                    result[i][j] += X[i][k] * Y[k][j]
        return result

    result = A
    for _ in range(m - 1):
        result = multiply(result, A)
    return result

if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    print(matrix_power(A, 2))
