
def count_pairs_with_sum(lst, target):
    count = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] + lst[j] == target:
                count += 1
    return count

if __name__ == "__main__":
    lst = [2, 7, 4, 1, 3, 6]
    print(count_pairs_with_sum(lst, 10))
