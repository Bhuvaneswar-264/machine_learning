
def find_range(lst):
    if len(lst) < 3:
        return "Range determination not possible"
    return max(lst) - min(lst)

if __name__ == "__main__":
    lst = [5, 3, 8, 1, 0, 4]
    print(find_range(lst))
