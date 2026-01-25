
import random
from statistics import mean, median, mode

def stats_random():
    nums = [random.randint(1, 10) for _ in range(25)]
    return nums, mean(nums), median(nums), mode(nums)

if __name__ == "__main__":
    print(stats_random())
