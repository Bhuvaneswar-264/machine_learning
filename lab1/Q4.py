
def highest_occurring_char(s):
    freq = {}
    for ch in s.lower():
        if ch.isalpha():
            freq[ch] = freq.get(ch, 0) + 1
    max_char = max(freq, key=freq.get)
    return max_char, freq[max_char]

if __name__ == "__main__":
    s = "hippopotamus"
    print(highest_occurring_char(s))
