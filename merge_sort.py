import random
import math
import matplotlib.pyplot as plt
import numpy as np



def merge(N, input, output, i):
    bucket_size = 2 * N
    bucket_idx = i // bucket_size
    bucket_start = bucket_idx * bucket_size
    a = input[(bucket_start) : (bucket_start + N)]
    b = input[(bucket_start + N) : (bucket_start + bucket_size)]

    # assert N == len(a) and N == len(b)

    # if
    index_in_bucket = i - bucket_start

    if index_in_bucket >= N:
        to_compare = a
        r = N
    else:
        to_compare = b
        r = max(0, min(N, len(input) - (bucket_start + N)))


    l = -1
    while l < r - 1:
        m = l + (r - l) // 2

        if index_in_bucket >= N:
            cond = to_compare[m] <= input[i]
        else:
            cond = to_compare[m] < input[i]

        if cond:
            l = m
        else:
            r = m
    idx = r

    if index_in_bucket >= N:
        index_in_bucket -= N

    destination_index = bucket_start + index_in_bucket + idx
    # print("+", i, bucket_start, index_in_bucket, idx, destination_index, input[i])

    output[destination_index] = input[i]


def main(a):
    N = len(a)

    buf1 = [i for i in a]
    c = [-228] * N

    for sorted_k in range(int(math.log2(N)) + 1):
        print(2**sorted_k)
        for i in range(0, N):
            merge(2**sorted_k, buf1, c, i)

        # print(sorted_k, c)
        # print()
        tmp = c
        c = buf1
        buf1 = tmp

    c = buf1
    # print(c)
    expected = sorted(a)
    # print(expected)
    assert c == expected
    # assert list(map(lambda x: x[1], c)) == expected


if __name__ == "__main__":
    with open("example.txt") as f:
        a = list(map(int, f.readline().strip().split(" ")))
    print(len(a))
    # a = [6, 5, 4, 3, 2, 1, 0]
    # a = [7, 6, 5, 4, 3, 2, 1, 0]
    main(a)
