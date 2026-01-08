import math
import random


def bnr(i, width=4):
    return bin(i)[2:].zfill(width)


def reduce(pow2_sum: list, pow2_sum_next: list, n: int):
    for i in range(n):
        pow2_sum_next[i] = pow2_sum[2 * i] + pow2_sum[2 * i + 1]
    # print("reduce", pow2_sum_next[:n])


def acc(pow2_sum: list, prefix_sum_accum: list, pow2: int):
    for i in range(len(prefix_sum_accum)):
        pow_i = (i >> (pow2 + 1)) << 1  # k = 2: 01101 -> 010
        if i & (1 << pow2):
            prefix_sum_accum[i] += pow2_sum[pow_i]
    # print("acc", pow2, prefix_sum_accum)


def calc_pref_sum(a, clazz, bits, digit):
    N = len(a)

    buf1 = [int(belongs_to_class(it, clazz, bits, digit)) for it in a]
    buf2 = [0] * N
    pref_sum = [int(belongs_to_class(it, clazz, bits, digit)) for it in a]
    acc(buf1, pref_sum, 0)
    for k in range(int(math.floor(math.log2(N)))):
        reduce(buf1, buf2, N // 2 ** (k + 1))
        acc(buf2, pref_sum, k + 1)

        buf1, buf2 = buf2, buf1

    return pref_sum


def belongs_to_class(v, clazz, bits, digit):
    mask = (1 << bits) - 1
    return ((v >> (digit * bits)) & mask) == clazz


def scatter(buff2, bits, digit, offset, buff1, clazz, pref_sum):
    for i in range(len(buff2)):
        if belongs_to_class(buff1[i], clazz, bits, digit):
            buff2[offset + pref_sum[i] - 1] = buff1[i]


def main(a):
    N = len(a)
    # print(list(map(lambda x: str(x).zfill(4), a)))
    print(list(map(bnr, a)))

    bits = 2
    buff1 = [it for it in a]
    buff2 = [0] * N

    for digit in range(32 // bits):
        offset = 0
        for clazz in range(2**bits):
            pref_sum = calc_pref_sum(buff1, clazz, 2, digit)

            scatter(buff2, bits, digit, offset, buff1, clazz, pref_sum)

            offset += pref_sum[N - 1]

        buff1, buff2 = buff2, buff1

        print(list(map(bnr, a)))

    got = buff2

    print("got", got)
    expected = list(sorted(a))
    print("expected", expected)

    assert got == expected


a = [random.randint(0, 100000) for _ in range(1000)]  # [3, 2, 5, 7, 8, 0, 1, 2]

main(a)
