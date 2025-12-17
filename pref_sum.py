import math
import itertools


def bnr(max, i):
    return bin(i)[2:].zfill(len(bin(max - 1)[2:]))


def reduce(pow2_sum: list, pow2_sum_next: list, n: int):
    for i in range(n):
        pow2_sum_next[i] = pow2_sum[2 * i] + pow2_sum[2 * i + 1]
    print("reduce", pow2_sum_next[: n])


def acc(pow2_sum: list, prefix_sum_accum: list, pow2: int):
    # print("acc", k, pow2_sum)
    for i in range(len(prefix_sum_accum)):
        # print(i)
        # print(bnr(len(prefix_sum_accum), i))
        # print(bnr(len(prefix_sum_accum), 1 << k))
        T = 2
        pow_i = (i >> (pow2 + 1)) << 1  # k = 2: 01101 -> 010
        # if i == T:
        #     print(f"i: {i}, cond: {int(i & (1 << pow2))} pow_i: {pow_i}")
        # print(bnr(len(prefix_sum_accum), pow_i))
        if i & (1 << pow2):
            # if i == T:
            #     print("== add", prefix_sum_accum[i], pow2_sum[pow_i])
            prefix_sum_accum[i] += pow2_sum[pow_i]
        # print()
    print("acc", prefix_sum_accum)


a = [3, 2, 5, 7, 8, 0, 1, 2]
N = len(a)

buf1 = [it for it in a]
buf2 = [0] * N
pref_sum = [it for it in a]

acc(buf1, pref_sum, 0)
for k in range(int(math.floor(math.log2(N)))):
    reduce(buf1, buf2, N // 2 ** (k + 1))
    acc(buf2, pref_sum, k + 1)

    buf1, buf2 = buf2, buf1

print(pref_sum)
print(list(itertools.accumulate(a)))
print([a - b for a, b in zip(itertools.accumulate(a), pref_sum)])

assert pref_sum == list(itertools.accumulate(a))
