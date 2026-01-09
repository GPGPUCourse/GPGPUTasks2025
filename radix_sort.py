import math
import random
from typing import List, Union


def to_int(values: List[int]) -> List[int]:
    """
    Converts a list of unsigned char (0-255) to integers.
    """
    return [int(value) for value in values]


def to_bits(values: List[int]) -> List[str]:
    """
    Converts a list of integers to their binary string representation.
    """
    res = []
    for value in values:
        bit_length = 32
        res.append(bin(value)[2:].zfill(bit_length))
    # print("   ", res)
    return res


def highlight_bits(values: List[str], bit_offset: int, nbits: int) -> List[str]:
    """
    Highlights specific bits in the binary string representation.
    """
    res = []
    for bits in values:
        bits_count = len(bits)
        bits_to = bits_count - bit_offset
        assert bits_to > 0, f"Invalid bit offset: {bits_to}"
        bits_from = max(0, bits_to - nbits)
        highlighted = (
            bits[:bits_from] + "[" + bits[bits_from:bits_to] + "]" + bits[bits_to:]
        )
        res.append(highlighted)
    return res


def cut_leading_bits(values: List[str], nbits: int) -> List[str]:
    """
    Removes the leading `nbits` from the binary string representation.
    """
    res = []
    for bits in values:
        assert len(bits) > nbits, f"Invalid nbits: {nbits}"
        res.append(bits[nbits:])
    return res


def count_leading_zero_bits(value: int) -> int:
    """
    Counts the number of leading zero bits in a 32-bit integer.
    """
    if value == 0:
        return 32
    clz = 0
    while clz < 31 and (value & (0xFFFFFFFF >> (clz + 1))) == value:
        clz += 1
    return clz


def pretty_bits(
    values: List[int], max_value: int, bit_offset: int, nbits: int
) -> List[str]:
    """
    Converts a list of integers to their binary representation, removes leading zero bits,
    and highlights specific bits.
    """
    # Unit tests for count_leading_zero_bits
    assert count_leading_zero_bits(0) == 32
    assert count_leading_zero_bits(1) == 31
    assert count_leading_zero_bits(2) == 30
    assert count_leading_zero_bits(3) == 30
    assert count_leading_zero_bits(4) == 29
    assert count_leading_zero_bits(8) == 28

    meaningless_bits_count = count_leading_zero_bits(max_value)

    full_bits = to_bits(values)
    short_bits = cut_leading_bits(full_bits, meaningless_bits_count)
    highlighted_bits = highlight_bits(short_bits, bit_offset, nbits)
    return highlighted_bits


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


def map_scan(a, clazz, bits, digit):
    N = len(a)

    buf1 = [int(belongs_to_class(it, clazz, bits, digit)) for it in a]
    print("buf1", " ".join(map(str, buf1)))
    buf2 = [255] * N
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


def scatter(
    buff1: list,
    buff2: list,
    pref_sum: list,
    bits,
    digit,
    offset,
    clazz,
):
    for i in range(len(buff2)):
        if belongs_to_class(buff1[i], clazz, bits, digit):
            buff2[offset + pref_sum[i] - 1] = buff1[i]


def main(a):
    max_value = max(a)
    N = len(a)
    # print(list(map(lambda x: str(x).zfill(4), a)))
    # print(list(map(bnr, a)))

    bits = 2
    buff1 = [it for it in a]
    buff2 = [0] * N

    for digit in range(32 // bits):
        offset = 0
        for clazz in range(2**bits):
            pref_sum = map_scan(buff1, clazz, 2, digit)

            print(" ".join(pretty_bits(buff1, max_value, digit * bits, bits)))
            print(" ".join(map(str, pref_sum)))

            scatter(buff1, buff2, pref_sum, bits, digit, offset, clazz)

            print(
                f"bits={bits} digit={digit} offset={offset} clazz=[{bin(clazz)[2:].zfill(2)}]"
            )
            print(f"{offset} + {pref_sum[N - 1]}")
            offset += pref_sum[N - 1]
            print()

        buff1, buff2 = buff2, buff1

        # print(list(map(bnr, a)))

    got = buff1

    print("got", got)
    expected = list(sorted(a))
    # print("expected", expected)

    assert got == expected


a = [
    917349335,
    523921852,
    1997655251,
    917349335,
    505355764,
    1851935775,
    1247297076,
    1247297076,
]
# [random.randint(0, 100000) for _ in range(1000)]  # [3, 2, 5, 7, 8, 0, 1, 2]

main(a)
