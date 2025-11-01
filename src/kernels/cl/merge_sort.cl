#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

#define ll long long

inline int get_value_of_a(__global const uint* a, int ind, int offset, uint sorted_block_len, uint n)
{
    if (ind == 0)
        return -1;
    if (ind > sorted_block_len)
        return INT_MAX;

    ind -= 1;

    if (ind + offset >= n)
        return INT_MAX;

    return (ll)(a[offset + ind]);
}

__kernel void merge_sort(
    __global const uint* a,
    __global uint* sorted_segments,
    uint sorted_block_len,
    int n)
{
    unsigned int i = get_global_id(0);
    unsigned int k = i % (2 * sorted_block_len);
    unsigned int pair_num_to_merge = i / (2 * sorted_block_len);
    unsigned int thr = get_local_id(0);

    int offset_x = pair_num_to_merge * 2 * sorted_block_len;
    int offset_y = offset_x + sorted_block_len;

    // if (k != 1)
    //     return;

    // printf("i = %u k = %u pair_id = %u offset_x = %u offset_y = %u\n", i, k, pair_num_to_merge, offset_x, offset_y);

    int x_prev, x, y_prev, y;
    int l = max(0, (ll)k - (ll)sorted_block_len);
    int r = min(k, sorted_block_len) + 1;
    // printf("l = %d r = %d\n", l, r);
    while (true) {
        int m_x = (l + r) / 2;
        int m_y = k - m_x;

        x_prev = get_value_of_a(a, m_x, offset_x, sorted_block_len, n);
        x = get_value_of_a(a, m_x + 1, offset_x, sorted_block_len, n);
        y_prev = get_value_of_a(a, m_y, offset_y, sorted_block_len, n);
        y = get_value_of_a(a, m_y + 1, offset_y, sorted_block_len, n);
        // printf("a[%lld] = %lld\n", m_x, x);
        // printf("b[%lld] = %lld\n", m_y, y);

        // printf("m_x = %ld m_y = %ld\n", m_x, m_y);
        // printf("x = %ld y_prev = %ld y = %ld x_prev = %ld\n", x, y_prev, y, x_prev);
        if (y < x_prev) {
            r = m_x;
        } else if (x < y_prev) {
            l = m_x;
        } else {
            l = m_x;
            break;
        }
    }

    // printf("anw: %ld %ld\n", l, k - l);

    // k -= 1;

    if (offset_x + k < n) {
        uint x = get_value_of_a(a, l + 1, offset_x, sorted_block_len, n);
        uint y = get_value_of_a(a, k - l + 1, offset_y, sorted_block_len, n);
        sorted_segments[offset_x + k] = min(x, y);
        // printf("min{%u = a[%u] %u = a[%u]} -> %u\n", x, offset_x + l, y, offset_y + k - l, offset_x + k);
    }
}
