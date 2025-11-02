#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge(
    __global const uint* input_data,
    __global uint* output_data,
    __global const uint* a_inds,
    __global const uint* b_inds,
    __global uint* sparse_buf,
    uint sorted_n,
    uint n
) {
    uint lbound = get_group_id(0) * TILE_SIZE;
    uint rbound = lbound + TILE_SIZE;
    uint lid = get_local_id(0);
    uint l = lbound / (2 * sorted_n) * (2 * sorted_n);
    uint m = l + sorted_n;

    __local uint a_start, b_start, a_end, b_end;
    if (lid == 0) {
        a_start = a_inds[lbound / WRITE_EVERY] * WRITE_EVERY;
        b_start = b_inds[lbound / WRITE_EVERY] * WRITE_EVERY;
        if (rbound == m + sorted_n) {
            a_end = sorted_n;
            b_end = sorted_n;
        } else {
            a_end = (a_inds[rbound / WRITE_EVERY] + 1) * WRITE_EVERY;
            b_end = (b_inds[rbound / WRITE_EVERY] + 1) * WRITE_EVERY;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __local uint a_buf[TILE_SIZE + 2 * WRITE_EVERY], b_buf[TILE_SIZE + 2 * WRITE_EVERY];
    for (int shift = 0; shift < TILE_SIZE + 2 * WRITE_EVERY; shift += GROUP_SIZE) {
        if (a_start + shift + lid < a_end) {
            a_buf[shift + lid] = input_data[l + a_start + shift + lid];
        }
        if (b_start + shift + lid < b_end) {
            b_buf[shift + lid] = input_data[m + b_start + shift + lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // printf("gid=%u lbound=%u rbound=%u\n l=%u m=%u\n", get_global_id(0), lbound, rbound, l, m);
    // printf("gid=%u a_start=%u a_end=%u b_start=%u b_end=%u\n", get_global_id(0), a_start, a_end, b_start, b_end);
    // printf("gid=%u a_buf={%u, %u, %u, %u, %u}\n", get_global_id(0), a_buf[0], a_buf[1], a_buf[2], a_buf[3], a_buf[4]);
    // printf("gid=%u b_buf={%u, %u, %u, %u, %u}\n", get_global_id(0), b_buf[0], b_buf[1], b_buf[2], b_buf[3], b_buf[4]);

    __local uint res[TILE_SIZE + 4 * WRITE_EVERY];
    
    for (int shift = 0; shift < (a_end - a_start); shift += GROUP_SIZE) {
        uint a_idx = shift + lid;
        if (a_start + a_idx >= a_end) {
            continue;
        }
        uint a_elem = a_buf[a_idx];
        int lhs = -1, rhs = b_end - b_start;
        while (rhs - lhs > 1) {
            uint mid = (lhs + rhs) / 2;
            if (b_buf[mid] >= a_elem) {
                rhs = mid;
            } else {
                lhs = mid;
            }
        }
        // if (get_global_id(0) == 1) {
        //     printf("a_elem=%u rhs=%u\n", a_elem, rhs);
        // }
        res[a_idx + rhs] = a_elem;
    }
    for (int shift = 0; shift < (b_end - b_start); shift += GROUP_SIZE) {
        int b_idx = shift + lid;
        if (b_start + b_idx >= b_end) {
            continue;
        }
        uint b_elem = b_buf[b_idx];
        int lhs = -1, rhs = a_end - a_start;
        while (rhs - lhs > 1) {
            uint mid = (lhs + rhs) / 2;
            if (a_buf[mid] > b_elem) {
                rhs = mid;
            } else {
                lhs = mid;
            }
        }
        // if (get_global_id(0) == 1) {
        //     printf("b_elem=%u rhs=%u\n", b_elem, rhs);
        // }
        res[b_idx + rhs] = b_elem;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // for (int i = 0; i < 2 * TILE_SIZE + 4 * WRITE_EVERY; ++i) {
    //     printf("gid=%u res[%u]=%u\n", get_global_id(0), i, res[i]);
    // }

    uint lol = lbound - l - a_start - b_start;
    for (int shift = 0; shift < TILE_SIZE; shift += GROUP_SIZE) {
        output_data[lbound + shift + lid] = res[shift + lid + lol];
        // printf("gid=%u write %u from res[%u]\n", get_global_id(0), lbound + shift + lid, shift + lid + lol);
        if ((shift + lid) % WRITE_EVERY == 0) {
            sparse_buf[(lbound + shift + lid) / WRITE_EVERY] = res[shift + lid + lol];
        }
    }
}
