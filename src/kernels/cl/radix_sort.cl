#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill_buffer_with_zeros(
    __global uint* buffer,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n)
        buffer[i] = 0;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* local_hist,
    unsigned int bit,
    unsigned int n
) {
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int global_sz = get_global_size(0);
    const unsigned int local_sz = get_local_size(0);

    __local unsigned int data[RADIX];
    for (unsigned int i = local_id; i < RADIX; i += local_sz) data[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = global_id; i < n; i += global_sz) {
        unsigned int b = (input[i] >> bit) & (RADIX - 1u);
        atomic_add((volatile __local unsigned int*)&data[b], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = local_id; i < RADIX; i += local_sz) {
        local_hist[i * get_num_groups(0) * get_num_groups(1) + group_id] = data[i];
    }
}

__attribute__((reqd_work_group_size(RADIX, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction_and_accumulation(
    __global       uint* local_hist,
    __global       uint* global_sums,
    unsigned int numGroups
) {
    const unsigned int global_id = get_global_id(0);

    unsigned int S = 0;
    for (unsigned int i = 0; i < numGroups; ++i) {
        S += local_hist[global_id * numGroups + i];
    }
    global_sums[global_id] = S;

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (global_id == 0) {
        unsigned int s = 0;
        for (unsigned int i = 0; i < RADIX; ++i) {
            unsigned int tmp = global_sums[i];
            global_sums[i] = s;
            s += tmp;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (global_id < RADIX) {
        unsigned int s = 0;
        unsigned int i = global_id * numGroups;
        for (unsigned int group = 0; group < numGroups; ++group) {
            unsigned int tmp = local_hist[i + group];
            local_hist[i + group] = s;
            s += tmp;
        }
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_scatter(
    __global const uint* input_keys,
    __global const uint* input_vals,
    __global       uint* output_keys,
    __global       uint* output_vals,
    __global const uint* global_offsets,
    __global const uint* local_offsets,
    unsigned int bit,
    unsigned int n)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_sz = get_local_size(0);
    const unsigned int num_groups = get_num_groups(0) * get_num_groups(1);

    __local unsigned int ldata[RADIX];

    for (unsigned int i = local_id; i < RADIX; i += local_sz) {
        ldata[i] = global_offsets[i] + local_offsets[i * num_groups + group_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id >= n) return;

    uint k = input_keys[global_id];
    uint val = input_vals[global_id];
    unsigned int b = (k >> bit) & (RADIX - 1u);

    unsigned int local_rank = 0;
    const unsigned int group_start = group_id * local_sz;
    for (unsigned int t = 0; t < local_id; ++t) {
        unsigned int b2 = (input_keys[group_start + t] >> bit) & (RADIX - 1u);
        if (b2 == b) local_rank++;
    }
    uint pos = ldata[b] + local_rank;
    output_keys[pos] = k;
    output_vals[pos] = val;
}