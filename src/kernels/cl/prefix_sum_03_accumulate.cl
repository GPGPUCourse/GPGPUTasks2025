#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_03_accumulate(
    __global const uint* in,
    __global const uint* sparse,
    __global uint* out,
    unsigned int n)
{
    __local uint buf[GROUP_SIZE];
    __local uint buf2[32];
    size_t global_idx = get_global_id(0);
    size_t local_idx = get_local_id(0);

    buf[local_idx] = in[global_idx];
    if (local_idx == 0) {
        // global_idx should % GROUP_SIZE
        size_t trunc_global_idx = global_idx >> GROUP_SIZE_LOG;
        size_t offset = 0;
        for (int i = 31 - GROUP_SIZE_LOG; i >= 0; --i) {
            if ((trunc_global_idx & (1 << i)) != 0) {
                offset += (trunc_global_idx & (1 << i));
                buf2[i] = offset;
            }
        }
    }
    if (local_idx < 32) {
        if (buf2[local_idx] != 0) {
            buf2[local_idx] = sparse[buf2[local_idx] - 1];
        } else {
            buf2[local_idx] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_idx == 0) {
        uint pref_sum = 0;
        for (int i = 0; i < 32; ++i) {
            pref_sum += buf2[i];
        }
        buf[0] += pref_sum;
        for (int i = 1; i < GROUP_SIZE; ++i) {
            buf[i] += buf[i - 1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    out[global_idx] = buf[local_idx];
}
