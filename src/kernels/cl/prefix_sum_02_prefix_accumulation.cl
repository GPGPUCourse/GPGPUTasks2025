#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global       uint* buf1,
    __global const uint* buf2,
    const unsigned int n,
    const unsigned int need_to_add_buf3,
    __global const uint* buf3
  )
{
    __local unsigned int data[GROUP_SIZE >> BATCH_LG];
    __local unsigned int buf3data[GROUP_SIZE];

    const unsigned int idx = get_global_id(0);
    unsigned int local_idx = get_local_id(0);

    if (local_idx < (GROUP_SIZE >> BATCH_LG)) {
        const unsigned int pt = ((idx - local_idx) >> BATCH_LG) + local_idx - 1;
        if (pt < (n >> BATCH_LG)) {
            data[local_idx] = buf2[pt];
        } else {
            data[local_idx] = 0;
        }
    }

    if (need_to_add_buf3) {
        buf3data[local_idx] = buf3[idx];
    } else {
        buf3data[local_idx] = buf1[idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx < n) {
        unsigned int summ = data[local_idx >> BATCH_LG];
        summ += buf3data[local_idx];
        while (local_idx & BATCH_MASK) {
            --local_idx;
            summ += buf3data[local_idx];
        }
        buf1[idx] = summ;
    }
}
