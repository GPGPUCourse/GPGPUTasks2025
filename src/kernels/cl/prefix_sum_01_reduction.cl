#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* buf1,
    __global       uint* buf2,
    const unsigned int n)
{
    __local unsigned int data[GROUP_SIZE];
    __local unsigned int prefix_sums[GROUP_SIZE];
    const unsigned int idx = get_global_id(0);
    const unsigned int local_idx = get_local_id(0);

    if (idx >= n) {
        return;
    }

    if (n <= WARP_SIZE) { // then actually buf1 == buf2
        data[idx] = buf1[idx];

        unsigned int local_pref_summ = 0;
        int i = idx;
        while (i >= 0) {
            local_pref_summ += data[i];
            --i;
        }
        buf2[idx] = local_pref_summ;

    } else {
        data[local_idx] = buf1[idx];

        unsigned int local_pref_summ = 0;
        unsigned int i = local_idx;
        while (i & BATCH_MASK) {
            local_pref_summ += data[i];
            --i;
        }
        local_pref_summ += data[i];

        if (!((local_idx + 1) & BATCH_MASK)) {
            prefix_sums[local_idx >> BATCH_LG] = local_pref_summ;
        }

        barrier(CLK_LOCAL_MEM_FENCE); // надо дождаться пока префсуммы посчитаются во всех ворпах, и потом одновременно отправить их в vram

        if (local_idx < (GROUP_SIZE >> BATCH_LG) && ((idx - local_idx) >> BATCH_LG) + local_idx < (n >> BATCH_LG)) {
            buf2[((idx - local_idx) >> BATCH_LG) + local_idx] = prefix_sums[local_idx];
        }
    }
}
