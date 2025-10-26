#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* buffer1,
    __global       uint* buffer2,
    const unsigned int buf1_sz,
    const unsigned int buf2_sz)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int local_idx = idx & 31u;
    __local unsigned int local_data[NUM_BOXES];
    __local unsigned int local_data2[NUM_BOXES];

    if (idx < buf2_sz) {
        const unsigned int base = ((idx & 0xfffffff0u) << 1);
        const unsigned int val1 = (base + local_idx < buf1_sz ? buffer1[base + local_idx] : 0u);
        const unsigned int val2 = (base + (local_idx ^ 16u) < buf1_sz ? buffer1[base + (local_idx ^ 16u)] : 0u);
        if (buf2_sz > 16u) {
            buffer2[idx] = val1 + val2;
        } else { // сразу сделаю префсумму в buf2 если он последний из буферов
            local_data[idx] = val1 + val2;
            local_data2[idx] = 0u;
            int i = idx;
            while (i >= 0) {
                local_data2[idx] += local_data[i];
                --i;
            }
            buffer2[idx] = local_data2[idx];
        }
    }
}
