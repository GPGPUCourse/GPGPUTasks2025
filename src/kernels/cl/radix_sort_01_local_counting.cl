#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1,
    unsigned int a2,
    unsigned int batch_num
)
{
    const uint glob_id = get_global_id(0);
    const uint number_working_threads = a1 / a2 + ((a1 % a2 == 0) ? 0 : 1);
    if (glob_id >= number_working_threads)
        return;
    const uint out_base = glob_id * 16;
    const uint start = glob_id * a2;
    const uint stop  = MIN(start + a2, a1);
    uint counts[16];
    for (int k = 0; k < 16; ++k)
        counts[k] = 0;
    for (uint idx = start; idx < stop; ++idx) {
        const uint v = buffer1[idx];
        const uint d = (v >> (batch_num * 4)) & 15;
        counts[d] += 1;
    }
    for (uint k = 0; k < 16; ++k) {
        buffer2[out_base + k] = counts[k];
    }
}
