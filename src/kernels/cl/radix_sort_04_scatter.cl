#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* buffer1,
    __global const uint* buffer2,
    __global       uint* buffer3,
    unsigned int a1,
    unsigned int a2,
    unsigned int batch_num
)
{
    const uint glob_id = get_global_id(0);
    const uint threads = a1 / a2 + ((a1 % a2 == 0) ? 0 : 1);
    if (glob_id >= threads)
        return;

    const uint start = glob_id * a2;
    const uint stop  = MIN((glob_id + 1) * a2, a1);
    uint cnt[16];
    for (int k = 0; k < 16; ++k)
        cnt[k] = 0;
    for (uint i = start; i < stop; ++i) {
        const uint v = buffer1[i];
        const uint d = (v >> (batch_num * 4)) & 15;
        cnt[d] += 1;
    }

    const uint last_entry = threads - 1;
    const uint last_base = last_entry * 16;
    uint total[16];
    for (uint k = 0; k < 16; ++k) {
        total[k] = buffer2[last_base + k];
    }
    uint base[16];
    uint s = 0;
    for (int k = 0; k < 16; ++k) {
        base[k] = s;
        s += total[k];
    }
    const uint start_idx = glob_id * 16;
    uint pref[16];
    for (uint k = 0; k < 16u; ++k) {
        pref[k] = buffer2[start_idx + k] - cnt[k];
    }
    uint cur_cell[16];
    for (uint k = 0; k < 16; ++k)
        cur_cell[k] = 0;
    for (uint i = start; i < stop; ++i) {
        const uint v = buffer1[i];
        const uint d = (v >> (batch_num * 4)) & 15;
        const uint new_idx = base[d] + pref[d] + cur_cell[d];
        cur_cell[d] += 1;
        buffer3[new_idx] = v;
    }
}
