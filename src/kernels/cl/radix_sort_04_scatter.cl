#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,           // [n]
    __global const uint* zeroPrefix,      // [n]
    __global const uint* zeroGroupOffset, // [groups] exclusive
    __global const uint* totalZerosBuf,   // [1]
    __global       uint* output,          // [n]
    uint n,
    uint bit_number
) {
    const uint gid = get_global_id(0);
    if (gid >= n) return;

    const uint lid   = get_local_id(0);
    const uint group = get_group_id(0);

    const uint x    = input[gid];
    const uint bit  = (x >> bit_number) & 1u;

    const uint zPref = zeroPrefix[gid];
    const uint zOff  = zeroGroupOffset[group];

    if (bit == 0u) {
        const uint pos = zOff + zPref;
        output[pos] = x;
    } else {
        const uint totalZeros = totalZerosBuf[0];
        const uint base = group * GROUP_SIZE;

        const uint onesBeforeGroup = base - zOff;  // total elements before group minus zeros before group
        const uint onePref = lid - zPref;          // ones before this element inside group

        const uint pos = totalZeros + onesBeforeGroup + onePref;
        output[pos] = x;
    }
}
