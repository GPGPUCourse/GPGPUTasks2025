#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* offsets,  // [numGroups * 2]: {zeroOffset[g], oneOffset[g]}
                   __global uint* out_values,
    unsigned int n,
    unsigned int bit)
{
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint idx = gid * GROUP_SIZE + lid;

    uint valid = idx < n ? 1u : 0u;
    uint v = valid ? values[idx] : 0u;
    uint flagZero = valid ? (uint)(((v >> bit) & 1u) == 0u) : 0u;
    uint flagOne  = valid ? (uint)(((v >> bit) & 1u) == 1u) : 0u;

    __local uint l_zero[GROUP_SIZE];
    __local uint l_one[GROUP_SIZE];
    l_zero[lid] = flagZero;
    l_one[lid]  = flagOne;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint ofs = 1u; ofs < GROUP_SIZE; ofs <<= 1u) {
        uint addZ = (lid >= ofs) ? l_zero[lid - ofs] : 0u;
        uint addO = (lid >= ofs) ? l_one[lid - ofs]  : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        l_zero[lid] += addZ;
        l_one[lid]  += addO;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (valid) {
        uint zeroBase = offsets[gid * 2 + 0];
        uint oneBase  = offsets[gid * 2 + 1];
        uint zeroRank = l_zero[lid] - flagZero; // exclusive
        uint oneRank  = l_one[lid]  - flagOne;  // exclusive
        uint pos = flagZero ? (zeroBase + zeroRank) : (oneBase + oneRank);
        out_values[pos] = v;
    }
}