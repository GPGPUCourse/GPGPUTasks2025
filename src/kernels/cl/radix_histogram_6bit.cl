#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__kernel void radix_histogram_6bit(
    __global const uint* keys_in,
    __global uint* groupHist,
    uint n,
    uint shiftBits)
{
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint lsz = get_local_size(0);

    __local uint lhist[64];

    for (uint b = lid; b < 64; b += lsz)
        lhist[b] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint idx = gid * lsz + lid;
    if (idx < n) {
        uint key = keys_in[idx];
        uint bin = (key >> shiftBits) & 63u;
        atomic_inc(&lhist[bin]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint b = lid; b < 64; b += lsz)
        groupHist[gid * 64u + b] = lhist[b];
}
