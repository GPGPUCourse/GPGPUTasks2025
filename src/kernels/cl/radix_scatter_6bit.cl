#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__kernel void radix_scatter_6bit(
    __global const uint* keys_in,
    __global const uint* vals_in,
    __global uint* keys_out,
    __global uint* vals_out,
    __global const uint* groupPrefix,
    uint n,
    uint shiftBits)
{
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint lsz = get_local_size(0);

    __local uint lhist[64];
    __local uint lbase[64];
    __local uint lrun[64];

    for (uint b = lid; b < 64; b += lsz)
        lhist[b] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint idx = gid * lsz + lid;
    uint bin = 0;
    uint key = 0;
    uint val = 0;
    if (idx < n) {
        key = keys_in[idx];
        val = vals_in[idx];
        bin = (key >> shiftBits) & 63u;
        atomic_inc(&lhist[bin]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint s = 0;
        for (uint b = 0; b < 64; ++b) {
            lbase[b] = s;
            lrun[b] = 0;
            s += lhist[b];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx < n) {
        uint r = atomic_inc(&lrun[bin]);
        uint outIdx = groupPrefix[gid * 64u + bin] + lbase[bin] + r;
        keys_out[outIdx] = key;
        vals_out[outIdx] = val;
    }
}
