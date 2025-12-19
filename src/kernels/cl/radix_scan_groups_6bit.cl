#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__kernel void radix_scan_groups_6bit(
    __global const uint* groupHist,
    __global uint* groupPrefix,
    uint numGroups)
{
    uint bin = get_global_id(0);
    if (bin >= 64) return;

    uint sum = 0;
    for (uint g = 0; g < numGroups; ++g) {
        uint c = groupHist[g * 64u + bin];
        groupPrefix[g * 64u + bin] = sum;
        sum += c;
    }
}
