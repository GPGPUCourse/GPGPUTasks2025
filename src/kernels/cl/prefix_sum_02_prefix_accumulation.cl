#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* reduced, 
    __global       uint* pref_sum,
    unsigned int n,
    unsigned int p // 2^p is a segment size
)
{
    unsigned int i = get_global_id(0);
    unsigned ind = i + 1;

    if (i < n)
    {
        // 2^p contributes to i
        if ((ind >> p) & 1)
        {
            unsigned int segment_ind = (ind >> (p + 1)) << 1;
            pref_sum[i] += reduced[segment_ind];
        }
    }
}
