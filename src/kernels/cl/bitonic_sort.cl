#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void bitonic_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  n)
{
    const int i = get_global_id(0);
    const int local_id = get_local_id(0);
    __local uint local_data[GROUP_SIZE];
    local_data[local_id] = (i < n ? input_data[i] : UINT_MAX);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 2; k <= GROUP_SIZE; k <<= 1) {
        for (int j = k; j >= 2; j >>= 1) {
            int friend = local_id ^ (j >> 1);
            int asc = ((local_id / k) & 1) == 0;
            if (local_id < friend) {
                if (asc == (local_data[local_id] > local_data[friend])) {
                    uint tmp = local_data[local_id];                                                                                   
                    local_data[local_id] = local_data[friend];                                                                        
                    local_data[friend] = tmp;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    if (i < n) {
        output_data[i] = local_data[local_id];
    }
}
