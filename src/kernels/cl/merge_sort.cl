#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned i = get_global_id(0);
    if(i >= n)
        return;
    unsigned jbase = ((i >> sorted_k) ^ 1) << sorted_k;
    unsigned diff = (i >> sorted_k) & 1; // we're guaranteed no 0s so subtraction is safe
    unsigned j = 0;
    unsigned R = max(0, min(1 << sorted_k, n - (int)(jbase)));
    while(R) {
        j += (input_data[jbase + j + R / 2] - diff < input_data[i]) ? R - R / 2 : 0;
        R /= 2;
    }
    output_data[(i &~ (1 << sorted_k)) + j] = input_data[i];
}
