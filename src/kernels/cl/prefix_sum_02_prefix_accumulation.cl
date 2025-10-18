#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* input_buffer,  
    __global       uint* output_buffer, 
    unsigned int n,
    unsigned int offset) 
{
    const unsigned int index = get_global_id(0);
    
    if (index >= n)
        return;
    
    if (index >= offset) {
        output_buffer[index] = input_buffer[index] + input_buffer[index - offset];
    } else {
        output_buffer[index] = input_buffer[index];
    }
}
