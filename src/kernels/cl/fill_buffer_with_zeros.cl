#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Simple utility: clear buffer by writing zeros
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill_buffer_with_zeros(
    __global uint* buffer,
    unsigned int n)
{
    const unsigned int work_item_id = get_global_id(0);
    if (work_item_id < n) {
        buffer[work_item_id] = 0;
    }
}
