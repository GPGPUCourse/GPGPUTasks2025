#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void fill_buffer_with_zeros(
    __global uint* buffer,
    unsigned int n)
{
    for (uint i = 0; i < n; i++) {
        buffer[i] = 0u;
    }
}
