#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
fill_buffer_with_zeros(__global uint* data, unsigned int n)
{
    const uint gid = get_global_id(0);
    if (gid < n)
        data[gid] = 0u;
}