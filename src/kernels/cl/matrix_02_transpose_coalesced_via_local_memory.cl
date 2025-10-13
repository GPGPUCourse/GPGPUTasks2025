#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= w || y >= h) return;

    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);

    const uint local_data_width = GROUP_SIZE_X + 1;
    const uint local_data_height = GROUP_SIZE_Y;
    const uint local_data_size = local_data_width * local_data_height;
    __local float local_data[local_data_size];
    local_data[local_y * local_data_width + local_x] = matrix[y * w + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint moved_y = x - local_x + local_y;
    const uint moved_x = y - local_y + local_x;
    transposed_matrix[moved_y * h + moved_x] = local_data[local_x * local_data_width + local_y];

}
