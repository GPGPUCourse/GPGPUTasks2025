#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float *matrix,      // h x w
    __global float *transposed_matrix, // w x h
    uint w,
    uint h)
{
    const uint H = GROUP_SIZE_X;
    const uint W = GROUP_SIZE_Y;
    __local float local_data[H * (W + 1)];

    const uint height = get_global_id(0);
    const uint width = get_global_id(1);
    const uint local_height = get_local_id(0);
    const uint local_width = get_local_id(1);

    const uint t_local_height = (local_height * W + local_width) % H;
    const uint t_local_width = (local_height * W + local_width) / H;
    const uint t_height = (width - local_width) + local_height;
    const uint t_width = (height - local_height) + local_width;
    // printf("(%d, %d)(%d, %d) -> (%d, %d)(%d, %d)\n", height, width, local_height, local_width, t_height, t_width, t_local_height, t_local_width);
    if (width < w && height < h)
    {
        local_data[local_height * (W + 1) + local_width] = matrix[height * w + width];
    }
    else
    {
        local_data[local_height * (W + 1) + local_width] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (t_width < h && t_height < w)
    {

        transposed_matrix[t_height * h + t_width] = local_data[t_local_height * (W + 1) + t_local_width];
    }
}
