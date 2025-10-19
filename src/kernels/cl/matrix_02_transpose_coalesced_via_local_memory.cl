#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    int offset_x = get_group_id(0) * 16;
    int offset_y = get_group_id(1) * 16;
    int x = get_local_id(0);
    int y = get_local_id(1);
    int input_x = offset_x + x;
    int input_y = offset_y + y;
    int output_x = offset_y + x;
    int output_y = offset_x + y;
    __local float local_data[16][16];
    if (input_x < w && input_y < h) {
        local_data[y][x] = matrix[input_y * w + input_x];
    } else {
        local_data[y][x] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (output_x < h && output_y < w) {
        transposed_matrix[output_y * h + output_x] = local_data[x][y];
    }
}
