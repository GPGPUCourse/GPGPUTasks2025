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
    __local float cache[16][16];
    unsigned int global_x = get_global_id(0);
    unsigned int global_y = get_global_id(1);
    unsigned int block_x = global_x / 16;
    unsigned int block_y = global_y / 16;
    unsigned int local_x = global_x % 16;
    unsigned int local_y = global_y % 16;
    unsigned int x = block_x * 16 + local_x;
    unsigned int y = block_y * 16 + local_y;

    if (x < w && y < h) {
        cache[local_y][local_x] = matrix[y * w + x];
    } else {
        cache[local_y][local_x] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int transposed_x = block_y * 16 + local_x;
    unsigned int transposed_y = block_x * 16 + local_y;
    if (transposed_y < w && transposed_x < h) {
        transposed_matrix[transposed_y * h + transposed_x] = cache[local_x][local_y];
    }
}