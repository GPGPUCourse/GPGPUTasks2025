#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

// require the WorkGroup to be a square
#if (GROUP_SIZE_X != GROUP_SIZE_Y)
#error GROUP_SIZE_X must be equal to GROUP_SIZE_Y
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    unsigned int global_idx_x = get_global_id(0);
    unsigned int global_idx_y = get_global_id(1);

    const unsigned int local_idx_x = get_local_id(0);
    const unsigned int local_idx_y = get_local_id(1);

    __local float a_chunk[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float b_chunk[GROUP_SIZE_X * GROUP_SIZE_Y];

    float sum_result = 0;
    for (unsigned int shift = 0; shift < k; shift += GROUP_SIZE_X) {
        // fill in local memory with chunks of a and b
        const unsigned int a_x = shift + local_idx_x;
        const unsigned int a_y = global_idx_y;
        const bool in_bounds_a = a_x < k && a_y < h;
        a_chunk[local_idx_y * GROUP_SIZE_X + local_idx_x] = in_bounds_a ? a[a_y * k + a_x] : 0.0f;

        const unsigned int b_x = global_idx_x;
        const unsigned int b_y = shift + local_idx_y;
        const bool in_bounds_b = b_x < w && b_y < k;
        b_chunk[local_idx_y * GROUP_SIZE_X + local_idx_x] = in_bounds_b ? b[b_y * w + b_x] : 0.0f;

        // wait until all chunks data is filled
        barrier(CLK_LOCAL_MEM_FENCE);

        // multiply and sum up
        for (unsigned int i = 0; i < GROUP_SIZE_X; i++) {
            sum_result += a_chunk[local_idx_y * GROUP_SIZE_X + i] * b_chunk[i * GROUP_SIZE_X + local_idx_x];
        }
        // wait until all chunks data is used
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_idx_y * w + global_idx_x] = sum_result;
}
