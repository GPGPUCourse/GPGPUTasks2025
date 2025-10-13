#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);
    float result = 0.0f;
    __local float a_local[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float b_local[GROUP_SIZE_X * GROUP_SIZE_Y];
    const uint local_write_index = local_y * GROUP_SIZE_X + local_x;

    for (uint i = 0; i * GROUP_SIZE_X < k; ++i) {
        uint a_local_x = i * GROUP_SIZE_X + local_x;
        a_local[local_write_index] = (a_local_x < k) ? a[y * k + a_local_x] : 0.0f;
        uint b_local_y = i * GROUP_SIZE_X + local_y;
        b_local[local_write_index] = (b_local_y < k) ? b[b_local_y * w + x] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = 0; j < GROUP_SIZE_X; ++j) {
            result += a_local[local_y * GROUP_SIZE_X + j] * b_local[j * GROUP_SIZE_X + local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[y * w + x] = result;
}
