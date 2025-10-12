#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float local_data_a[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    __local float local_data_b[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint i = get_local_id(0);
    uint j = get_local_id(1);

    float result = 0;
    for (uint t = 0; t < k; t += 16) {
        local_data_a[j][i] = a[y * k + t + i];
        local_data_b[j][i] = b[(t + j) * w + x];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint r = 0; r < 16; ++r) {
            result += local_data_a[j][r] * local_data_b[r][i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[y * w + x] = result;
}
