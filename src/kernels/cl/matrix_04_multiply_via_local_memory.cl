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
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const uint local_index_i = get_local_id(0);
    const uint local_index_j = get_local_id(1);
    const uint local_index = local_index_j * GROUP_SIZE_X + local_index_i;

    __local float local_data_a[GROUP_SIZE];
    __local float local_data_b[GROUP_SIZE];
    c[j * w + i] = 0;

    for (int t = 0; t < k; t += GROUP_SIZE_X) {
        if (i >= w || j >= h) {
            local_data_a[local_index] = 0;
            local_data_b[local_index] = 0;
        }
        else {
            const uint a_index = j * k + t + local_index_i;
            const uint b_index = (t + local_index_j) * w + i;
            local_data_a[local_index] = a[a_index];
            local_data_b[local_index] = b[b_index];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i < w && j < h) {
            for (int local_index_t = 0; local_index_t < GROUP_SIZE_X; ++local_index_t) {
                const uint a_local_index = local_index_j * GROUP_SIZE_X + local_index_t;
                const uint b_local_index = local_index_t * GROUP_SIZE_X + local_index_i;
                c[j * w + i] += local_data_a[a_local_index] * local_data_b[b_local_index];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
