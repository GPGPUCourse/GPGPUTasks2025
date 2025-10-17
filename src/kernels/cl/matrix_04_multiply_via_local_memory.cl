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
    __local float locA[GROUP_SIZE];
    __local float locB[GROUP_SIZE];
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);

    // fill with zeros
    const size_t loc_i = get_local_id(0);
    const size_t loc_j = get_local_id(1);
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0;
    // block begin (i - loc_i, j - loc_j)
    for (size_t l = 0; l < k; l += GROUP_SIZE_X) {
        // A block begin (l, j - loc_j)
        // B block begin (i - loc_i, l)
        locA[loc_j * GROUP_SIZE_X + loc_i] = a[(j - loc_j + loc_j) * k + (l + loc_i)];
        // read transposed
        locB[loc_i * GROUP_SIZE_X + loc_j] = b[(l + loc_j) * w + (i - loc_i + loc_i)];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t x = 0; x < GROUP_SIZE_X; ++x) {
            sum += locA[loc_j * GROUP_SIZE_X + x] * locB[loc_i * GROUP_SIZE_X + x];
        }
        // sum because of aggregation in for
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * w + i] = sum;
}
