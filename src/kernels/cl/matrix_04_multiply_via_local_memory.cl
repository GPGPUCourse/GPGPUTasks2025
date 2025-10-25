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
    // TODO
    __local float local_B_tile[GROUP_SIZE_X*GROUP_SIZE_Y];
    __local float local_A_tile[GROUP_SIZE_X*GROUP_SIZE_Y];
    float sum = 0;

    size_t total_local_coord = get_local_id(0) + get_local_id(1)*GROUP_SIZE_X;

    for(size_t ki = 0; ki<k; ki+=GROUP_SIZE_X) {
        local_A_tile[total_local_coord] = a[k*get_global_id(1) + (ki + get_local_id(0))];
        local_B_tile[total_local_coord] = b[w*(ki + get_local_id(1)) + get_global_id(0)];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(size_t t = 0; t<GROUP_SIZE_X; ++t) {
            sum+=
                local_A_tile[t + get_local_id(1)*GROUP_SIZE_X] *
                local_B_tile[get_local_id(0) + t*GROUP_SIZE_X];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[get_global_id(0) + get_global_id(1)*w] = sum;
}
