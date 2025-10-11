#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define GROUP_SIZE 16

__attribute__((reqd_work_group_size(GROUP_SIZE, GROUP_SIZE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    unsigned fragment_index = ly * GROUP_SIZE + lx;

    __local float a_fragment[GROUP_SIZE * GROUP_SIZE];
    __local float b_fragment[GROUP_SIZE * GROUP_SIZE];
    __local float c_fragment[GROUP_SIZE * GROUP_SIZE];
    c_fragment[fragment_index] = 0;

    unsigned int k_limit = (k + GROUP_SIZE - 1) / GROUP_SIZE; // Round up

    for (int i = 0; i < k_limit; i++) {
        unsigned int offset = i * GROUP_SIZE;

        if ((offset+lx) < k && y < h)
            a_fragment[fragment_index] = a[y * k + offset + lx];
        else
            a_fragment[fragment_index] = 0;

        if ((offset + ly) < k && x < w)
            b_fragment[fragment_index] = b[(offset + ly) * w + x];
        else
            b_fragment[fragment_index] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < GROUP_SIZE; j++) {
            c_fragment[fragment_index] += a_fragment[ly * GROUP_SIZE + j] * b_fragment[j * GROUP_SIZE + lx];
        }
    }

    c[y * w + x] = c_fragment[fragment_index];
}
