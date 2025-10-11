#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define GROUP_WIDTH 32
#define GROUP_HEIGHT 8

__attribute__((reqd_work_group_size(GROUP_WIDTH, GROUP_HEIGHT, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float mem[GROUP_WIDTH * GROUP_HEIGHT];

    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int li = get_local_id(0);
    unsigned int lj = get_local_id(1);

    unsigned int mem_index = lj * GROUP_WIDTH + ((li + lj) % GROUP_WIDTH);
    if (i >= w || j >= h) mem[mem_index] = 0.0f;
    else mem[mem_index] = matrix[j * w + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (li < GROUP_HEIGHT) {
        unsigned int write_index = (li * GROUP_WIDTH) + ((lj + li) % GROUP_WIDTH);
        float num = mem[write_index];

        unsigned int original_i = (i - li + lj);
        unsigned int original_j = (j - lj + li);

        if (original_i < w && original_j < h)
            transposed_matrix[original_i * h + original_j] = num;
    }
}
