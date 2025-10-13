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
    __local float mt[16 * 17];
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    uint li = get_local_id(0);
    uint lj = get_local_id(1);

    if (i < w && j < h) {
        mt[li * 17 + lj] = matrix[j * w + i];
    } else {
        mt[li * 17 + lj] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint group_i = get_group_id(0);
    uint group_j = get_group_id(1);
    uint new_i = group_j * 16 + li;
    uint new_j = group_i * 16 + lj;
    
    if (new_i < h && new_j < w) {
        transposed_matrix[new_j * h + new_i] = mt[lj * 17 + li];
    }
}
