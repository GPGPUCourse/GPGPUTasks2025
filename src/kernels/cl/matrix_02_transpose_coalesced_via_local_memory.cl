#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define WG_W (16)
#define WG_H (16)

__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float local_data[GROUP_SIZE];
    uint global_i = get_global_id(1);
    uint global_j = get_global_id(0);

    const uint local_i = get_local_id(1);
    const uint local_j = get_local_id(0);

    if (global_i >= h || global_j >= w) {
        return;
    }

    local_data[local_i * WG_W + local_j] = matrix[global_i * w + global_j];
    barrier(CLK_LOCAL_MEM_FENCE);
    transposed_matrix[global_j * h + global_i] = local_data[local_i * WG_W + local_j];
}
