#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // TODO
    __local float buffer[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    
    if (x < w && y < h) {
        buffer[ly][lx] = matrix[y * w + x];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint tx = get_group_id(1) * GROUP_SIZE_Y + lx;
    const uint ty = get_group_id(0) * GROUP_SIZE_X + ly;
    
    if (tx < h && ty < w) {
        transposed_matrix[ty * h + tx] = buffer[lx][ly];
    }
}
