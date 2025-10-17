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
    __local float chunk[GROUP_SIZE_X][GROUP_SIZE_Y];
    unsigned lx = get_local_id(0), ly = get_local_id(1);
    unsigned cx = get_group_id(0), cy = get_group_id(1);
    unsigned gx = cx * GROUP_SIZE_X + lx, gy = cy * GROUP_SIZE_Y + ly;
    unsigned tx = cy * GROUP_SIZE_X + lx, ty = cx * GROUP_SIZE_Y + ly;
    if(gx < w && gy < h)
        chunk[lx][ly] = matrix[gy * w + gx];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tx < h && ty < w)
        transposed_matrix[ty * h + tx] = chunk[ly][lx];
}
