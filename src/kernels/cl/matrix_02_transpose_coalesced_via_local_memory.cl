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
    const unsigned int gx = get_group_id(0);
    const unsigned int gy = get_group_id(1);
    const unsigned int lx =  get_local_id(0);
    const unsigned int ly =  get_local_id(1);

    __local float tile[GROUP_SIZE_X][GROUP_SIZE_Y];

    unsigned int x =  gx * GROUP_SIZE_X + lx;
    unsigned int y =  gy * GROUP_SIZE_Y + ly;

    if(w > x && h > y) {
        tile[ly][lx] = matrix[w * y + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int tx = gy * GROUP_SIZE_Y + lx;
    unsigned int ty = gx * GROUP_SIZE_X + ly;
    
    if(x < h && y < w) {
        transposed_matrix[ty * h + tx] = tile[lx][ly];
    }
}
