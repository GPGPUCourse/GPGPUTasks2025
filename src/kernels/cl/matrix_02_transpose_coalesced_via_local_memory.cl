#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define TILE_SIZE 16

__attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w * h
                       __global       float* transposed_matrix, // h * w
                                unsigned int w,
                                unsigned int h)
{
    __local float tile[TILE_SIZE][TILE_SIZE + 1]; 
    
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);
    
    const unsigned int read_x = group_x * TILE_SIZE + local_x; // read global pos
    const unsigned int read_y = group_y * TILE_SIZE + local_y;
    
    const unsigned int write_x = group_y * TILE_SIZE + local_x; // write global pos
    const unsigned int write_y = group_x * TILE_SIZE + local_y;
    
    if (read_x < w && read_y < h) {

        tile[local_y][local_x] = matrix[read_y * w + read_x];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (write_x < h && write_y < w) {

        transposed_matrix[write_y * h + write_x] = tile[local_x][local_y];
    }
}
