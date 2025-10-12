#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float* matrix, // w x h
    __global float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    __local float localData[GROUP_SIZE_Y][GROUP_SIZE_X];
    
    unsigned int localX = get_local_id(0);
    unsigned int localY = get_local_id(1);

    unsigned int globalX = get_global_id(0);
    unsigned int globalY = get_global_id(1);

   if (globalX < w && globalY < h) {
        localData[localY][localX] = matrix[globalY * w + globalX];
    }
    else {
        // printf("LOLLOL x: %d , y: %d \n",globalX,globalY );
        localData[localY][localX] = 0;

    }

    barrier(CLK_LOCAL_MEM_FENCE);

  printf("ZALUPA %d \n", get_group_id(0));
    unsigned int transGlobalX = get_group_id(1) * GROUP_SIZE_Y + localY;
    unsigned int transGlobalY = get_group_id(0) * GROUP_SIZE_X + localX;


    if (transGlobalX < w && transGlobalY < h) {
        transposed_matrix[transGlobalX * h + transGlobalY] = localData[localX][localY];
    }

   


}
