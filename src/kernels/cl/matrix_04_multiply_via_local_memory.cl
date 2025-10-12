#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    // static_assert(GROUP_SIZE_X==GROUP_SIZE_Y, "423427364274230467"); // FOR SAFE

    __local float localTileA[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float localTileB[GROUP_SIZE_Y][GROUP_SIZE_X];

    unsigned int globalX = get_global_id(0);
    unsigned int globalY = get_global_id(0);

    unsigned int localX = get_local_id(0);
    unsigned int localY = get_local_id(0);

    unsigned int titleCount = k/GROUP_SIZE_X + (k%GROUP_SIZE_X==0?0:1) ;

    unsigned int slide = 0;

    for(unsigned int titleN =0 ;titleN <titleCount; titleN++ ){        

    if (localX+slide <k && globalY < h ) {
        localTileA[localX][localY] = a[localX+slide + globalY*k ];
    } else {
        localTileA[localX][localY] = 0;
    }

    if (localX <w && globalY +slide < k ) {
        localTileB[localY][localX] = b[localY + (globalX+slide)*w ];// ??????? 
    } else {
        localTileB[localY][localX] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float acc  = 0.0f;

    for(unsigned int i =0 ; i<GROUP_SIZE_X; ++i) {
        acc +=  localTileA[ i][localY]* localTileB[localX][i];
    }

    c[globalX + globalY*w ] += acc;

    

    slide=+GROUP_SIZE_X;

    }
    
}
