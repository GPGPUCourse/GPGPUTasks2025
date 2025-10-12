#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c,       // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{

    const int globalX = get_global_id(0); 
    const int globalY = get_global_id(1);

    const int localX = get_local_id(0);
    const int localY = get_local_id(1);


    __local float localTileA[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float localTileB[GROUP_SIZE_Y][GROUP_SIZE_X];

    float acc = 0.0f;


    const int numTiles = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (int t = 0; t < numTiles; ++t) {

        const int tiledX = t * GROUP_SIZE_X + localX;
        const int tiledY = t * GROUP_SIZE_Y + localY;


        if (tiledX < k && globalY < h) {
            localTileA[localY][localX] = a[globalY * k + tiledX];
        } else {
            localTileA[localY][localX] = 0.0f;
        }

        if (globalX < w && tiledY < k) {
            localTileB[localY][localX] = b[tiledY * w + globalX];
        } else {
            localTileB[localY][localX] = 0.0f;
        }


        barrier(CLK_LOCAL_MEM_FENCE);


        for (int i = 0; i < GROUP_SIZE_X; ++i) {
            acc += localTileA[localY][i] * localTileB[i][localX];
        }


        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalX < w && globalY < h) {
        c[globalY * w + globalX] = acc;
    }
}