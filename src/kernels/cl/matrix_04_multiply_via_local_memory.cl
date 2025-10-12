#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(32, 32, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    
    unsigned int globalX = get_global_id(0);
    unsigned int globalY = get_global_id(1);

    unsigned int localX = get_local_id(0);
    unsigned int localY = get_local_id(1);

    unsigned int groupSizeX = get_local_size(0);
    unsigned int groupSizeY = get_local_size(1);

    unsigned int numTiles = (k + groupSizeX - 1) / groupSizeX;

    __local float tileA[32][32];
    __local float tileB[32][32];

    float acc = 0.0;

    for (unsigned int i = 0; i != numTiles; ++i) {
        unsigned int idX = i * groupSizeX + localX;
        unsigned int idY = globalY;
        if (idX >= k || idY >= h) {
            tileA[localY][localX] = 0.0;
        } else {
            tileA[localY][localX] = a[idY * k + idX];
        }

        idX = globalX;
        idY = i * groupSizeY + localY;

        if (idX >= w || idY >= k) {
            tileB[localY][localX] = 0.0;
        } else {
            tileB[localY][localX] = b[idY * w + idX];
        }

        barrier(CLK_LOCAL_MEM_FENCE);


        #pragma unroll
        for (unsigned int j = 0; j != 32; ++j) {
            acc += tileA[localY][j] * tileB[j][localX];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (globalY < h && globalX < w) {
        c[globalY * w + globalX] = acc;
    }
}
