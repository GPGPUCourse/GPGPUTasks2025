#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

float multiplyLocal(__local const float* a, __local const float* b, 
                    const uint x, const uint y) {
    float sum = 0;
    for (int i = 0; i < TILE_SIZE; ++i) {
        sum += a[y * TILE_SIZE + i] * b[i * TILE_SIZE + x];
    }
    return sum;
}

__attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float localA[TILE_SIZE * TILE_SIZE];
    __local float localB[TILE_SIZE * TILE_SIZE];

    const uint localX = get_local_id(0);
    const uint localY = get_local_id(1);

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    uint localIdx = localY * TILE_SIZE + localX;
    const uint loadCnt = (k + TILE_SIZE - 1) / TILE_SIZE;

    float sum = 0;
    for (int i = 0; i < loadCnt; ++i) {
        uint loadX = localX + TILE_SIZE * i;
        uint loadY = localY + TILE_SIZE * i;

        localA[localIdx] = (loadX < k && y < h ? a[y * k + loadX] : 0);
        localB[localIdx] = (loadY < k && x < w ? b[loadY * w + x] : 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        sum += multiplyLocal(localA, localB, localX, localY);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (y < h && x < w) {
        c[y * w + x] = sum;
    }
}
