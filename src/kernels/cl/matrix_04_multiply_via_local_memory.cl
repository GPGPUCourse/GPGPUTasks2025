#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    const uint local_i = get_local_id(0);
    const uint local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int tileK = 0; tileK * TILE_SIZE < k; ++tileK) {

        tileA[local_j][local_i] = a[j * k + tileK * TILE_SIZE + local_i];
        tileB[local_j][local_i] = b[(tileK * TILE_SIZE + local_j) * w + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int z = 0; z < TILE_SIZE; ++z) {
            sum += tileA[local_j][z] * tileB[z][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * w + i] = sum;
}
