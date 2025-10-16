#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#ifndef GROUP_SIZE_X
  #define GROUP_SIZE_X 16
#endif
#ifndef GROUP_SIZE_Y
  #define GROUP_SIZE_Y 16
#endif

#ifndef TILE_X
  #define TILE_X GROUP_SIZE_X
#endif
#ifndef TILE_Y
  #define TILE_Y GROUP_SIZE_Y
#endif
#ifndef TILE_K
  #define TILE_K 16
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    // TODO
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    const uint gx = get_group_id(0);
    const uint gy = get_group_id(1);

    const uint x = gx * TILE_X + lx;
    const uint y = gy * TILE_Y + ly;

    __local float Asub[TILE_Y][TILE_K];
    __local float Bsub[TILE_K][TILE_X + 1];

    float acc = 0.0f;

    for (uint k0 = 0; k0 < k; k0 += TILE_K) {
        for (uint t = lx; t < TILE_K; t += TILE_X) {
            const uint kk = k0 + t;
            if (y < h && kk < k) {
                Asub[ly][t] = a[y * k + kk];
            } else {
                Asub[ly][t] = 0.0f;
            }
        }

        for (uint t = ly; t < TILE_K; t += TILE_Y) {
            const uint kk = k0 + t;
            if (kk < k && x < w) {
                Bsub[t][lx] = b[kk * w + x];
            } else {
                Bsub[t][lx] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint t = 0; t < TILE_K; ++t) {
            acc += Asub[ly][t] * Bsub[t][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = acc;
    }
}
