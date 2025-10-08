#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    // TODO
    uint grx = get_group_id(0);
    uint gry = get_group_id(1);

    uint i = get_local_id(0);
    uint j = get_local_id(1);

    uint mask = 0xf;
    __local float a_piece[256], b_piece[256];
    __local float res_piece[256];
    res_piece[16 * j + i] = 0;
    // barrier(CLK_LOCAL_MEM_FENCE);

    for (uint l = 0; l < (k + 15) >> 4; l++) {
        // if (grx * 16 + i >= w || gry * 16 + j >= h || l * 16)
        if ((gry * 16 + j) >= h || (l * 16 + i) >= k)
            a_piece[16 * j + (i + j & mask)] = 0;
        else
            a_piece[16 * j + (i + j & mask)] = a[(gry * 16 + j) * k + l * 16 + i];

        if ((l * 16 + j) >= k || (grx * 16 + i) >= w)
            b_piece[16 * j + (i + j & mask)] = 0;
        else
            b_piece[16 * j + (i + j & mask)] = b[(l * 16 + j) * w + grx * 16 + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        float mid_res = 0;

        for (uint m = 0; m < 16; m++) {
            // barrier(CLK_LOCAL_MEM_FENCE);
            // fma (a_piece[16 * j + (m + j & mask)], b_piece[16 * m + (i + m & mask)],res_piece[16*j + i]); // bank conflict is better than sync extra time
            uint idxa = 16 * j + ((m + i & mask) + j & mask);
            uint idxb = 16 * (m + i & mask) + (i + (m + i & mask) & mask);

            mid_res += a_piece[idxa] * b_piece[idxb]; // this version has bank conflict 2
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        res_piece[16 * j + i] += mid_res;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (gry * 16 + j < h && grx * 16 + i < w)
        c[(gry * 16 + j) * w + (grx * 16 + i)] = res_piece[j * 16 + i];
}
