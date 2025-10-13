#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float A[16][16];
    __local float B[16][16];

    uint local_i = get_local_id(1);
    uint local_j = get_local_id(0);
    uint i = get_global_id(1);
    uint j = get_global_id(0);

    float sum = 0;

    uint block_size = k / 16 + (k % 16 > 0);
    for(uint m = 0; m < block_size; ++m) {
        uint li = m * 16 + local_i;
        uint lj = m * 16 + local_j;
        if(i < h && lj < k) {
            A[local_i][local_j] = a[i * k + lj];
        } else {
            A[local_i][local_j] = 0;
        }

        if(li < k && j < w){
            B[local_i][local_j] = b[li * w + j];
        } else {
            B[local_i][local_j] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(uint mm=0; mm<16; ++mm) {
            sum += A[local_i][mm] * B[mm][local_j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < h && j < w) {
        c[i * w + j] = sum;
    }
}
