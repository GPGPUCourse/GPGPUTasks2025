#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define BLOCK_DIM GROUP_SIZE_X // GROUP_SIZE_X has to be equal to GROUP_SIZE_Y
#define LOCAL_WIDTH (BLOCK_DIM + 1)
#define LOCAL_HEIGHT BLOCK_DIM

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    __local float a_chunk[LOCAL_WIDTH * LOCAL_HEIGHT];
    __local float b_chunk[LOCAL_WIDTH * LOCAL_HEIGHT];
    const unsigned int chunk_address = local_y * LOCAL_WIDTH + local_x;

    float sum = 0;
    for (unsigned int block = 0; block * BLOCK_DIM < k; block++) {
        unsigned int z = block * BLOCK_DIM + local_x;
        a_chunk[chunk_address] = z < k ? a[k * y + z] : 0;
        z = block * BLOCK_DIM + local_y;
        b_chunk[chunk_address] = z < k ? b[w * z + x] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (z = 0; z < BLOCK_DIM; z++){
            //if (x == 0 && y == 0 && block == 0) {
            //    printf("%f %f", a_chunk[local_y * LOCAL_WIDTH + z], b_chunk[z * LOCAL_WIDTH + local_x]);
            //}
            sum += a_chunk[local_y * LOCAL_WIDTH + z] * b_chunk[z * LOCAL_WIDTH + local_x];
        }
        //if (x == 0 && y == 0) {
        //    printf("\n");
        //}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[y * w + x] = sum;
}
