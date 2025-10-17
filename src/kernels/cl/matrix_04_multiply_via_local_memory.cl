#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    unsigned x = get_global_id(0), y = get_global_id(1);
    unsigned cx = get_group_id(0), cy = get_group_id(1);
    unsigned lx = get_local_id(0), ly = get_local_id(1);
    if(x >= w || y >= h)
        return;
    const unsigned SIZE_Z = GROUP_SIZE_X;
    __local float cache_a[SIZE_Z][GROUP_SIZE_Y];
    __local float cache_b[SIZE_Z][GROUP_SIZE_X];
    float res = 0;
    for(unsigned i = 0; i < (k + SIZE_Z - 1) / SIZE_Z; i++)
    {
        size_t rem = (i + 1) * SIZE_Z > k ? k - i * SIZE_Z : SIZE_Z;
        if(lx < rem)
            cache_a[lx][ly] = a[(cy * GROUP_SIZE_Y + ly) * k + i * SIZE_Z + lx];
        if(ly < rem)
            cache_b[ly][lx] = b[(i * SIZE_Z + ly) * w + cx * GROUP_SIZE_X + lx];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(unsigned j = 0; j < rem; j++)
            res += cache_a[j][ly] * cache_b[j][lx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[y * w + x] = res;
}
