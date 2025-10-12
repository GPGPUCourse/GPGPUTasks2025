#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{

    unsigned int index = get_global_id(0);

    unsigned int curX = index%w;
    unsigned int curY = index/w;
    float acc  = 0.0f;

    for(unsigned int i =0 ; i<k; ++i) {
        acc +=  a[curY*k + i]* b[i*w + curX];
    }
     c[index] = acc;


}
