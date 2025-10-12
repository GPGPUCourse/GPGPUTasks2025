#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float *a, // rows=h x cols=k
    __global const float *b, // rows=k x cols=w
    __global float *c,       // rows=h x cols=w
    unsigned int h,
    unsigned int w,
    unsigned int k)
{
    const uint S = GROUP_SIZE_X;
    // const uint H = GROUP_SIZE_X;
    // const uint W = GROUP_SIZE_Y;
    // assert(H == W);

    const uint local_height = get_local_id(0);
    const uint local_width = get_local_id(1);
    const uint local_index = local_height * S + local_width;
    const uint height = get_global_id(0);
    const uint width = get_global_id(1);

    __local float a_local_data[S * S];
    __local float b_local_data[S * S];
    float value = 0;
    for (uint block = 0; block * S < k; ++block)
    {
        uint load_height = height;
        uint load_width = S * block + local_width;
        if (load_height < h && load_width < k)
        {
            a_local_data[local_index] = a[load_height * k + load_width];
        }
        else
        {
            a_local_data[local_index] = 0;
        }

        load_height = S * block + local_height;
        load_width = width;
        if (load_height < k && load_width < w)
        {
            b_local_data[local_index] = b[load_height * w + load_width];
        }
        else
        {
            b_local_data[local_index] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (height < h && width < w)
        {
            // value = c[height][width] = c_local_data[local_height][local_width]
            for (uint i = 0; i < S; ++i)
            {
                value += a_local_data[local_height * S + i] * b_local_data[i * S + local_width];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (height < h && width < w)
    {
        c[height * w + width] = value;
    }
}
