#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h * cols=k
                       __global const float* b, // rows=k * cols=w
                       __global       float* c, // rows=h * cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const unsigned int x = get_global_id(0); // column idx in c_mat
    const unsigned int y = get_global_id(1); // row idx in c_mat
    
    if (x < w && y < h) {

        float sum = 0.0f;
        
        for (unsigned int i = 0; i < k; ++i) {
		      
            float a_val = a[y * k + i];        // a_mat y row elem
            float b_val = b[i * w + x];        // b_mat x col elem
	    sum += a_val * b_val;
        }
        
        c[y * w + x] = sum;
    }
}
