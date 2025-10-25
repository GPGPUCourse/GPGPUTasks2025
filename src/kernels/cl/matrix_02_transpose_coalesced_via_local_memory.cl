#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float local_buffer[256];
    size_t ig = get_global_id(0);
    size_t jg = get_global_id(1);
    size_t idx_inp = jg*w + ig;
    if (ig<w && jg<h) {
        size_t il = get_local_id(0);
        size_t jl = get_local_id(1);
        size_t t = (il+jl)%GROUP_SIZE_X;
        local_buffer[jl*GROUP_SIZE_X + t] = matrix[idx_inp];


        barrier(CLK_LOCAL_MEM_FENCE);


        size_t idx_out = (get_group_id(0)*GROUP_SIZE_X + jl)*h + get_group_id(1)*GROUP_SIZE_Y + il;
        transposed_matrix[idx_out] = local_buffer[t + il*(GROUP_SIZE_X)];
        //Эту индексацию нельзя понять аналитически, можно только покрутить прямоугольнички на листочке
    }
}
