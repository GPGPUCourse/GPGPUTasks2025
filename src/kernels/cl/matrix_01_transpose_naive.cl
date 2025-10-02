#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_01_transpose_naive(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if ((j * w + i) < w * h) {
        transposed_matrix[j + i * h] = matrix[j * w + i];
        // вопрос смешнявка для студентов, почему нельзя так???)))
        // transposed_matrix[j * w + i] = matrix[j + i * h];
        // казалось бы просто теперь мы пишем coalesced, а не читаем
    }

}
