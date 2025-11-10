#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* rows_offset,
    __global const uint* cols,
    __global const uint* values,
    __global const uint* vector_b,
    __global       uint* output,
    unsigned       int n
)
{
  const unsigned int index = get_global_id(0);
  const unsigned int local_index = get_local_id(0);


  if (index < n) {
    const unsigned int begin = rows_offset[index];
    const unsigned int end = rows_offset[index + 1];


    unsigned int res = 0;
    for (unsigned int i = begin; i < end; ++i) {
        res += values[i] * vector_b[cols[i]];
    }

    output[index] = res;
  }
}
