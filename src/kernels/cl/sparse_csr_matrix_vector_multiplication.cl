#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    unsigned n,
    __global const unsigned* row_index,
    __global const unsigned* col_index,
    __global const unsigned* coefs,
    __global const unsigned* vec,
    __global unsigned* res)
{
   unsigned i = get_global_id(0);
   if(i >= n)
       return;
   unsigned sum = 0;
   for(unsigned j = row_index[i]; j < row_index[i + 1]; j++)
        sum += coefs[j] * vec[col_index[j]];
   res[i] = sum;
}
