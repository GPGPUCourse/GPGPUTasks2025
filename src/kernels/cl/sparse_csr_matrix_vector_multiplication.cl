#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(__global const uint* row_csr,
                                                      __global const uint* col,
                                                      __global const uint* val,
                                                      __global const uint* vector,
                                                      __global uint* res,
                                                      int nrows)
{
    const unsigned int idx = get_global_id(0);
    if (idx >= nrows) return;
    int start = row_csr[idx], end = row_csr[idx+1];
    uint ans = 0;
    for (int i = start; i < end; i++){
        ans += val[i] * vector[col[i]];
    }
    res[idx] = ans;
}
