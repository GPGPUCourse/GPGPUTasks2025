#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    // csr representation
    __global const unsigned int* offsets,
    __global const unsigned int* cols,
    __global const unsigned int* values,
    //
    __global const unsigned int* vector,
    __global unsigned int* out_vector,
    unsigned int n // rows
)
{
    const unsigned int gid = get_group_id(0);
    if (gid > n) {
        return;
    }

    const unsigned int lid = get_local_id(0);
    const unsigned int lsize = get_local_size(0);

    __local uint l_sum[GROUP_SIZE];
    unsigned int cur_sum = 0;

    if (gid == 0) {
        l_sum[lid] = 0;
    }

    const unsigned int start = offsets[gid] + lid;
    const unsigned int end = offsets[gid + 1];

    #pragma unroll
    for (unsigned int i = start; i < end; i += lsize) {
        cur_sum += values[i] * vector[cols[i]];
    }

    l_sum[lid] = cur_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (unsigned int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            l_sum[lid] += l_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out_vector[gid] = l_sum[0];
    }
}
