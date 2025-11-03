#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* vec,
    __global const uint* val,
    __global const uint* col,
    __global const uint* offset,
    __global       uint* output
)
{
    const uint glob_id = get_global_id(0);
    const uint loc_id = get_local_id(0);
    const uint work_group_id = get_group_id(0);
    const uint group_start = offset[work_group_id];
    const uint group_end = offset[work_group_id + 1];
    __local int loc[32];
    
    loc[loc_id] = 0;
    for (uint i = group_start; i < group_end; i += 32) {
        if (loc_id + i < group_end) 
            loc[loc_id] += vec[col[loc_id + i]] * val[loc_id + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 32 >> 1; i > 0; i >>= 1) {
        if (loc_id < i)
            loc[loc_id] += loc[loc_id + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (loc_id == 0)
        output[work_group_id] = loc[0];
}
