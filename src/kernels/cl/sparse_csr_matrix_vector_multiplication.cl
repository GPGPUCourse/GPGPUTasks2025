#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"



__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector_values,
    const uint cols, const uint rows, const uint n_values
) 
{
   const uint row_id = get_group_id(0); 
   const uint lid = get_local_id(0);
   if (row_id >= rows) return;

   const uint row_start = csr_row_offsets[row_id];
   const uint row_end = (row_id == rows - 1) ? n_values : csr_row_offsets[row_id+1];
   __local volatile uint acc;
   if (lid == 0) acc = 0;
   barrier(CLK_LOCAL_MEM_FENCE);

   for (uint i = row_start + lid; i < row_end; i += GROUP_SIZE) {
        const uint mv =  csr_values[i];
        const uint vv =  vector_values[csr_columns[i]];
        atomic_add(&acc, mv * vv);
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   if (lid == 0) output_vector_values[row_id] = acc;
}
