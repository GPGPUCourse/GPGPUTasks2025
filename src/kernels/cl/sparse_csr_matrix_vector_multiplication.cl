#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void sparse_csr_matrix_vector_multiplication(
    __global const unsigned int* csr_values, // nnz
    __global const unsigned int* csr_columns, // nnz
    __global const unsigned int* csr_row_offset, // rows
             const unsigned int nrows,
             const unsigned int nnz, 
    __global const unsigned int* v, // rows
    __global       unsigned int* output
)
{
    unsigned int i = get_global_id(0);
    unsigned int col = csr_columns[i];
    
    // unsigned int val = csr_values[i];

    if (i < nnz) {
        int l = -1;
        int r = nrows; 
        while (l < r - 1) {
            int m = (l + r) / 2;
            if (i >= csr_row_offset[m])
                l = m;
            else
                r = m;
        }

        unsigned int row = l;

        // printf("For %u, found row = %d\n", csr_values[i], row);

        atomic_add(&output[row], csr_values[i] * v[col]);
    }
}
