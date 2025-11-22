#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(int num_rows, int num_columns,
    __global unsigned int *offsets,
    __global unsigned int *columns,
    __global unsigned int *values,
    __global unsigned int *vector,
    __global unsigned int *answer,
    int nnz
    )
{
    int index = get_local_id(0);
    int id = get_group_id(0);
    int glob_index = get_global_id(0);

    if (glob_index >= nnz) return;

    __local unsigned int cache[GROUP_SIZE];
    int counts = 0;

    int left = 0;
    int right = num_rows;


    while (right - left > 1) {

        int mid = (right + left) / 2;

        int mid_value = offsets[mid];

        if (mid_value <= glob_index) {
                left = mid;
        }
        else right = mid;
    }

    int num_str = left;

    atomic_add(&answer[num_str],
        values[glob_index] * vector[columns[glob_index]]);


}
