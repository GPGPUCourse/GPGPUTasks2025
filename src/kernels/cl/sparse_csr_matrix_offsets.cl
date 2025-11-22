#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void sparse_csr_matrix_offsets(int num_rows, int num_columns, __global const unsigned int *A,
    __global unsigned int *offsets
    ) // TODO input/output buffers
{
    int index = get_local_id(0);
    int id = get_group_id(0);

    __local unsigned int cache[GROUP_SIZE];
    unsigned int counts = 0;

    for (int i = 0; i < num_columns; i+=GROUP_SIZE) {

//        if (index + i >= num_columns) {
//            continue;
//        }

        if (index + i < num_columns && A[id * num_columns + index + i] != 0)
            counts++;
    }

    cache[index] = counts;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int size = GROUP_SIZE; size > 1; size /= 2) {
        if (2 * index < size){
            cache[index] = cache[index] + cache[index + size / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index == 0) {
        offsets[id + 1] = cache[0];
    } // затем в main надо было сделать prefix

}
