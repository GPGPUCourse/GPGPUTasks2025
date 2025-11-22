#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void sparse_csr_matrix_decomp(int num_rows, int num_columns, __global const unsigned int *A,
    __global unsigned int *offsets,
    __global unsigned int *columns,
    __global unsigned int *values
    ) // TODO input/output buffers
{
    int index = get_local_id(0);
    int id = get_group_id(0);

    __local unsigned int cache[GROUP_SIZE];

    int last_sum = 0;

    for (int i = 0; i < num_columns; i += GROUP_SIZE) {

        if (i != 0) last_sum += cache[GROUP_SIZE - 1]; // то сколько элементов в прошлой префикс сумме

        barrier(CLK_LOCAL_MEM_FENCE);

        if (index + i < num_columns && A[id * num_columns + index + i] != 0)
            cache[index] = 1;

        else {
            cache[index] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

         int step = 2;
         while (step / 2 < 2 * GROUP_SIZE){
             barrier(CLK_LOCAL_MEM_FENCE);
             if(index < 2 * GROUP_SIZE / step){
                 int idx = (index + 1) * step - 1;
                 cache[idx] = cache[idx] + cache[idx - step / 2];
             }
             step *= 2;
         }
         step /= (2 * 2);

         while (step > 1){
             barrier(CLK_LOCAL_MEM_FENCE);
             if(index < 2 * GROUP_SIZE / step - 1) {
                 int idx = (index + 1) * step + step / 2 - 1;
                 cache[idx] = cache[idx] + cache[idx - step / 2];
             }
             step /= 2;
         }

         barrier(CLK_LOCAL_MEM_FENCE);

         if (index != 0) {
             if (cache[index] - cache[index - 1] == 1)
                 values[offsets[id] + last_sum + cache[index] - 1] = A[id * num_columns + index + i];
                 columns[offsets[id] + last_sum + cache[index] - 1] = i + index;
         }
         else {
             if (cache[index] == 1) {
                 values[offsets[id] + last_sum + cache[index] - 1] = A[id * num_columns + index + i];
                 columns[offsets[id] + last_sum + cache[index] - 1] = i + index;
             }
         }
         barrier(CLK_LOCAL_MEM_FENCE);

    }

}
