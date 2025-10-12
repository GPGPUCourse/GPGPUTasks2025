#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

// #include "../defines.h"

#define GROUP_SIZE_X 16 // в перемножении матриц я хочу группу 32х8, а тут квадратную
#define GROUP_SIZE_Y 16

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    unsigned int row = get_global_id(1);
    unsigned int col = get_global_id(0);
    unsigned int local_row = get_local_id(1);
    unsigned int local_col = get_local_id(0);

    __local float buffer[GROUP_SIZE_X * GROUP_SIZE_Y];

    if (row < h && col < w)
        buffer[local_row * GROUP_SIZE_X + local_col] = matrix[row * w + col];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int row_offset = col - local_col; 
    unsigned int col_offset = row - local_row;
    unsigned int res_row = row_offset + local_row;
    unsigned int res_col = col_offset + local_col;
    if (res_row < w && res_col < h)
        transposed_matrix[res_row * h + res_col] = buffer[local_col * GROUP_SIZE_Y + local_row];

    // почему в два раза медленней чем должно быть...
    // даже если убрать банк-конфликты (втупую, сделав код некорректным) - не ускорится
}
