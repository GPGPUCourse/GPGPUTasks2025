#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{

    // я пыталась сделать так чтобы группа могла быть разных размеров, но здесь все еще нужно обработать ситуацию, когда
    // warp'ы выходят за границы workSpace

    unsigned int col = get_global_id(0);
    unsigned int row = get_global_id(1);
    unsigned int loc_col = get_local_id(0);
    unsigned int loc_row = get_local_id(1);
    unsigned int size_grp_by_col = get_local_size(0);
    unsigned int size_grp_by_row = get_local_size(1);
    __local float local_data[GROUP_SIZE];

    if (col < w && row < h) {
        local_data[loc_row * size_grp_by_col + ((loc_col + loc_row) % size_grp_by_col)] = matrix[row * w + col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int start_group_from = get_group_id(0) * size_grp_by_col * h + get_group_id(1) * size_grp_by_row;
    unsigned int loc_idx = loc_row * size_grp_by_col + loc_col;
    unsigned int new_col = loc_idx % size_grp_by_row;
    unsigned int new_row = loc_idx / size_grp_by_row;

    if (start_group_from + new_row * h + new_col < w * h) {
        transposed_matrix[start_group_from + new_row * h + new_col] = local_data[loc_col * size_grp_by_col + ((loc_col + loc_row) % size_grp_by_col)];
    }
}
