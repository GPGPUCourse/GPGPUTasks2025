#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{

    // Допустим, что группа у нас одинаковая в ширину и высоту. Иначе это каторга с индексами, которая в целом
    // не относится к задаче как таковая, и которую я уже пережила при реализации транспортирования :)

    unsigned int col = get_global_id(0);
    unsigned int row = get_global_id(1);
    unsigned int loc_col = get_local_id(0);
    unsigned int loc_row = get_local_id(1);
    unsigned int size_grp_by_col = get_local_size(0);
    unsigned int size_grp_by_row = get_local_size(1);

    __local float local_data_a[GROUP_SIZE];
    __local float local_data_b[GROUP_SIZE];
    unsigned int num_blocks = (k + size_grp_by_col - 1) / size_grp_by_col;

    float answer = 0.0f;
    for (unsigned int id_block = 0; id_block < num_blocks; ++id_block) {

        float elem_A = 0.0f;
        unsigned int cur_col = id_block * size_grp_by_col + loc_col;
        if (cur_col < k && row < h) {
            elem_A = a[row * k + cur_col];
        }
        local_data_a[loc_row * size_grp_by_col + loc_col] = elem_A;

        float elem_B = 0.0f;
        unsigned int cur_row = id_block * size_grp_by_col + loc_row;
        if (cur_row < k && col < w) {
            elem_B = b[cur_row * w + col];
        }
        local_data_b[loc_row * size_grp_by_col + loc_col] = elem_B;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < size_grp_by_col; ++i) {
            answer += local_data_a[loc_row * size_grp_by_col + i] * local_data_b[i * size_grp_by_col + loc_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < h && col < w)
        c[row * w + col] = answer;

}
