#include "helpers/rassert.cl"
#include "../defines.h"


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* restrict input_data,
    __global       uint* restrict output_data,
    const int sorted_now_log,
    const int n)
{
    __local unsigned int local_data[GROUP_SIZE * DEFAULT_SORT_BLOCK];
    const unsigned int idx = get_global_id(0);
    const unsigned int local_idx = idx & 255u;
    const unsigned int local_base = local_idx << 2;
    unsigned int temp;

    if (sorted_now_log == 0) {
        for (unsigned int j = 0u; j < DEFAULT_SORT_BLOCK; ++j) {
            local_data[local_base + j] = (((idx << 2) + j < n) ? input_data[(idx << 2) + j] : UINT_MAX);
        }        

        if (local_data[local_base] > local_data[local_base + 1]) {
            temp = local_data[local_base];
            local_data[local_base] = local_data[local_base + 1];
            local_data[local_base + 1] = temp;
        }

        if (local_data[local_base + 2] > local_data[local_base + 3]) {
            temp = local_data[local_base + 2];
            local_data[local_base + 2] = local_data[local_base + 3];
            local_data[local_base + 3] = temp;
        }

        if (local_data[local_base + 1] > local_data[local_base + 3]) {
            temp = local_data[local_base + 1];
            local_data[local_base + 1] = local_data[local_base + 3];
            local_data[local_base + 3] = temp;
        }

        if (local_data[local_base] > local_data[local_base + 2]) {
            temp = local_data[local_base];
            local_data[local_base] = local_data[local_base + 2];
            local_data[local_base + 2] = temp;
        }

        if (local_data[local_base + 1] > local_data[local_base + 2]) {
            temp = local_data[local_base + 1];
            local_data[local_base + 1] = local_data[local_base + 2];
            local_data[local_base + 2] = temp;
        }

        for (unsigned int j = 0u; j < DEFAULT_SORT_BLOCK; ++j) {
            if ((idx << 2) + j < n) {
                output_data[(idx << 2) + j] = local_data[local_base + j];
            }
        }
    } else {
        const unsigned int val = ((idx < n) ? input_data[idx] : 0xfffffffdu);
        const int defaultL = ((idx >> sorted_now_log) ^ 1u) << sorted_now_log;
        const unsigned int searching_in_left_half = (defaultL <= idx);
        const unsigned int comp_val = (searching_in_left_half ? val + 1u : val);

        int L = defaultL - 1;
        int R = defaultL + (1 << sorted_now_log);


        while (R - L > 1) {
            const unsigned int M = L + R >> 1;
            const unsigned int comparing_to = (M < n ? input_data[M] : UINT_MAX);
            if (comparing_to >= comp_val) {
                R = M;
            } else {
                L = M;
            }
        }


        int idx_to_put = idx + R - defaultL;
        if (searching_in_left_half) idx_to_put -= (1 << sorted_now_log);

        // if (idx < 64) {
        //     printf("idx=%d, L=%d, R=%d, searching_in_left_half=%d, comp_val=%d, idx_to_put=%d\n", idx, L, R, searching_in_left_half, comp_val, idx_to_put);
        // }

        if (idx_to_put < n) output_data[idx_to_put] = val;
    }
}
