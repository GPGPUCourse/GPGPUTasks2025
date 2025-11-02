#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
select_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int n)
{
    const unsigned int i = get_global_id(0);
    const unsigned int l = get_local_id(0);
    const unsigned int g = get_group_id(0);

    __local uint chunk[GROUP_SIZE];

    chunk[l] = (i < n) ? input_data[i] : (~0);

    int before_me = 0;
    // uint my_val = chunk[l];

    barrier(CLK_LOCAL_MEM_FENCE);

    int sorted = 1;

    while (sorted < GROUP_SIZE) {
        uint my_val = chunk[l];

        int begin = l & (~((sorted << 1) - 1));
        int mid = begin + sorted;
        int end = begin + 2 * sorted;
        bool is_left = l < mid;

        int s_l = is_left ? (mid - 1) : (begin - 1);
        int s_r = is_left ? end : mid;

        while (s_r > s_l + 1) {
            int s_m = (s_l + s_r) / 2;
            uint midval = chunk[s_m];

            bool is_less = is_left ? (my_val > midval) : (my_val >= midval);

            if (is_less) {
                s_l = s_m;
            } else
                s_r = s_m;
        }


        barrier(CLK_LOCAL_MEM_FENCE);

        chunk[l + s_r - mid] = my_val;


        barrier(CLK_LOCAL_MEM_FENCE);
        sorted *= 2;
    }

    // for (int k = 0; k < l; k++) {
    //     if (chunk[k] <= my_val)
    //         before_me += 1;
    // }

    // for (int k = l + 1; k < GROUP_SIZE; k++) {
    //     if (chunk[k] < my_val)
    //         before_me += 1;
    // }

    // if (i == 0) {
    //     for (int j = 0; j < GROUP_SIZE; j++) {
    //         printf ("asd %d, i = %d\n", chunk[j], j);
    //     }
    // }

    // if (i < n) {
        // if (g == 1) {
        //     printf("g * GROUP_SIZE + before_me = %d, my_val = %d\n", g * GROUP_SIZE + before_me, my_val);
        // }
        // output_data[g * GROUP_SIZE + before_me] = my_val;
        output_data[i] = chunk[l];
    // }
}
