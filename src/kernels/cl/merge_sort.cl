#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if (i < n)
    {
        const int merge_size = 2 * sorted_k;
        const int part_num = i / merge_size;
        const int id_inside_part = i % merge_size;
        const int ll = part_num * merge_size;
        const int lr = min(ll + sorted_k, n);
        const int rl = lr;
        const int rr = min(rl + sorted_k, n);
        const int l_size = lr - ll;
        const int r_size = rr - rl;
        const uint el = input_data[ll + id_inside_part];

        if (id_inside_part < l_size)
        {
            int l = 0;
            int r = r_size;
            while (r - l > 0)
            {
                int mid = (r + l) / 2;
                if (input_data[rl + mid] < el)
                {
                    l = mid + 1;
                }
                else
                {
                    r = mid;
                }
            }
            output_data[ll + l + id_inside_part] = el;
        }
        else if (id_inside_part < l_size + r_size)
        {
            int l = 0;
            int r = l_size;
            while (r - l > 0)
            {
                int mid = (r + l) / 2;
                if (input_data[ll + mid] <= el)
                {
                    l = mid + 1;
                }
                else
                {
                    r = mid;
                }
            }
            output_data[ll + id_inside_part - l_size + l] = el;
        }
    }
}
