#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  size,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    
    if (i >= n) return;

    uint group_index_init = i / size;
    bool is_first = group_index_init % 2 == 0;
    uint group_index_pair = is_first ? group_index_init + 1 : group_index_init - 1;
    
    uint group_init = group_index_init * size;
    uint group_pair = group_index_pair * size;

    uint index_init = i % size;

    uint l = group_pair;
    uint r = group_pair + size;
    uint m;

    while (l < r) {
        m = l + (r - l) / 2;
        if (input_data[i] > input_data[m]) l = m + 1;
        else if (!is_first && input_data[i] == input_data[m]) l = m + 1;
        else r = m;
    }

    uint index_pair = l - group_pair;
    
    uint ind = (is_first ? group_init : group_pair) + index_init + index_pair;
    output_data[ind] = input_data[i];
}
