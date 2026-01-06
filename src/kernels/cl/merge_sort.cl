#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  sorted_k,
                   uint  n)
{
    const unsigned int i = get_global_id(0);
    if(i >= n) return;

    const __private unsigned int block = i / sorted_k;
    const __private unsigned int s = (block & ~1) * sorted_k;

    // begin binary search
    __private unsigned int l_s = s;
    __private unsigned int l_e = min(l_s + sorted_k, n);
    __private unsigned int r_s = s + sorted_k;
    __private unsigned int r_e = min(r_s + sorted_k, n);

    __private unsigned int l = 0;
    __private unsigned int m = 0;
    __private unsigned int r = 0;
    __private unsigned int rank = 0;
    // binary search
    if((block & 1) == 0) {
        l = r_s;
        r = r_e;
    } else {
        l = l_s;
        r = l_e;
    }
    while(l < r) {
        m = (l + r)/2;
        if(input_data[m] < input_data[i] || ((block & 1) == 1 && input_data[m] == input_data[i])) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    rank = i + l - r_s - l_s;
    output_data[s + rank] = input_data[i];

}
