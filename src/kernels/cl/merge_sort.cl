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
    if (i >= n) return;
    int start_block = (i / (2*sorted_k)) * (2*sorted_k);
    int isRight = (i/sorted_k) % 2, pos = i - (i/sorted_k) * sorted_k + 1, l, r; // pos>=1
    if (isRight){
        r = (i/sorted_k) * sorted_k;
        l = r - sorted_k;
    } else {
        l = (i/sorted_k) * sorted_k + sorted_k;
        r = l + sorted_k;
    }
    int bl = l, br = r, bm;
    uint x = input_data[i];
    while(bl < br){
        bm = (bl + br) / 2;
        uint mid = input_data[bm];
        if (isRight){
            if(mid <= x) bl = bm + 1;
            else br = bm;
        } else {
            if(mid < x) bl = bm + 1;
            else br = bm;
        }
    }
    bl -= l;
    output_data[start_block + bl + pos - 1] = x;
}
