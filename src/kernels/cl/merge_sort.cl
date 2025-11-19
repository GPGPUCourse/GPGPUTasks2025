#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

uint binsearch(
    __global const uint* array,
    const int key,
    const uint lbound,
    const uint k,
    const bool leq) 
{
    int l = -1;
    int r = k;
    int m = 0;

    while (l < r - 1) {
        m = (l + r) / 2;
        if (leq) {
            if (array[lbound + m] <= key) {
                l = m;
            } else {
                r = m;
            }
        } else {
            if (array[lbound + m] < key) {
                l = m;
            } else {
                r = m;
            }
        }
    }

    return r;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
             const uint  sorted_k,
             const uint  n)
{
    const uint i = get_global_id(0);

    if (i >= n) {
        return;
    }
    
    const uint workspace_size = sorted_k * 2;

    const uint lbound = (i / workspace_size) * workspace_size;
    const uint mbound = min(lbound + sorted_k, n);
    const uint rbound = min(lbound + workspace_size, n);
    const uint mod = i % sorted_k;

    uint pos = 0;

    if (i < mbound) {
        pos = binsearch(input_data, input_data[i], mbound, rbound - mbound, false);
    } else {
        pos = binsearch(input_data, input_data[i], lbound, mbound - lbound, true);
    } 
    uint shift = lbound + pos + mod;

    //printf("%d %d %d %d %d\n", lbound, pos, mod, input_data[i], i < mbound); 

    if (shift < n) {
        output_data[shift] = input_data[i];
    }
}
