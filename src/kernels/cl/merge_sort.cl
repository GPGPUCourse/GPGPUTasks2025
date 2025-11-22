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
    const unsigned int gid = get_global_id(0);
    if (gid >= n) return;

    const int block = 2 * sorted_k;

    const int i_block = (gid / block) * block;

    const int a0 = i_block;
    const int b0 = i_block + sorted_k;

    const int lenB = (b0 < n) ? min(sorted_k, n - b0) : 0;

    if (gid < b0 || lenB == 0)
    {
        // lower bound
        const uint in = input_data[gid];

        int lo = 0, hi = lenB;
        while (lo < hi)
        {
            const int mid = (lo + hi) >> 1;
            const uint x = input_data[b0 + mid];
            if (x < in) lo = mid + 1;
            else hi = mid;
        }
        const int cntB = lo;
        output_data[i_block + gid - a0 + cntB] = in;
    }
    else
    {
        // upper bound
        const uint in = input_data[gid];

        int lo = 0, hi = min(sorted_k, n - a0);
        while (lo < hi)
        {
            const int mid = (lo + hi) >> 1;
            const uint x = input_data[a0 + mid];
            if (x <= in) lo = mid + 1;
            else hi = mid;
        }
        const int cntA = lo;
        output_data[i_block + gid - b0 + cntA] = in;
    }
}
