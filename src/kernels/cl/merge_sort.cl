#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  block_size,
                   uint  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    const uint block_i = i % block_size;
    uint L = i - block_i;
    uint R = L + block_size;
    const uint constM = (L + R) >> 1;

    //printf("i = %d, block_size = %d, L = %d, constM = %d, R = %d\n", i, block_size, L, constM, R);

    uint comparee = input_data[i];
    if (i < constM) {
        L = constM;
    } else {
        R = constM;
        comparee += 1;
    }

    while (R > L) {
        uint M = (L + R) >> 1;
        if (M >= n || comparee <= input_data[M]) {
            R = M;
        } else {
            L = M + 1;
        }
    }

    //printf("i = %d, L = %d, constM = %d, input_data[i] = %d, input_data[L] = %d\n", i, L, constM, input_data[i], input_data[L]);

    output_data[i + L - constM] = input_data[i];
}
