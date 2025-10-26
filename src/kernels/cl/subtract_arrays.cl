#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void subtract_arrays(
    __global       uint* arr1,
    __global const uint* arr2,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n)
        arr1[i] -= arr2[i];
}
