#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_02_prefix_accumulation(
    __global uint* pref_sum, // it is segmented pref sum here, with segment of GROUP_SIZE
    const int n
)
{

}
