#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void zerofy(
    __global       unsigned int* a,
                   unsigned int n
)
{
    unsigned int i = get_global_id(0);
    if (i < n)
        a[i] = 0;    
}
