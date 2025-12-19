#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"


__kernel void sort_psum(
    __global unsigned gHist[4][256]
)
{
    unsigned l = get_local_id(0);
    unsigned sum = 0;
    for(unsigned i = 0; i < 256; i++)
    {
        unsigned val = gHist[l][i];
        gHist[l][i] = sum;
        sum += val;
    }
}
