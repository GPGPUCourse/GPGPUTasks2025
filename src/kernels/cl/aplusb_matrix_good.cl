#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__kernel void aplusb_matrix_good(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    // код такой же как в BAD версии, но размер рабочей группы поменял:
    // в BAD версии он был равен 1xGROUP_SIZE, то есть при достаточно большой
    // высоте height все подгружаемые значения в рамках одного SM лежат в разных кэш-линиях
    // а тут наоборот, кэш-линия, когда подгружается, используется на фулл
    const unsigned int index0 = get_global_id(0);
    const unsigned int index1 = get_global_id(1);
    if (index0 >= width || index1 >= height)
        return;

    const unsigned int index = index0 + index1 * width;
    c[index] = a[index] + b[index];
}
