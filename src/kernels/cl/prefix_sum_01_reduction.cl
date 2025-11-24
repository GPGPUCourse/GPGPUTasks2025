#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{

    // TODO
    unsigned int gid = get_global_id(0); // индекс потока
    unsigned int index = gid * 2;   // каждый поток отвечает за пару элементов 2*gid и 2*gid + 1
    if (index < n) {
        uint first = pow2_sum[index]; // первый элемент пары всегда существует
        uint second = (index + 1 < n) ? pow2_sum[index + 1] : 0; // если есть второй элемент пары - берем его, иначе 0
        next_pow2_sum[gid] = first + second; // записываем сумму пары в следующий буфер
    }

}
