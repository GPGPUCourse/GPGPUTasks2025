#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{

    // TODO
    const unsigned int index = get_global_id(0);
    const unsigned int step = 1u << pow2; // step = 2^pow2
    const unsigned int block_id = index >> pow2; // block_id = index / step
    if (block_id == 0) // если это первый блок - ничего не добавляем тк там нечего
        return;
    if ((index & step) == 0u) // кароч тк мы строим преф сумму по шагам то на шаге 2^pow2 мы меняем только те индексы кооторые кратны этому шагу иначе ничо не делаем
        return;
    prefix_sum_accum[index] += pow2_sum[block_id - 1];
}
