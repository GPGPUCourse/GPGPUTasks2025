#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#define TILE 16

__attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global       float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    __local float As[TILE][TILE]; // локальная память для тайла A
    __local float Bs[TILE][TILE]; // локальная память для тайла B

    uint x  = get_global_id(0); // колонка
    uint y  = get_global_id(1); // ряд
    uint lx = get_local_id(0);  // локальная колонка в тайле
    uint ly = get_local_id(1);  // локальный ряд в тайле

    float sum = 0.0f;

    uint numTiles = (k + TILE - 1) / TILE; // количество тайлов по k

    for (uint m = 0; m < numTiles; ++m) {
        uint a_col = m * TILE + lx; // колонка в A для текущего тайла
        if (y < h && a_col < k)
            As[ly][lx] = a[y * k + a_col];
        else
            As[ly][lx] = 0.0f;

        uint b_row = m * TILE + ly; // ряд в B для текущего тайла
        if (b_row < k && x < w)
            Bs[ly][lx] = b[b_row * w + x];
        else
            Bs[ly][lx] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE); // ждем пока все загрузят

        for (uint t = 0; t < TILE; ++t) // умножаем тайлы
            sum += As[ly][t] * Bs[t][lx];

        barrier(CLK_LOCAL_MEM_FENCE); // ждем пока все закончат использовать тайлы
    }

    if (x < w && y < h) // зщапись результата
        c[y * w + x] = sum;
}
