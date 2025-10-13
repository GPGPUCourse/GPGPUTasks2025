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
    __local float As[TILE][TILE]; // локальная память для тайла из матрицы A
    __local float Bs[TILE][TILE]; // локальная память для тайла из
    uint x = get_global_id(0); // колонка
    uint y = get_global_id(1); // ряд
    uint lx = get_local_id(0); // локальная колонка в рабочей группе (внутри тайла)
    uint ly = get_local_id(1); // локальный ряд в рабочей группе(внутри тайла)
    uint group_x = get_group_id(0); // номер рабочей группы по x(номер тайла по x)
    uint group_y = get_group_id(1); // номер рабочей группы по y(номер тайла по y)
    float sum = 0.0f;
    uint numTiles = (k + TILE - 1) / TILE; // количество тайлов по k
    for (uint m = 0; m < tiles; ++m) {
        uint a_col = m * TILE + tx; // колонка в матрице A для текущего тайла
        if (row < h && a_col < k)
            As[ty][tx] = a[row * k + a_col]; // подтайл A[row, m*TILE ..]
        else
            As[ty][tx] = 0.0f; //  0 если вышли за границы

        uint b_row = m * TILE + ty;
        if (b_row < k && col < w)
            Bs[ty][tx] = b[b_row * w + col];
        else
            Bs[ty][tx] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE); // ждем пока все загрузят

        for (uint t = 0; t < TILE; ++t) {
            acc += As[ty][t] * Bs[t][tx]; // умножаем подтайлы
        }

        barrier(CLK_LOCAL_MEM_FENCE); // ждем пока все закончат вычисления перед загрузкой новых тайлов
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
