#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"
#define TILE 16

__attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float tile[TILE][TILE + 1]; // чтобы избежать банк-конфликтов когда несколько потоков обращаются к одной ячейке
    uint x = get_global_id(0); // колонка
    uint y = get_global_id(1); // ряд
    uint lx = get_local_id(0); // локальная колонка в рабочей группе (внутри тайла)
    uint ly = get_local_id(1); // локальный ряд в рабочей группе(внутри тайла)
    uint group_x = get_group_id(0); // номер рабочей группы по x(номер тайла по x)
    uint group_y = get_group_id(1); // номер рабочей группы по y(номер тайла по y)
    tile[ly][lx] = (x < w && y < h) ? matrix[y * w + x] : 0.0f; // загружаем в локальную память
    barrier(CLK_LOCAL_MEM_FENCE); // ждем пока все загрузят

    uint transposed_x = group_y * TILE + lx; // координаты в транспонированной матрице
    uint transposed_y = group_x * TILE + ly; // координаты в транспонированной матрице
    if (transposed_x < h && transposed_y < w) {
        transposed_matrix[transposed_y * h + transposed_x] = tile[lx][ly]; // записываем из локальной памяти в глобальную
        }

}
