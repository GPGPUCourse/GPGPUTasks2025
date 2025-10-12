#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* at, // rows=k x cols=h
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    // одна ворк-группа будет вычислять блок матрицы С размером LDS_TILE_XY x LDS_TILE_XY (64x64)
    // это координаты левого верхнего угла этого блока внутри матрицы С
    unsigned int LDS_row = get_group_id(0) / (w / LDS_TILE_XY) * LDS_TILE_XY;
    unsigned int LDS_col = get_group_id(1) % (w / LDS_TILE_XY) * LDS_TILE_XY;
    unsigned int loc_id = get_local_id(0);
    // каждый поток будет вычислять блок матрицы С размера REGISTER_TILE_XY x REGISTER_TILE_XY (4x4)
    // это координаты левого верхнего угла этого блока внутри lds блока
    unsigned int reg_row = loc_id / (LDS_TILE_XY / REGISTER_TILE_XY) * REGISTER_TILE_XY;
    unsigned int reg_col = loc_id % (LDS_TILE_XY / REGISTER_TILE_XY) * REGISTER_TILE_XY;

    __local float At_lds[REGISTER_TILE_XY][LDS_TILE_XY + GROUP_SIZE / WARP_SIZE]; // добавка для устранения банк-конфликтов
    __local float B_lds[REGISTER_TILE_XY][LDS_TILE_XY + GROUP_SIZE / WARP_SIZE];

    // строка подматрицы A
    float A_reg[REGISTER_TILE_XY];
    // столбец подматрицы B
    float B_reg[REGISTER_TILE_XY];

    float accum[REGISTER_TILE_XY][REGISTER_TILE_XY];
    for (int row_i = 0; row_i < REGISTER_TILE_XY; ++row_i) {
        for (int col_i = 0; col_i < REGISTER_TILE_XY; ++col_i) {
            accum[row_i][col_i] = 0.0f;
        }
    }

    float A_future[REGISTER_TILE_XY]; // двойная буферизация
    float B_future[REGISTER_TILE_XY];

    #pragma unroll // старт для двойной буферизации
    for (unsigned int row_i = 0; row_i < REGISTER_TILE_XY; row_i += GROUP_SIZE / LDS_TILE_XY) {
        unsigned int read_row = loc_id / LDS_TILE_XY + row_i;
        unsigned int read_col = loc_id % LDS_TILE_XY;
        A_future[row_i] = at[read_row * h + LDS_row + read_col];
        B_future[row_i] = b[read_row * w + LDS_col + read_col];
    }
    for (int lds_offset = 0; lds_offset < k; lds_offset += REGISTER_TILE_XY) {
        // считываем lds блок
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (unsigned int row_i = 0; row_i < REGISTER_TILE_XY; row_i += GROUP_SIZE / LDS_TILE_XY) {
            unsigned int read_row = loc_id / LDS_TILE_XY + row_i;
            unsigned int read_col = loc_id % LDS_TILE_XY;
            // убиваем банк-конфликты
            At_lds[read_row][read_col + read_col / WARP_SIZE] = A_future[row_i];
            B_lds[read_row][read_col + read_col / WARP_SIZE] = B_future[row_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // просим кэш-линни заранее на следующую итерацию
        if (lds_offset + REGISTER_TILE_XY < k) {
            #pragma unroll
            for (unsigned int row_i = 0; row_i < REGISTER_TILE_XY; row_i += GROUP_SIZE / LDS_TILE_XY) {
                unsigned int read_row = loc_id / LDS_TILE_XY + row_i;
                unsigned int read_col = loc_id % LDS_TILE_XY;
                A_future[row_i] = at[(lds_offset + REGISTER_TILE_XY + read_row) * h + LDS_row + read_col];
                B_future[row_i] = b[(lds_offset + REGISTER_TILE_XY + read_row) * w + LDS_col + read_col];
            }
        }
        
        #pragma unroll
        for (int i = 0; i < REGISTER_TILE_XY; ++i) {
            // считываем нужную строку и столбец - так меньше локальной памяти используется, но большей чтений
            // до добавления двойной буферизации примерно одинаково по скорости
            #pragma unroll
            for (int col_i = 0; col_i < REGISTER_TILE_XY; ++col_i) {
                // транспонируем A обратно
                A_reg[col_i] = At_lds[i][reg_row + col_i + reg_row / WARP_SIZE]; // убиваем банк-конфликты
                B_reg[col_i] = B_lds[i][reg_col + col_i + reg_col / WARP_SIZE];
            }
            
            // считаем
            #pragma unroll
            for (int row_i = 0; row_i < REGISTER_TILE_XY; ++row_i) {
                #pragma unroll
                for (int col_i = 0; col_i < REGISTER_TILE_XY; ++col_i) {
                    accum[row_i][col_i] += A_reg[row_i] * B_reg[col_i];
                }
            }
        }
    }
    for (int row_i = 0; row_i < REGISTER_TILE_XY; ++row_i) {
        for (int col_i = 0; col_i < REGISTER_TILE_XY; ++col_i) {
            c[(LDS_row + reg_row + row_i) * w + (LDS_col + reg_col + col_i)] = accum[row_i][col_i];
        }
    }
}
