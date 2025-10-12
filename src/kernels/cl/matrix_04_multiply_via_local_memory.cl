#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    unsigned int col = get_global_id(0);
    unsigned int row = GROUP_SIZE_X / GROUP_SIZE_Y * get_global_id(1);
    unsigned int loc_col = get_local_id(0);
    unsigned int loc_row = GROUP_SIZE_X / GROUP_SIZE_Y * get_local_id(1);

    __local float A[GROUP_SIZE_X][GROUP_SIZE_X];
    __local float B[GROUP_SIZE_X][GROUP_SIZE_X];

    float accum[GROUP_SIZE_X / GROUP_SIZE_Y];
    for (int row_i = 0; row_i < GROUP_SIZE_X / GROUP_SIZE_Y; ++row_i) {
        accum[row_i] = 0.0f;
    }

    for (int offset = 0; offset < k; offset += GROUP_SIZE_X) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int row_i = 0; row_i < GROUP_SIZE_X / GROUP_SIZE_Y; ++row_i) {
            A[loc_row + row_i][loc_col] = a[(row + row_i) * k + offset + loc_col];
            B[loc_row + row_i][loc_col] = b[(offset + loc_row + row_i) * w + col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float a_elem = a[row * k + offset + loc_col];
        for (int row_i = 0; row_i < GROUP_SIZE_X / GROUP_SIZE_Y; ++row_i) {
            for (int i = 0; i < GROUP_SIZE_X; ++i) {
                // здесь банк-конфликт - весь wave обращается в одну ячейку A
                accum[row_i] += A[loc_row + row_i][i] * B[i][loc_col];
                // здесь нет банк-конфликтов внутри wave-а, но скорость та же самая
                // accum[row_i] += A[loc_row + row_i][(i + loc_col) % GROUP_SIZE_X] * B[(i + loc_col) % GROUP_SIZE_X][loc_col];
                // разгадка - chatGPT сказал, что broadcast одного значения работает быстро и банк-конфликта не будет (вроде это не упомяналось на лекции, было бы славно упомянуть)
            }
        }
    }
    for (int row_i = 0; row_i < GROUP_SIZE_X / GROUP_SIZE_Y; ++row_i) {
        c[(row + row_i) * w + col] = accum[row_i];
    }
}
