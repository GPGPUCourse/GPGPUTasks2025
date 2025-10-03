#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

typedef struct{
    unsigned int x, y, w, h;
} Index;

// returns abs. Index in row wise format.
inline unsigned int id(const Index p) { return p.y * p.w + p.x; }
inline Index transpose(const Index p) {
    Index q = {p.y, p.x, p.w, p.h};
    return q;
}

// __attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    unsigned int i = get_local_id(0);
    unsigned int j = get_local_id(1);

    unsigned int g_i = get_global_id(0);
    unsigned int g_j = get_global_id(1);

    // printf("liliput #%ld %ld does: %ld %ld\n", i, j, g_i, g_j);

    __local float matrix_cache[GROUP_SIZE];

    Index local_p = {i, j, GROUP_SIZE_X, GROUP_SIZE_Y};
    Index global_p = {g_i, g_j, w, h};

    // Чтобы получить колаесд паттаерн доступа к записи нам нужно взять нашу глобальную координату
    // отразить ее относительно диагонали в группе (что делать если все не квадратоное - непонятно)
    // и вернуть к глобальной системе отсчета
    // (g_i, g_j) = (i + k * GROUP_SIZE_X, j + q * GROUP_SIZE_Y)
    // значит вот такая хитро отраженная 
    // (g_i_t, g_j_t) = (j + k * GROUP_SIZE_X, i + q * GROUP_SIZE_Y) т.е
    // а теперь и ее надо транспонировать(Но только если локальный квадрат - не диагональный!!!)
    // и еще не надо транспонировать внутренние координаты если мы локальном диагональном квадрате
    // т.е записываем из
    // (j, i) -> (i + q * GROUP_SIZE_Y, j + k * GROUP_SIZE_X)
    unsigned int k = g_i / GROUP_SIZE_X;
    unsigned int q = g_j / GROUP_SIZE_Y;

    Index local_p_t = transpose(local_p);
    Index global_p_t = {local_p_t.x + k * GROUP_SIZE_X, local_p_t.y + q * GROUP_SIZE_Y, h, w};
    
    if (k == q)
        local_p_t = transpose(local_p_t);
    if (k != q)
        global_p_t = transpose(global_p_t);

    unsigned int local_id = id(local_p);
    unsigned int global_id = id(global_p);
    
    if (local_id < GROUP_SIZE && global_id < w * h)
        matrix_cache[local_id] = matrix[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    // some hard debugging
    // printf("global       : %u %u --> %u\n", global_p.x, global_p.y, id(global_p));
    // printf("local        : %u %u --> %u\n", local_p.x, local_p.y, id(local_p));
    // printf("k, q         : %u %u\n", k, q);
    // printf("local_tr     : %u %u --> %u %f\n", local_p_t.x, local_p_t.y, id(local_p_t), matrix_cache[id(local_p_t)]);
    // printf("g_transposed : %u %u --> %u\n", global_p_t.x, global_p_t.y, id(global_p_t));
    

    if (local_id < GROUP_SIZE && global_id < w * h)
        transposed_matrix[id(global_p_t)] = matrix_cache[id(local_p_t)];
}
