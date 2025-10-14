#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define GROUP_SIZE_X_MUL 16
#define GROUP_SIZE_Y_MUL 16

__attribute__((reqd_work_group_size(GROUP_SIZE_X_MUL, GROUP_SIZE_Y_MUL, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    
    if (x >= w || y >= h) {
        return;
    }
    
    size_t local_x = get_local_id(0);
    size_t local_y = get_local_id(1);
    
    #define y_(i) (y * VEC_SIZE_V + (i))
    #define x_(j) (x * VEC_SIZE_H + (j))
    
    float buffer[VEC_SIZE_V][VEC_SIZE_H];
    #pragma unroll
    for (uint i = 0; i < VEC_SIZE_V; ++i) {
        #pragma unroll
        for (uint j = 0; j < VEC_SIZE_H; ++j) {
            buffer[i][j] = 0.0f;
        }
    }
    
    local float la[TILE][TILE][VEC_SIZE_V];
    local float lb[TILE][VEC_SIZE_H][TILE];
    
    for (uint ki = 0; ki < k; ki += TILE) {
        #pragma unroll
        for (uint i = 0; i < VEC_SIZE_V; ++i) {
            if (y_(i) < h && (ki + local_x) < k) {
                la[local_y][local_x][i] = a[y_(i) * k + (ki + local_x)];
            } else {
                la[local_y][local_x][i] = 0.0f;
            }
        }
        #pragma unroll
        for (uint j = 0; j < VEC_SIZE_H; ++j) {
            if (x_(j) < w && (ki + local_y) < k) {
                lb[local_y][j][local_x] = b[(ki + local_y) * w + x_(j)];
            } else {
                lb[local_y][j][local_x] = 0.0f;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (uint z = 0; z < TILE && (ki + z) < k; ++z) {
            float la_buf[VEC_SIZE_V];
            #pragma unroll
            for (uint i = 0; i < VEC_SIZE_V; ++i) {
                la_buf[i] = la[local_y][z][i];
            }
            #pragma unroll
            for (uint i = 0; i < VEC_SIZE_V; ++i) {
                float lb_buf[VEC_SIZE_H];
                for (uint j = 0; j < VEC_SIZE_H; ++j) {
                    lb_buf[j] = lb[z][j][local_x];
                }
                #pragma unroll
                for (uint j = 0; j < VEC_SIZE_H; ++j) {
                    buffer[i][j] += la_buf[i] * lb_buf[j];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    #pragma unroll
    for (uint i = 0; i < VEC_SIZE_V; ++i) {
        #pragma unroll
        for (uint j = 0; j < VEC_SIZE_H; ++j) {
            if (y_(i) < h && x_(j) < w) {
                c[y_(i) * w + x_(j)] = buffer[i][j];
            }
        }
    }
}
