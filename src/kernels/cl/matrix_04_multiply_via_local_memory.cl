#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define TILE_SIZE 16

__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h * cols=k
                       __global const float* b, // rows=k * cols=w
                       __global       float* c, // rows=h * cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float tile_a[TILE_SIZE][TILE_SIZE]; // local tile mem
    __local float tile_b[TILE_SIZE][TILE_SIZE];
    
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);
    
    const unsigned int global_x = group_x * TILE_SIZE + local_x; // global c_mat pos
    const unsigned int global_y = group_y * TILE_SIZE + local_y;
    
    float sum = 0.0f;
    
    for (unsigned int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {

        unsigned int a_tile_x = tile * TILE_SIZE + local_x; // tile from a_mat to lmem
        unsigned int a_tile_y = group_y * TILE_SIZE + local_y;
        
        if (a_tile_x < k && a_tile_y < h) {

            tile_a[local_y][local_x] = a[a_tile_y * k + a_tile_x];
        } else {
	
            tile_a[local_y][local_x] = 0.0f;
        }
        
	unsigned int b_tile_x = group_x * TILE_SIZE + local_x; // tile from a_mat to lmem
        unsigned int b_tile_y = tile * TILE_SIZE + local_y;
        
        if (b_tile_x < w && b_tile_y < k) {

            tile_b[local_y][local_x] = b[b_tile_y * w + b_tile_x];
        } else {
	
            tile_b[local_y][local_x] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (unsigned int i = 0; i < TILE_SIZE; ++i) {
		      
            sum += tile_a[local_y][i] * tile_b[i][local_x]; // sum for tile
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_x < w && global_y < h) {
    
        c[global_y * w + global_x] = sum;
    }
}
