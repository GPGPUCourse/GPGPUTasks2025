# 0 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/aplusb_matrix_good.cl"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/aplusb_matrix_good.cl"




# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/../defines.h" 1
# 6 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/aplusb_matrix_good.cl" 2

__kernel void aplusb_matrix_good(__global const uint* a,
                     __global const uint* b,
                     __global uint* c,
                     unsigned int width,
                     unsigned int height)
{







    const uint w = get_global_id(0);
    const uint h = get_global_id(1);
    if (w >= width || h >= height) {
        return;
    }
    const uint index = w + h * width;
    c[index] = a[index] + b[index];
}
