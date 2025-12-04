# 0 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/mandelbrot.cl"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/mandelbrot.cl"




# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/rassert.cl" 1
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/../../defines.h" 1
# 2 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/rassert.cl" 2
# 6 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/mandelbrot.cl" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/../defines.h" 1
# 7 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/mandelbrot.cl" 2

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void mandelbrot(__global float* results,
                     unsigned int width, unsigned int height,
                     float fromX, float fromY,
                     float sizeX, float sizeY,
                     unsigned int iters, unsigned int isSmoothing)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);


}
