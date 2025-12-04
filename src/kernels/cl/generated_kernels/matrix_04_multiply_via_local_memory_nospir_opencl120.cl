# 0 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/matrix_04_multiply_via_local_memory.cl"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/matrix_04_multiply_via_local_memory.cl"




# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/../defines.h" 1
# 6 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/matrix_04_multiply_via_local_memory.cl" 2

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a,
                       __global const float* b,
                       __global float* c,
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{

}
