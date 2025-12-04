# 0 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/sum_01_atomics.cl"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/sum_01_atomics.cl"




# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/../defines.h" 1
# 6 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/sum_01_atomics.cl" 2

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void sum_01_atomics(__global const uint* a,
                             __global uint* sum,
                                    unsigned int n)
{
    const uint index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(sum, a[index]);
}
