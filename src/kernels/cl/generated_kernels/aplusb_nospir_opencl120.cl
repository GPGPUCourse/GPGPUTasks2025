# 0 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/aplusb.cl"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/aplusb.cl"




# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/rassert.cl" 1
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/../../defines.h" 1
# 2 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/rassert.cl" 2
# 6 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/aplusb.cl" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/../defines.h" 1
# 7 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/aplusb.cl" 2

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void aplusb(__global const uint* a,
                     __global const uint* b,
                     __global uint* c,
                            unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    if (index == 0) {




        printf("OpenCL printf test in aplusb.cl kernel! a[index]=%d b[index]=%d \n", a[index], b[index]);
    }




    ;
    ;

    c[index] = a[index] + b[index];
}
