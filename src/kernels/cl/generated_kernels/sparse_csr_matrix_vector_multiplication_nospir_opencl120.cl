# 0 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/sparse_csr_matrix_vector_multiplication.cl"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/sparse_csr_matrix_vector_multiplication.cl"




# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/rassert.cl" 1
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/../../defines.h" 1
# 2 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/rassert.cl" 2
# 6 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/sparse_csr_matrix_vector_multiplication.cl" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/../defines.h" 1
# 7 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/sparse_csr_matrix_vector_multiplication.cl" 2

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector_values,
    const uint nrows
)
{
    const uint row = get_global_id(0);

    if (row >= nrows) return;

    ulong ac = 0;
    for (uint i = csr_row_offsets[row]; i < csr_row_offsets[row+ 1]; ++i) {
        ac += (ulong)csr_values[i] * (ulong)vector_values[csr_columns[i]];
    }
    output_vector_values[row] = (uint)ac;
}
