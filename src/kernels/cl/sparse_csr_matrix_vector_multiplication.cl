#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// идея что у меня есть csr_row_offsets[row] и csr_row_offsets[row + 1] они показывают где в массиве ненулевых
// элементов начинаются и заканчиваются элементы строки row
// csr_columns - массив столбцов для каждого ненулевого элемента
// csr_values - массив значений для каждого ненулевого элемента
// vector_values - входной вектор
// output_vector - выходной вектор (хранит сумму произведений для каждой строки)

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets, // массив смещений строк
    __global const uint* csr_columns,      // массив столбцов
    __global const uint* csr_values,       // массив значений
    __global const uint* vector_values,   // входной вектор
    __global uint* output_vector,         // выходной вектор
    const uint nrows                 // количество строк в разреженной матрице
){
    uint row = get_global_id(0);
    if (row >= nrows)
        return;
    uint from = csr_row_offsets[row];
    uint to = csr_row_offsets[row + 1];
    ulong acc = 0;
    for (uint i = from; i < to; i++)
    {
        uint col = csr_columns[i];
        acc += (ulong)csr_values[i] * (ulong)vector_values[col];
    }
    output_vector[row] = (uint)acc;
}