#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void sparse_csr_matrix_vector_multiplication(
    gpu::WorkSize const& workSize,
    gpu::gpu_mem_32u const& csr_row_offsets_gpu,
    gpu::gpu_mem_32u const& csr_columns_gpu,
    gpu::gpu_mem_32u const& csr_values_gpu,
    gpu::gpu_mem_32u const& vector_values_gpu,
    gpu::gpu_mem_32u      & output_vector_values_gpu,
    uint nnz,
    uint nrows,
    uint ncols );
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}
