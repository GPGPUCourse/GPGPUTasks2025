#pragma once

#include <libgpu/vulkan/engine.h>
#include "defines.h"


using u32 = unsigned int;
namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
                                             const u32* csr_row_offsets,
                                             const u32* csr_columns,
                                             const u32* csr_values,
                                             const  u32* vector_values,
                                             u32* output_buffer,
                                             const u32 nrows, const u32 nnz);
} // namespace cuda

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}
