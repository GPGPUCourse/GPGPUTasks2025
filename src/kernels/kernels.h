#pragma once

#include <libgpu/vulkan/engine.h>

#include "../models.h"

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize, const models::CSRMatrix& matrix, const gpu::gpu_mem_32u& vector, gpu::gpu_mem_32u& output);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getSparseCSRMatrixVectorMult();
}
