#include "matrix_03_multiply_naive_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_matrix_03_multiply_naive[] = {
		&opencl_binary_matrix_03_multiply_naive_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_matrix_03_multiply_naive = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_matrix_03_multiply_naive, opencl_versioned_binaries_matrix_03_multiply_naive + sizeof(opencl_versioned_binaries_matrix_03_multiply_naive) / sizeof(opencl_versioned_binaries_matrix_03_multiply_naive[0])), "matrix_03_multiply_naive");
}
