#include "matrix_04_multiply_via_local_memory_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_matrix_04_multiply_via_local_memory[] = {
		&opencl_binary_matrix_04_multiply_via_local_memory_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_matrix_04_multiply_via_local_memory = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_matrix_04_multiply_via_local_memory, opencl_versioned_binaries_matrix_04_multiply_via_local_memory + sizeof(opencl_versioned_binaries_matrix_04_multiply_via_local_memory) / sizeof(opencl_versioned_binaries_matrix_04_multiply_via_local_memory[0])), "matrix_04_multiply_via_local_memory");
}
