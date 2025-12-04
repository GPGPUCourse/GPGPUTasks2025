#include "sum_01_atomics_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_sum_01_atomics[] = {
		&opencl_binary_sum_01_atomics_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_sum_01_atomics = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_sum_01_atomics, opencl_versioned_binaries_sum_01_atomics + sizeof(opencl_versioned_binaries_sum_01_atomics) / sizeof(opencl_versioned_binaries_sum_01_atomics[0])), "sum_01_atomics");
}
