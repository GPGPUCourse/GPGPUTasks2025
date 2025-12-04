#include "sum_02_atomics_load_k_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_sum_02_atomics_load_k[] = {
		&opencl_binary_sum_02_atomics_load_k_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_sum_02_atomics_load_k = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_sum_02_atomics_load_k, opencl_versioned_binaries_sum_02_atomics_load_k + sizeof(opencl_versioned_binaries_sum_02_atomics_load_k) / sizeof(opencl_versioned_binaries_sum_02_atomics_load_k[0])), "sum_02_atomics_load_k");
}
