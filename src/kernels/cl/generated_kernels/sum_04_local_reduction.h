#include "sum_04_local_reduction_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_sum_04_local_reduction[] = {
		&opencl_binary_sum_04_local_reduction_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_sum_04_local_reduction = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_sum_04_local_reduction, opencl_versioned_binaries_sum_04_local_reduction + sizeof(opencl_versioned_binaries_sum_04_local_reduction) / sizeof(opencl_versioned_binaries_sum_04_local_reduction[0])), "sum_04_local_reduction");
}
