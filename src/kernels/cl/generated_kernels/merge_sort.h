#include "merge_sort_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_merge_sort[] = {
		&opencl_binary_merge_sort_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_merge_sort = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_merge_sort, opencl_versioned_binaries_merge_sort + sizeof(opencl_versioned_binaries_merge_sort) / sizeof(opencl_versioned_binaries_merge_sort[0])), "merge_sort");
}
