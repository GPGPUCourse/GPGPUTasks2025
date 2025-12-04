#include "aplusb_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_aplusb[] = {
		&opencl_binary_aplusb_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_aplusb = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_aplusb, opencl_versioned_binaries_aplusb + sizeof(opencl_versioned_binaries_aplusb) / sizeof(opencl_versioned_binaries_aplusb[0])), "aplusb");
}
