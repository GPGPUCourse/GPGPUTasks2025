#include "aplusb_matrix_bad_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_aplusb_matrix_bad[] = {
		&opencl_binary_aplusb_matrix_bad_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_aplusb_matrix_bad = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_aplusb_matrix_bad, opencl_versioned_binaries_aplusb_matrix_bad + sizeof(opencl_versioned_binaries_aplusb_matrix_bad) / sizeof(opencl_versioned_binaries_aplusb_matrix_bad[0])), "aplusb_matrix_bad");
}
