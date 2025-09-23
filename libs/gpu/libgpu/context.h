#pragma once

#include <vector>
#include <libgpu/opencl/engine.h>

typedef struct CUctx_st *cudaContext_t;
typedef struct CUstream_st *cudaStream_t;

namespace avk2 {
	class VulkanEngine;
	typedef std::shared_ptr<avk2::VulkanEngine>	sh_ptr_vk_engine;
}

namespace gpu {

class device_memory_pool;

class Context {
public:
	Context();

	enum Type {
		TypeUndefined,
		TypeOpenCL,
		TypeCUDA,
		TypeVulkan,
	};

	void	clear();

	void	init(int device_id_cuda);
	void	init(struct _cl_device_id *device_id_opencl);
	void	initVulkan(uint64_t device_id_vulkan);

	void	setVKValidationLayers(bool enabled);
	void	setMemoryGuards(bool enabled);
	void	setMemoryGuardsChecksAfterKernels(bool enabled);

	bool	isInitialized();
	bool	isActive();
	bool	isGPU();
	bool	isIntelGPU();
	bool	isGoldChecksEnabled();
	bool	isMemoryGuardsEnabled();
	bool	isMemoryGuardsChecksAfterKernelsEnabled();

	void	activate();

	size_t 				getCoresEstimate();
	uint64_t			getTotalMemory();
	uint64_t			getFreeMemory();
	uint64_t			getMaxMemAlloc();
	size_t				getMaxWorkgroupSize();
	std::vector<size_t>	getMaxWorkItemSizes();

	Type	type() const;

	ocl::sh_ptr_ocl_engine	cl() const;
	avk2::sh_ptr_vk_engine	vk() const;
	cudaStream_t			cudaStream() const;
	device_memory_pool &	memoryPool() const;

protected:
	class Data;

	Data *	data() const;

	Data *						data_;
	std::shared_ptr<Data>		data_ref_;
	static thread_local Data *	data_current_;
};

}
