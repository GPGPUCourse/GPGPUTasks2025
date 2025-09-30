#define _SHORT_FILE_ "gpu/context.cpp"

#include "context.h"
#include <libgpu/device_memory_pool.h>
#include <libgpu/vulkan/engine.h>

#ifdef CUDA_SUPPORT
#include <libgpu/cuda/utils.h>
#include <libgpu/cuda/cuda_api.h>
#include <cuda_runtime_api.h>
#endif

namespace gpu {

thread_local Context::Data *Context::data_current_ = 0;

class Context::Data {
public:
	Data();
	~Data();

	Type	type;

	int						cuda_device;
	cudaContext_t			cuda_context;
	cudaStream_t			cuda_stream;
	struct _cl_device_id *	ocl_device;
	ocl::sh_ptr_ocl_engine	ocl_engine;
	uint64_t				vk_device_id;
	avk2::sh_ptr_vk_engine	vk_engine;
	bool					vk_enable_validation_layers;
	bool					memory_guards_enabled;
	bool					memory_guards_checks_after_kernels_enabled;
	device_memory_pool		memory_pool;
	bool					activated;
#ifdef CUDA_SUPPORT
	cudaDeviceProp			cuda_device_prop;
#endif
};

Context::Data::Data()
{
	type						= TypeUndefined;
	cuda_device					= 0;
	cuda_context				= 0;
	cuda_stream					= 0;
	ocl_device					= 0;
	vk_device_id				= 0;
	vk_enable_validation_layers	= false;
	memory_guards_enabled		= false;
	memory_guards_checks_after_kernels_enabled	= false;
	activated					= false;
}

Context::Data::~Data()
{
	// we need to clear memory pool before GPGPU context destruction
	// because we want to execute cudaFree on all buffers in memory_pool while context still exists (and thus - memory buffers still exists)
	// otherwise if memory_pool is non-empty - cudaFree will try to deallocate them after context destruction and will raise error invalid argument (1)
	// see ticket #4965
	memory_pool.clear();

	// Vulkan Kernels destruction leads to destruction of their rassert code storages
	// and those gpu buffers on their de-allocations will require active Context,
	// so we need to force their deallocation before active Context will be cleared 
	if (vk_engine) {
		vk_engine->clearKernels();
		vk_engine->clearStagingBuffers();
	}

	if (data_current_ != this) {
		if (data_current_ != 0) {
			std::cout << "Another GPU context found on context destruction" << std::endl;
		}
	} else {
		data_current_ = 0;
	}

#ifdef CUDA_SUPPORT
	if (cuda_stream) {
		cudaError_t err;
		CUDA_TRACE(err = cudaStreamDestroy(cuda_stream));
		if (cudaSuccess != err)
			std::cerr << "Warning: cudaStreamDestroy failed: " << cuda::formatError(err) << std::endl;
	}

#ifndef CUDA_USE_PRIMARY_CONTEXT
	if (cuda_context) {
		CUresult err;
		CUDA_TRACE(err = cuCtxDestroy(cuda_context));
		if (CUDA_SUCCESS != err)
			std::cerr << "Warning: cuCtxDestroy failed: " << cuda::formatDriverError(err) << std::endl;
	}
#endif
#endif
}

Context::Context()
{
	data_ = data_current_;
}

void Context::clear()
{
	data_ = NULL;
}

void Context::init(int device_id_cuda)
{
#ifdef CUDA_SUPPORT
#ifndef CUDA_USE_PRIMARY_CONTEXT
	if (!cuda_api_init())
		throw cuda::cuda_exception("Can't load nvcuda library");
#endif
	std::shared_ptr<Data> data = std::make_shared<Data>();
	data->type				= TypeCUDA;
	data->cuda_device		= device_id_cuda;
	data_ref_	= data;
#endif
}

void Context::init(struct _cl_device_id *device_id_opencl)
{
	std::shared_ptr<Data> data = std::make_shared<Data>();
	data->type				= TypeOpenCL;
	data->ocl_device		= device_id_opencl;
	data_ref_	= data;
}

void Context::initVulkan(uint64_t device_id_vulkan)
{
	std::shared_ptr<Data> data = std::make_shared<Data>();
	data->type							= TypeVulkan;
	data->vk_device_id					= device_id_vulkan;
	data_ref_	= data;
}

void Context::setVKValidationLayers(bool enabled)
{
	rassert(data_ref_, 351341241);
	data_ref_->vk_enable_validation_layers = enabled;
}

void Context::setMemoryGuards(bool enabled)
{
	rassert(data_ref_, 769596522);
	data_ref_->memory_guards_enabled = enabled;
}

void Context::setMemoryGuardsChecksAfterKernels(bool enabled)
{
	rassert(data_ref_, 409350039);
	data_ref_->memory_guards_checks_after_kernels_enabled = enabled;
}

bool Context::isInitialized()
{
	return data_ref_.get() && data_ref_->type != TypeUndefined;
}

bool Context::isActive()
{
	return data_current_ != 0;
}

bool Context::isGPU()
{
	return (type() != TypeUndefined);
}

bool Context::isIntelGPU()
{
	if (type() != TypeOpenCL) {
		return false;
	}

	return cl()->deviceInfo().isIntelGPU();
}

bool Context::isGoldChecksEnabled()
{
	return false; // TODO: Make it switchable
}

bool Context::isMemoryGuardsEnabled()
{
	return data_->memory_guards_enabled;
}

bool Context::isMemoryGuardsChecksAfterKernelsEnabled()
{
	return data_->memory_guards_checks_after_kernels_enabled;
}

void Context::activate()
{
	if (!data_ref_)
		throw std::runtime_error("Unexpected GPU context activate call");

	// create cuda stream on first activate call
	if (!data_ref_->activated) {
#ifdef CUDA_SUPPORT
		if (data_ref_->type == TypeCUDA) {
			CUDA_SAFE_CALL( cudaGetDeviceProperties(&data_ref_->cuda_device_prop, data_ref_->cuda_device) );

#ifndef CUDA_USE_PRIMARY_CONTEXT
			// It is claimed that contexts are thread safe starting from CUDA 4.0.
			// Nevertheless, we observe crashes in nvcuda.dll if the same device is used in parallel from 2 threads using its primary context.
			// To avoid this problem we create a separate standard context for each processing thread.
			// https://devtalk.nvidia.com/default/topic/519087/cuda-programming-and-performance/cuda-context-and-threading/post/3689477/#3689477
			// http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#axzz4g8KX5QV5

			CUdevice device = 0;
			CU_SAFE_CALL( cuDeviceGet(&device, data_ref_->cuda_device) );

                        int computeMode = 9999;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
                        CU_SAFE_CALL( cuDeviceGetAttribute(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device) );
#else
                        // CUDA <= 12.x: keep old runtime field
                        computeMode = data_ref_->cuda_device_prop.computeMode;
#endif

                        bool computeModeIsProhibited = (computeMode == cudaComputeModeProhibited);
                        bool computeModeIsExclusive = (computeMode == cudaComputeModeExclusive);

			// Note that cuCtxCreate(...) returns CUDA_ERROR_UNKNOWN if the compute mode of the device is CU_COMPUTEMODE_PROHIBITED
			// See https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
			if (computeModeIsProhibited) {
				throw cuda::cuda_exception("GPU device " + std::string(data_ref_->cuda_device_prop.name) + " in ComputeMode=PROHIBITED which is not supported, please use nvidia-smi to change compute mode and restart Metashape");
			}
			// Note that cuCtxCreate(...) returns CUDA_ERROR_INVALID_DEVICE or CUDA_ERROR_UNKNOWN if the device is CU_COMPUTEMODE_EXCLUSIVE
			if (computeModeIsExclusive) {
				throw cuda::cuda_exception("GPU device " + std::string(data_ref_->cuda_device_prop.name) + " in ComputeMode=EXCLUSIVE which is not supported, please use nvidia-smi to change compute mode and restart Metashape");
			}

			CUresult err;

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
                        // CUDA 13+: new signature with CUctxCreateParams
                        CUctxCreateParams params{};
                        CUDA_TRACE(err = cuCtxCreate(&data_ref_->cuda_context, &params, 0, device));
#else
                        // CUDA 12.x and older
                        CUDA_TRACE(err = cuCtxCreate(&data_ref_->cuda_context, 0, device));
#endif

			cuda::reportErrorCU(err, __LINE__, "cuCtxCreate failed: ");
#else
			CUDA_SAFE_CALL( cudaSetDevice(data_ref_->cuda_device) );
#endif
			CUDA_SAFE_CALL( cudaStreamCreate(&data_ref_->cuda_stream) );
		}
#endif

		if (data_ref_->type == TypeOpenCL) {
			ocl::sh_ptr_ocl_engine engine = std::make_shared<ocl::OpenCLEngine>();
			engine->init(data_ref_->ocl_device);
			data_ref_->ocl_engine = engine;
		}

		if (data_ref_->type == TypeVulkan) {
			avk2::sh_ptr_vk_engine engine = std::make_shared<avk2::VulkanEngine>();
			engine->init(data_ref_->vk_device_id, data_ref_->vk_enable_validation_layers);
			data_ref_->vk_engine = engine;
		}

		data_ref_->activated = true;
	}

	if (data_current_ && data_current_ != data_ref_.get())
		throw std::runtime_error("Another GPU context is already active");

	data_			= data_ref_.get();
	data_current_	= data_;
}

Context::Data *Context::data() const
{
	if (!data_)
		throw std::runtime_error("Null context");

	return data_;
}

size_t Context::getCoresEstimate()
{
	size_t compute_units = 1;

	switch (type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		compute_units = (size_t) data()->cuda_device_prop.multiProcessorCount;
		break;
#endif
	case Context::TypeOpenCL:
		compute_units = cl()->maxComputeUnits();
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	return compute_units * 256;
}

uint64_t Context::getTotalMemory()
{
	uint64_t res = 0;

	switch (type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
	{
		res = data()->cuda_device_prop.totalGlobalMem;
		break;
	}
#endif
	case Context::TypeOpenCL:
		res = cl()->totalMemSize();
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	return res;
}

uint64_t Context::getFreeMemory()
{
	uint64_t res = 0;

	switch (type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
	{
		size_t total_mem_size = 0;
		size_t free_mem_size = 0;
		CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_size, &total_mem_size));
		res = free_mem_size;
		break;
	}
#endif
	case Context::TypeOpenCL:
		res = cl()->freeMem();
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	return res;
}

uint64_t Context::getMaxMemAlloc()
{
	uint64_t max_mem_alloc_size = 0;

#ifdef CUDA_SUPPORT
	if (type() == gpu::Context::TypeCUDA) {
		max_mem_alloc_size = data()->cuda_device_prop.totalGlobalMem / 2;
	} else
#endif
	if (type() == gpu::Context::TypeOpenCL) {
		max_mem_alloc_size = cl()->maxMemAllocSize();
	} else {
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	return max_mem_alloc_size;
}

size_t Context::getMaxWorkgroupSize()
{
	size_t max_workgroup_size = 0;

	switch (type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		max_workgroup_size = data()->cuda_device_prop.maxThreadsPerBlock;
		break;
#endif
	case Context::TypeOpenCL:
		max_workgroup_size = cl()->maxWorkgroupSize();
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	return max_workgroup_size;
}

std::vector<size_t> Context::getMaxWorkItemSizes()
{
	std::vector<size_t> work_item_sizes(3);

	switch (type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		for (int i = 0; i < 3; ++i) {
			work_item_sizes[i] = data()->cuda_device_prop.maxThreadsDim[i];
		}
		break;
#endif
	case Context::TypeOpenCL:
		for (int i = 0; i < 3; ++i) {
			work_item_sizes[i] = cl()->maxWorkItemSizes(i);
		}
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	return work_item_sizes;
}

Context::Type Context::type() const
{
	if (data_)
		return data_->type;
	if (data_ref_)
		return data_ref_->type;
	return TypeUndefined;
}

ocl::sh_ptr_ocl_engine Context::cl() const
{
	return data()->ocl_engine;
}

avk2::sh_ptr_vk_engine Context::vk() const
{
	return data()->vk_engine;
}

cudaStream_t Context::cudaStream() const
{
	return data()->cuda_stream;
}

device_memory_pool &Context::memoryPool() const
{
	return data()->memory_pool;
}

}
