#include "engine.h"

#define _SHORT_FILE_ "ocl_engine.cpp"

#include "utils.h"
#include <libbase/timer.h>
#include <libbase/thread_mutex.h>

#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>

#include <libclew/ocl_init.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/device.h>

#define LOAD_KERNEL_BINARIES_FROM_FILE ""
#define DUMP_KERNEL_BINARIES_TO_FILE ""
#define OCL_VERBOSE_COMPILE_LOG false

#ifdef _MSC_VER
typedef unsigned long long uint64_t;
#endif

using namespace ocl;

// Gets the platform ID for NVIDIA if available, otherwise default
cl_platform_id oclGetPlatformID(void)
{
	char chBuffer[1024];
	cl_uint num_platforms;

	cl_platform_id	platform = 0;

	// Get OpenCL platform count
	OCL_SAFE_CALL(clGetPlatformIDs (0, NULL, &num_platforms));

	if (num_platforms == 0)
		throw ocl_exception("No OpenCL platforms found");

	// if there's a platform or more, make space for ID's
	std::vector<cl_platform_id> clPlatformIDs(num_platforms);

	// get platform info for each platform and trap the NVIDIA platform if found
	OCL_SAFE_CALL(clGetPlatformIDs (num_platforms, clPlatformIDs.data(), NULL));
	for (cl_uint i = 0; i < num_platforms; ++i) {
		OCL_SAFE_CALL(clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL));
		if (strstr(chBuffer, "NVIDIA") != NULL) {
			platform = clPlatformIDs[i];
			break;
		}
	}

	// default to zeroeth platform if NVIDIA not found
	if (platform == 0)
		platform = clPlatformIDs[0];

	return platform;
}

OpenCLKernel::OpenCLKernel()
{
	kernel_				= 0;
	work_group_size_	= 0;
}

OpenCLKernel::~OpenCLKernel()
{
	if (kernel_)
		OCL_NOTHROW(OCL_SAFE_CALL(clReleaseKernel(kernel_)));
}

void OpenCLKernel::create(cl_program program, const char *kernel_name, cl_device_id device_id_)
{
	if (device_id_ == NULL) {
		gpu::Context context;
		GPU_CHECKED_VERBOSE(context.type() == gpu::Context::TypeOpenCL, "Can not link with OpenCL kernel");
		device_id_ = context.cl()->device();
	}

	cl_int ciErrNum = CL_SUCCESS;
	kernel_name_ = std::string(kernel_name);
	kernel_ = clCreateKernel(program, kernel_name, &ciErrNum);

	if (ciErrNum != CL_SUCCESS)
		throw std::runtime_error("clCreateKernel " + to_string(kernel_name_) + " failed: " + errorString(ciErrNum));

	size_t kernel_workgroup_size = 0;

	ciErrNum = clGetKernelWorkGroupInfo(kernel_, device_id_, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_workgroup_size, NULL);
	if (ciErrNum != CL_SUCCESS)
		throw std::runtime_error("clGetKernelWorkGroupInfo failed: " + errorString(ciErrNum));

	work_group_size_ = kernel_workgroup_size;
}

void OpenCLKernel::setArg(cl_uint arg_index, size_t arg_size, const void *arg_value)
{
	cl_int ciErrNum = clSetKernelArg(kernel_, arg_index, arg_size, arg_value);

	if (ciErrNum != CL_SUCCESS)
		throw std::runtime_error("clSetKernelArg " + to_string(kernel_name_) + "#" + to_string(arg_index) + " (" +to_string(arg_size) + " bytes) failed: " + errorString(ciErrNum));
}

OpenCLEngine::ContextMap	OpenCLEngine::contexts_;
std::mutex					OpenCLEngine::contexts_mutex_;

OpenCLEngine::OpenCLEngine()
{
	platform_id_					= 0;
	device_id_						= 0;
	context_						= 0;
	command_queue_					= 0;
	total_mem_size_					= 0;

	supported_image_formats_inited_	= false;
	print_supported_image_formats_	= false;
}

OpenCLEngine::~OpenCLEngine()
{
	for (std::map<int, OpenCLKernel *>::iterator it = kernels_.begin(); it != kernels_.end(); ++it)
		delete it->second;

	for (std::map<int, cl_program>::iterator it = programs_.begin(); it != programs_.end(); ++it)
		OCL_NOTHROW(OCL_SAFE_CALL(clReleaseProgram(it->second)));

	if (command_queue_)
		OCL_NOTHROW(OCL_SAFE_CALL(clReleaseCommandQueue(command_queue_)));

	OCL_NOTHROW(releaseContext());
}

void OpenCLEngine::createContext(cl_platform_id platform_id, cl_device_id device_id)
{
	std::lock_guard<std::mutex> lock(contexts_mutex_);

	cl_context context = 0;

	ContextMap::iterator it = contexts_.find(std::make_pair(platform_id, device_id));
	if (it != contexts_.end()) {
		it->second.count += 1;

		context = it->second.context;
	} else {
		cl_context_properties context_props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id, 0 };

		cl_int ciErrNum;
		OCL_TRACE(context = clCreateContext(context_props, 1, &device_id, NULL, NULL, &ciErrNum));
		OCL_SAFE_CALL(ciErrNum);

		bool opencl_share_context = true;
		if (opencl_share_context) {
			OpenCLContext ctx;
			ctx.platform_id	= platform_id;
			ctx.device_id	= device_id;
			ctx.context		= context;
			ctx.count		= 1;
			contexts_[std::make_pair(platform_id, device_id)] = ctx;
		}
	}

	context_		= context;
	platform_id_	= platform_id;
	device_id_		= device_id;
}

void OpenCLEngine::releaseContext()
{
	cl_context context = context_;
	if (!context)
		return;

	std::lock_guard<std::mutex> lock(contexts_mutex_);

	ContextMap::key_type context_key(platform_id_, device_id_);

	context_		= 0;
	platform_id_	= 0;
	device_id_		= 0;

	ContextMap::iterator it = contexts_.find(context_key);

	if (it != contexts_.end() && it->second.context == context) {
		if (it->second.count > 1) {
			it->second.count -= 1;
			return;
		} else {
			contexts_.erase(it);
		}
	}

	OCL_SAFE_CALL(clReleaseContext(context));
}

void OpenCLEngine::init(cl_device_id device_id, const char *cl_params, bool verbose)
{
	if (!device_id) {
		init((cl_platform_id) 0, (cl_device_id) 0, cl_params, verbose);
		return;
	}

	cl_platform_id platform_id = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_PLATFORM, sizeof(platform_id), &platform_id, NULL));

	return init(platform_id, device_id, cl_params, verbose);
}

namespace ocl {
	static std::map<std::pair<cl_platform_id, cl_device_id>, bool>	cached_spir_checks;
	static Mutex													cached_spir_checks_mutex;
	static Mutex													shrink_cache_mutex;
}

void OpenCLEngine::init(cl_platform_id platform_id, cl_device_id device_id, const char *cl_params, bool verbose)
{
	if (!ocl_init())
		throw ocl_exception("Can't init OpenCL driver");

	if (command_queue_) {
		OCL_SAFE_CALL(clReleaseCommandQueue(command_queue_));
		command_queue_ = 0;
	}

	releaseContext();

	if (!platform_id) {
		device_id	= 0;
		platform_id	= oclGetPlatformID();
	}

	if (!device_id) {
		// Get all the devices
		cl_uint			uiNumDevices	= 0;		// Number of devices available

		OCL_SAFE_CALL(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices));

		if (uiNumDevices < 1)
			throw ocl_exception("No OpenCL devices found");

		std::cout << uiNumDevices << " devices available" << std::endl;

		std::vector<cl_device_id> devices(uiNumDevices);
		OCL_SAFE_CALL(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, uiNumDevices, devices.data(), NULL));
		device_id = devices[0];
	}

	device_info_.init(device_id);

	if (device_info_.max_work_item_dimensions < 3)
		throw ocl_exception("3 dimensional work items not supported");

	total_mem_size_ = device_info_.global_mem_size;

	createContext(platform_id, device_id);

	cl_int ciErrNum;
	OCL_TRACE(command_queue_ = clCreateCommandQueue(context_, device_id, 0, &ciErrNum));
	OCL_SAFE_CALL(ciErrNum);

	if (device_info_.device_type == CL_DEVICE_TYPE_GPU) {
		if (device_info_.warp_size) {
			wavefront_size_ = device_info_.warp_size;
		} else if (device_info_.wavefront_width) {
			wavefront_size_ = device_info_.wavefront_width;
		} else if (device_info_.isIntelGPU()) {
			wavefront_size_ = 8;
		} else {
			wavefront_size_ = 1;
		}
	} else {
		wavefront_size_ = 1;
	}

	if (verbose) {
		device_info_.print();
		if (device_info_.warp_size == 0 && device_info_.wavefront_width == 0) {
			std::cout << "  wavefront width " << wavefront_size_ << std::endl;
		}
	}
}

void ocl::oclPrintBuildLog(cl_program program)
{
	size_t device_count;
	OCL_SAFE_CALL(clGetProgramInfo(program, CL_PROGRAM_DEVICES, 0, NULL, &device_count));
	device_count /= sizeof(cl_device_id);

	std::vector<cl_device_id> devices(device_count);

	OCL_SAFE_CALL(clGetProgramInfo(program, CL_PROGRAM_DEVICES, device_count * sizeof(cl_device_id), devices.data(), NULL));

	for (size_t k = 0; k < device_count; k++) {
		std::cout << "Device " << k + 1 << std::endl;
		size_t log_size = 0;
		OCL_SAFE_CALL(clGetProgramBuildInfo(program, devices[k], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
		if (log_size > 0) {
			std::cout << "\tProgram build log:" << std::endl;
			std::vector<char> log(log_size + 1);
			OCL_SAFE_CALL(clGetProgramBuildInfo(program, devices[k], CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL));

			log[log_size] = 0;
			std::cout << log.data() << std::endl << std::endl;
		} else {
			std::cout << "\tProgram build log is clear" << std::endl;
		}
	}
}

void OpenCLEngine::initSupportedImageFormatsList()
{
	if (supported_image_formats_inited_)
		return;
	if (!device_info_.image_support)
		throw ocl_bad_alloc("Can't fetch supported image formats list, because of a lack of CL_DEVICE_IMAGE_SUPPORT");

	cl_uint num_image_formats = 0;
	OCL_SAFE_CALL(clGetSupportedImageFormats(context_, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 0, NULL, &num_image_formats));
	std::vector<cl_image_format> image_formats(num_image_formats);
	OCL_SAFE_CALL(clGetSupportedImageFormats(context_, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, num_image_formats, image_formats.data(), NULL));

	supported_image_formats_ = image_formats;
	supported_image_formats_inited_ = true;
}

cl_mem OpenCLEngine::createImage2D(size_t width, size_t height, size_t cn, DataType data_type)
{
	if (!device_info_.image_support) {
		throw ocl_bad_alloc("Can't allocate 2D image, because of a lack of CL_DEVICE_IMAGE_SUPPORT");
	}
	if (width > device_info_.image2d_max_width || height > device_info_.image2d_max_height) {
		throw ocl_bad_alloc("Can't allocate " + to_string(width) + "x" + to_string(height) + " 2D image, because max image size on this device: " + to_string(device_info_.image2d_max_width) + "x" + to_string(device_info_.image2d_max_height));
	}
	if (!supported_image_formats_inited_) {
		initSupportedImageFormatsList();
	}

	cl_mem_flags flags = CL_MEM_READ_WRITE;
	cl_image_format format;

	{
		// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/cl_image_format.html
		if        (cn == 1) {
			format.image_channel_order = CL_R;
		} else if (cn == 2) {
			format.image_channel_order = CL_RG;
		} else if (cn == 3) {
			format.image_channel_order = CL_RGB;
		} else if (cn == 4) {
			format.image_channel_order = CL_RGBA;
		}
		if        (data_type == DataType8u) {
			format.image_channel_data_type = CL_UNORM_INT8;
		} else if (data_type == DataType16u) {
			format.image_channel_data_type = CL_UNORM_INT16;
		} else if (data_type == DataType32u) {
			throw std::runtime_error("OpenCLEngine::createImage2D(): DataType32u not supported!"); // because it seems that it can't be used with bilinear interpolation
		} else if (data_type == DataType8i) {
			format.image_channel_data_type = CL_SNORM_INT8;
		} else if (data_type == DataType16i) {
			format.image_channel_data_type = CL_SNORM_INT16;
		} else if (data_type == DataType32i) {
			throw std::runtime_error("OpenCLEngine::createImage2D(): DataType32i not supported!"); // because it seems that it can't be used with bilinear interpolation
		} else if (data_type == DataType32f) {
			format.image_channel_data_type = CL_FLOAT;
		} else {
			throw std::runtime_error("Unsupported data type: " + typeName(data_type));
		}
	}

#if 1
	cl_image_desc image_desc;
	{
		// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/cl_image_desc.html
		image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
		image_desc.image_width = width;
		image_desc.image_height = height;
		image_desc.image_depth = 0;
		image_desc.image_array_size = 0;
		image_desc.image_row_pitch = 0;
		image_desc.image_slice_pitch = 0;
		image_desc.num_mip_levels = 0;
		image_desc.num_samples = 0;
		image_desc.buffer = NULL;
	}

	cl_int status = CL_SUCCESS;
	cl_mem res;
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateImage.html
	OCL_TRACE(res = clCreateImage(context_, flags, &format, &image_desc, NULL, &status)); // requires OpenCL 1.2
	OCL_SAFE_CALL_MESSAGE(status, "Image type " + to_string(cn) + "x" + typeName(data_type) + ": ");
#else
	cl_int status = CL_SUCCESS;
	cl_mem res;
	// https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/clCreateImage2D.html
	OCL_TRACE(res = clCreateImage2D(context_, flags, &format, width, height, 0, NULL, &status)); // deprecated but works
	OCL_SAFE_CALL(status);
#endif

	return res;
}

cl_sampler OpenCLEngine::createImageSampler()
{
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateSampler.html
	cl_bool normalized_coords = CL_FALSE;

	cl_int status = CL_SUCCESS;
	cl_sampler res;
	OCL_TRACE(res = clCreateSampler(context_, normalized_coords, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR, &status));
	OCL_SAFE_CALL(status);

	return res;
}

cl_mem OpenCLEngine::createBuffer(cl_mem_flags flags, size_t size)
{
	if (size > device_info_.max_mem_alloc_size) {
		throw ocl_bad_alloc("Can't allocate " + to_string(size) + " bytes, because max allocation size is " + to_string(device_info_.max_mem_alloc_size));
	}

	// avoiding CL_INVALID_BUFFER_SIZE error when trying to allocate buffer with zero size
	// in case of zero requested size we provide some correct pointer which can be safely passed as an argument to kernels
	size = std::max<size_t>(size, 1);

	cl_int status = CL_SUCCESS;
	cl_mem res;
	OCL_TRACE(res = clCreateBuffer(context_, flags, size, NULL, &status));
	OCL_SAFE_CALL(status);

	// forcing buffer allocation by fictive write
	size_t data_size = (size >= 8) ? 4 : 1;
	assert (size >= 2 * data_size);

	int test_data = 239;
	try {
		writeBuffer(res, CL_TRUE, 0, data_size, &test_data);
	} catch (ocl_exception &) {
		releaseMemObject(res);
		throw;
	}

	return res;
}

void OpenCLEngine::writeBuffer(cl_mem buffer, cl_bool blocking_write, size_t offset, size_t cb, const void *ptr)
{
	if (cb == 0)
		return;

	OCL_SAFE_CALL(clEnqueueWriteBuffer(queue(), buffer, blocking_write, offset, cb, ptr, 0, NULL, NULL));
}

void OpenCLEngine::writeBufferRect(cl_mem buffer, cl_bool blocking_write, const size_t buffer_origin[3], const size_t host_origin[3], const size_t region[3],
								size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, const void *ptr)
{
	if (region[0] == 0 || region[1] == 0 || region[2] == 0)
		return;

	OCL_SAFE_CALL(clEnqueueWriteBufferRect(queue(), buffer, blocking_write, buffer_origin, host_origin, region,
								buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, 0, NULL, NULL));
}

void OpenCLEngine::readBuffer(cl_mem buffer, cl_bool blocking_read, size_t offset, size_t cb, void *ptr)
{
	if (cb == 0)
		return;

	OCL_SAFE_CALL(clEnqueueReadBuffer(queue(), buffer, blocking_read, offset, cb, ptr, 0, NULL, NULL));
}

void OpenCLEngine::readBufferRect(cl_mem buffer, cl_bool blocking_write, const size_t buffer_origin[3], const size_t host_origin[3], const size_t region[3],
								size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, void *ptr)
{
	if (region[0] == 0 || region[1] == 0 || region[2] == 0)
		return;

	OCL_SAFE_CALL(clEnqueueReadBufferRect(queue(), buffer, blocking_write, buffer_origin, host_origin, region,
								buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, 0, NULL, NULL));
}

void OpenCLEngine::copyBuffer(cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t cb)
{
	if (cb == 0)
		return;

	cl_event ev = NULL;
	OCL_SAFE_CALL(clEnqueueCopyBuffer(queue(), src_buffer, dst_buffer, src_offset, dst_offset, cb, 0, NULL, &ev));
	trackEvent(ev);
}

void OpenCLEngine::copyBufferToImage(cl_mem src_buffer, cl_mem dst_image,
		size_t width, size_t height,
		size_t src_offset,
		size_t dst_x_offset, size_t dst_y_offset,
		bool async)
{
	size_t dst_origin[3] = {dst_x_offset, dst_y_offset, 0};
	size_t region[3] = {width, height, 1};
	cl_event ev = NULL;
	OCL_SAFE_CALL(clEnqueueCopyBufferToImage(queue(), src_buffer, dst_image, src_offset, dst_origin, region, 0, NULL, async ? NULL : &ev));
	if (!async) {
		trackEvent(ev);
	}
}

void OpenCLEngine::readImage(cl_mem src_image, size_t width, size_t height, size_t row_pitch, void *ptr)
{
	if (width == 0 || height == 0)
		return;

	cl_bool blocking_read = CL_TRUE;
	const size_t origin[3] = {0, 0, 0};
	const size_t region[3] = {width, height, 1};
	const size_t slice_pitch = 0;
	cl_event ev = NULL;
	OCL_SAFE_CALL(clEnqueueReadImage(queue(), src_image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, 0, NULL, &ev));
	trackEvent(ev);
}

void OpenCLEngine::releaseMemObject(cl_mem memobj)
{
	if (memobj == NULL)
		return;

	OCL_SAFE_CALL(clReleaseMemObject(memobj));
}

void OpenCLEngine::ndRangeKernel(OpenCLKernel &kernel, cl_uint work_dim, const size_t *global_work_offset,
								 const size_t *global_work_size, const size_t *local_work_size, bool synchronized)
{
	if (work_dim < 1 || work_dim > 3)
		throw ocl_exception("Wrong work dimension size: " + to_string(work_dim));

	// check workgroup size
	if (local_work_size) {
		size_t workgroup_size = 1;
		for (cl_uint dim = 0; dim < work_dim; dim++) {
			if (local_work_size[dim] > device_info_.max_work_item_sizes[dim])
				throw ocl_exception("Wrong work_size[" + to_string(dim) + "] value: " + to_string(local_work_size[dim]));
			workgroup_size *= local_work_size[dim];
		}
		if (workgroup_size > device_info_.max_workgroup_size)
			throw ocl_exception("Too big workgroup size: " + to_string(workgroup_size));
		if (workgroup_size > kernel.workGroupSize())
			throw ocl_exception("Too big workgroup size for this kernel: " + to_string(workgroup_size));
	}

	// If, for example, CL_DEVICE_ADDRESS_BITS = 32, i.e. the device uses a 32-bit address space,
	// size_t is a 32-bit unsigned integer and global_work_size values must be in the range 1 .. 2^32 - 1.
	// Values outside this range return a CL_OUT_OF_RESOURCES error.
	uint64_t max_global_work_size = (uint64_t) 1 << (device_info_.device_address_bits - 1);
	max_global_work_size = max_global_work_size + (max_global_work_size - 1);
	for (size_t d = 0; d < work_dim; ++d) {
		if (global_work_size[d] == 0) {
			std::cerr << "Global work size is zero" << std::endl;
			throw ocl_exception("Global work_size[" + to_string(d) + "] value is zero");
		} else if (global_work_size[d] > max_global_work_size && device_info_.device_address_bits <= 64) {
			throw ocl_exception("Global work_size[" + to_string(d) + "] value is too big for this device address bits: "
								+ to_string(global_work_size[d]) + ", while device has " + to_string(device_info_.device_address_bits) + " address bits");
		}
	}

	cl_event ev = NULL;
	OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue(), kernel.kernel(), work_dim, global_work_offset, global_work_size, local_work_size, 0, NULL, &ev));
	if (synchronized)
		trackEvent(ev, "Kernel " + kernel.kernelName() + ": ");
	else
		OCL_SAFE_CALL_MESSAGE(clReleaseEvent(ev), "Kernel " + kernel.kernelName() + ": ");
}

void OpenCLEngine::trackEvent(cl_event ev, std::string message)
{
	cl_int result = CL_SUCCESS;

	try {
		OCL_SAFE_CALL_MESSAGE(clFlush(queue()), message);
		OCL_SAFE_CALL_MESSAGE(clWaitForEvents(1, &ev), message);
		OCL_SAFE_CALL_MESSAGE(clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &result, 0), message);

		if (result != CL_COMPLETE) {
			throw ocl_exception("Wait for event succeed, but it is still is not complete with execution status: " + to_string(result));
		}
	} catch (...) {
		OCL_SAFE_CALL_MESSAGE(clReleaseEvent(ev), message);
		throw;
	}

	OCL_SAFE_CALL_MESSAGE(clReleaseEvent(ev), message);
}

cl_program OpenCLEngine::findProgram(int id) const
{
	std::map<int, cl_program>::const_iterator it = programs_.find(id);
	if (it != programs_.end())
		return it->second;
	return 0;
}

OpenCLKernel *OpenCLEngine::findKernel(int id) const
{
	std::map<int, OpenCLKernel *>::const_iterator it = kernels_.find(id);
	if (it != kernels_.end())
		return it->second;
	return nullptr;
}

uint64_t OpenCLEngine::freeMem()
{
	return device_info_.freeMemory();
}

std::string OpenCLEngine::getProgramBinaryShortId(const std::string &program_name) const
{
	return program_name + std::to_string((long long)deviceInfo().vendor_id) + deviceInfo().clean_device_name;
}

VersionedBinary::VersionedBinary(const char *data, const size_t size,
								 int bits, const int opencl_major_version, const int opencl_minor_version)
		: data_(data), size_(size), device_address_bits_(bits), opencl_major_version_(opencl_major_version), opencl_minor_version_(opencl_minor_version)
{}

ProgramBinaries::ProgramBinaries(std::vector<const VersionedBinary *> binaries, std::string program_name, std::string defines) : binaries_(binaries)
{
	static int next_program_id = 0;
	program_name_ = program_name;
	id_			= next_program_id++;
	defines_	= defines;
}

ProgramBinaries::ProgramBinaries(const char *source_code, size_t source_code_length, std::string program_name, std::string defines) : binary_with_lifetime_(std::make_shared<VersionedBinary>(VersionedBinary(source_code, source_code_length, 0, 1, 2))), binaries_({binary_with_lifetime_.get()})
{
	static int next_program_id = 0;
	program_name_ = program_name;
	id_			= next_program_id++;
	defines_	= defines;
}

const VersionedBinary* ProgramBinaries::getBinary(const std::shared_ptr<OpenCLEngine> &cl) const
{
	return getBinary(cl.get());
}

const VersionedBinary* ProgramBinaries::getBinary(OpenCLEngine *cl) const
{
	for (int i = 0; i < binaries_.size(); ++i) {
		const VersionedBinary* binary = binaries_[i];

		if (binary->deviceAddressBits() && binary->deviceAddressBits() != cl->deviceAddressBits())
			continue;

		if (binary->openclMajorVersion() > cl->deviceInfo().opencl_major_version)
			continue;

		if (binary->openclMajorVersion() == cl->deviceInfo().opencl_major_version && binary->openclMinorVersion() > cl->deviceInfo().opencl_minor_version)
			continue;

		return binary;
	}

	throw ocl_exception("No OpenCL/SPIR version for " + to_string(cl->deviceAddressBits()) + "-bit device with OpenCL "
						+ to_string(cl->deviceInfo().opencl_major_version) + "." + to_string(cl->deviceInfo().opencl_minor_version));
}

KernelSource::KernelSource(const ocl::ProgramBinaries& program) : program_(program)
{
	id_		= getNextKernelId();
	name_	= std::string(program.programName());
}

KernelSource::KernelSource(const ocl::ProgramBinaries& program, const char *name) : program_(program)
{
	id_		= getNextKernelId();
	name_	= std::string(name);
}

KernelSource::KernelSource(const ocl::ProgramBinaries& program, const std::string &name) : program_(program)
{
	id_		= getNextKernelId();
	name_	= std::string(name);
}

KernelSource::KernelSource(std::shared_ptr<ocl::ProgramBinaries> program, const char *name) : program_lifetime_(program), program_(*program)
{
	id_		= getNextKernelId();
	name_	= std::string(name);
}

KernelSource::KernelSource(std::shared_ptr<ocl::ProgramBinaries> program, const std::string &name) : program_lifetime_(program), program_(*program)
{
	id_		= getNextKernelId();
	name_	= name;
}

int KernelSource::getNextKernelId()
{
	static int next_kernel_id = 0;
	return next_kernel_id++;
}

namespace ocl {
	typedef std::map<std::pair<cl_platform_id, cl_device_id>, std::vector<unsigned char>> binaries_by_device;
	static std::map<int, binaries_by_device>	cached_kernels_binaries;
	static Mutex								cached_kernels_mutex;

	std::vector<unsigned char>* getCachedBinary(int programId, cl_platform_id platform, cl_device_id device)
	{
		auto programCacheIt = cached_kernels_binaries.find(programId);
		if (programCacheIt == cached_kernels_binaries.end())
			cached_kernels_binaries[programId] = binaries_by_device();
		auto binaryIt = cached_kernels_binaries[programId].find(std::make_pair(platform, device));
		if (binaryIt != cached_kernels_binaries[programId].end()) {
			return &binaryIt->second;
		} else {
			return NULL;
		}
	}

	void setCachedBinary(int programId, cl_platform_id platform, cl_device_id device, std::vector<unsigned char> binaries)
	{
		auto programCacheIt = cached_kernels_binaries.find(programId);
		if (programCacheIt == cached_kernels_binaries.end())
			cached_kernels_binaries[programId] = binaries_by_device();
		cached_kernels_binaries[programId][std::make_pair(platform, device)] = binaries;
	}

	std::vector<unsigned char> getProgramBinaries(cl_program program)
	{
		size_t binaries_size;
		OCL_SAFE_CALL(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
									   sizeof(size_t), &binaries_size, NULL));

		std::vector<unsigned char> binaries(binaries_size);
		unsigned char *data = binaries.data();
		OCL_SAFE_CALL(clGetProgramInfo(program, CL_PROGRAM_BINARIES,
									   sizeof(unsigned char *), &data, NULL));

		return binaries;
	}
}

OpenCLKernel *KernelSource::getKernel(const std::shared_ptr<OpenCLEngine> &cl, bool printLog)
{
	OpenCLKernel *kernel = cl->findKernel(id_);
	if (kernel)
		return kernel;

	cl_program program = cl->findProgram(program_.id());

	if (!program) {
		Lock lock(cached_kernels_mutex);

		bool verbose = printLog || OCL_VERBOSE_COMPILE_LOG;

		const VersionedBinary* binary = program_.getBinary(cl);
		const std::vector<unsigned char>* cachedCompiledBinary = getCachedBinary(program_.id(), cl->platform(), cl->device());

		cl_int ciErrNum = CL_SUCCESS;

		std::string options = program_.defines();

		std::vector<unsigned char> loaded_from_file_binaries;
		std::string binaries_to_load_filename = LOAD_KERNEL_BINARIES_FROM_FILE;

		if (!binaries_to_load_filename.empty()){
			std::ifstream program_binaries_file;
			program_binaries_file.open(binaries_to_load_filename);

			std::string binaries_string((std::istreambuf_iterator<char>(program_binaries_file)), std::istreambuf_iterator<char>());

			loaded_from_file_binaries = std::vector<unsigned char>(binaries_string.size());
			for (int i = 0; i < binaries_string.size(); ++i) {
				loaded_from_file_binaries[i] = (unsigned char) binaries_string[i];
			}
			cachedCompiledBinary = &loaded_from_file_binaries;

			program_binaries_file.close();
		}

		if (cachedCompiledBinary != NULL) {
			std::vector<const unsigned char *>	kernel_ptrs;
			std::vector<size_t>					kernel_sizes;

			kernel_ptrs.push_back(cachedCompiledBinary->data());
			kernel_sizes.push_back(cachedCompiledBinary->size());

			cl_device_id device = cl->device();
			cl_int binary_status;

			program = clCreateProgramWithBinary(cl->context(), 1, &device, &kernel_sizes[0], &kernel_ptrs[0], &binary_status, &ciErrNum);
			OCL_SAFE_CALL(binary_status);
			OCL_SAFE_CALL(ciErrNum);
		} else if (binary->deviceAddressBits() == 0) {
			std::vector<const char *>			kernel_ptrs;
			std::vector<size_t>					kernel_sizes;

			kernel_ptrs.push_back(binary->data());
			kernel_sizes.push_back(binary->size());

			program = clCreateProgramWithSource(cl->context(), kernel_ptrs.size(), &kernel_ptrs[0], &kernel_sizes[0], &ciErrNum);
			OCL_SAFE_CALL(ciErrNum);
		} else {
			std::vector<const unsigned char *>	kernel_ptrs;
			std::vector<size_t>					kernel_sizes;

			kernel_ptrs.push_back((unsigned char*) binary->data());
			kernel_sizes.push_back(binary->size());

			cl_device_id device = cl->device();
			cl_int binary_status;

			program = clCreateProgramWithBinary(cl->context(), 1, &device, &kernel_sizes[0], &kernel_ptrs[0], &binary_status, &ciErrNum);
			OCL_SAFE_CALL(binary_status);
			OCL_SAFE_CALL(ciErrNum);

			if (cl->deviceInfo().extensions.count("cl_khr_spir") == 0)
				throw ocl_exception("Device does not support SPIR!");

			options += " -x spir";
		}

		options += " -D WARP_SIZE=" + to_string(cl->wavefrontSize());

		timer tm;
		tm.start();

		if (cachedCompiledBinary == NULL && verbose) {
			if (program_.programName() == "") {
				std::cout << "Building kernels for " << cl->deviceName() << "... " << std::endl;
			}
//			else {
//				std::cout << "Building kernel " << program_.programName() << " for " << cl->deviceName() << "... " << std::endl;
//			}
		}

		OCL_TRACE(ciErrNum = clBuildProgram(program, 0, NULL, options.c_str(), NULL, NULL));

		if (ciErrNum == CL_SUCCESS) {
			if (cachedCompiledBinary == NULL) {
				std::cout << "Kernels compilation done in " << tm.elapsed() << " seconds" << std::endl;
			} else {
				std::cout << "Kernel loaded in " << tm.elapsed() << " seconds" << std::endl;
			}

			std::vector<unsigned char> binaries = getProgramBinaries(program);
			setCachedBinary(program_.id(), cl->platform(), cl->device(), binaries);
		}

		if (ciErrNum != CL_SUCCESS || verbose) {
			ocl::oclPrintBuildLog(program);

			std::string binaries_filename = DUMP_KERNEL_BINARIES_TO_FILE;
			if (!binaries_filename.empty()) {
				std::vector<unsigned char> binaries = getProgramBinaries(program);
				std::string binaries_string((char*) binaries.data(), binaries.size());

				std::ofstream program_binaries_file;
				program_binaries_file.open(binaries_filename + "_platform" + to_string(cl->platform()) + "_device" + to_string(cl->device()) + "_program" + to_string(program_.id()));

				program_binaries_file << binaries_string;
				program_binaries_file.close();
			}
		}

		if (ciErrNum != CL_SUCCESS) {
			OCL_SAFE_CALL(clReleaseProgram(program));
			program = 0;
		}

		OCL_SAFE_CALL(ciErrNum);
		cl->programs()[program_.id()] = program;
	}

	kernel = new OpenCLKernel;
	kernel->create(program, name_.c_str());

	cl->kernels()[id_] = kernel;

	return kernel;
}

size_t KernelSource::getMaxWorkgroupSize()
{
	gpu::Context context;

	OpenCLKernel *kernel = getKernel(context.cl());

	return kernel->workGroupSize();
}

void KernelSource::exec(const gpu::WorkSize &ws, const Arg &arg0, const Arg &arg1, const Arg &arg2, const Arg &arg3, const Arg &arg4, const Arg &arg5, const Arg &arg6, const Arg &arg7, const Arg &arg8, const Arg &arg9, const Arg &arg10, const Arg &arg11, const Arg &arg12, const Arg &arg13, const Arg &arg14, const Arg &arg15, const Arg &arg16, const Arg &arg17, const Arg &arg18, const Arg &arg19, const Arg &arg20, const Arg &arg21, const Arg &arg22, const Arg &arg23, const Arg &arg24, const Arg &arg25, const Arg &arg26, const Arg &arg27, const Arg &arg28, const Arg &arg29, const Arg &arg30, const Arg &arg31, const Arg &arg32, const Arg &arg33, const Arg &arg34, const Arg &arg35, const Arg &arg36, const Arg &arg37, const Arg &arg38, const Arg &arg39, const Arg &arg40)
{
	gpu::Context context;

	OpenCLKernel *kernel = getKernel(context.cl());

	kernel->setArgs(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40);

	context.cl()->ndRangeKernel(*kernel, 3, NULL, ws.clGlobalSize(), ws.clLocalSize());
}

void KernelSource::execAsynchronized(const gpu::WorkSize &ws, const Arg &arg0, const Arg &arg1, const Arg &arg2, const Arg &arg3, const Arg &arg4, const Arg &arg5, const Arg &arg6, const Arg &arg7, const Arg &arg8, const Arg &arg9, const Arg &arg10, const Arg &arg11, const Arg &arg12, const Arg &arg13, const Arg &arg14, const Arg &arg15, const Arg &arg16, const Arg &arg17, const Arg &arg18, const Arg &arg19, const Arg &arg20, const Arg &arg21, const Arg &arg22, const Arg &arg23, const Arg &arg24, const Arg &arg25, const Arg &arg26, const Arg &arg27, const Arg &arg28, const Arg &arg29, const Arg &arg30, const Arg &arg31, const Arg &arg32, const Arg &arg33, const Arg &arg34, const Arg &arg35, const Arg &arg36, const Arg &arg37, const Arg &arg38, const Arg &arg39, const Arg &arg40)
{
	gpu::Context context;

	OpenCLKernel *kernel = getKernel(context.cl());

	kernel->setArgs(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40);

	context.cl()->ndRangeKernel(*kernel, 3, NULL, ws.clGlobalSize(), ws.clLocalSize(), false);
}

void KernelSource::execSubdivided(const gpu::WorkSize &ws, const Arg &arg0, const Arg &arg1, const Arg &arg2, const Arg &arg3, const Arg &arg4, const Arg &arg5, const Arg &arg6, const Arg &arg7, const Arg &arg8, const Arg &arg9, const Arg &arg10, const Arg &arg11, const Arg &arg12, const Arg &arg13, const Arg &arg14, const Arg &arg15, const Arg &arg16, const Arg &arg17, const Arg &arg18, const Arg &arg19, const Arg &arg20, const Arg &arg21, const Arg &arg22, const Arg &arg23, const Arg &arg24, const Arg &arg25, const Arg &arg26, const Arg &arg27, const Arg &arg28, const Arg &arg29, const Arg &arg30, const Arg &arg31, const Arg &arg32, const Arg &arg33, const Arg &arg34, const Arg &arg35, const Arg &arg36, const Arg &arg37, const Arg &arg38, const Arg &arg39, const Arg &arg40)
{
	const size_t max_total_size = 1000000;

	const size_t local_x = ws.clLocalSize()[0];
	const size_t local_y = ws.clLocalSize()[1];
	const size_t local_z = ws.clLocalSize()[2];

	const size_t total_x = ws.clGlobalSize()[0];
	const size_t total_y = ws.clGlobalSize()[1];
	const size_t total_z = ws.clGlobalSize()[2];

	size_t nparts_x = 1;
	size_t nparts_y = 1;
	size_t nparts_z = 1;

	size_t part_x = total_x;
	size_t part_y = total_y;
	size_t part_z = total_z;

	while (part_x * part_y * part_z > max_total_size && part_x > local_x) {
		nparts_x *= 2;
		part_x = local_x * gpu::divup(gpu::divup(total_x, nparts_x), local_x);
	}
	while (part_x * part_y * part_z > max_total_size && part_y > local_y) {
		nparts_y *= 2;
		part_y = local_y * gpu::divup(gpu::divup(total_y, nparts_y), local_y);
	}
	while (part_x * part_y * part_z > max_total_size && part_z > local_z) {
		nparts_z *= 2;
		part_z = local_z * gpu::divup(gpu::divup(total_z, nparts_z), local_z);
	}

	gpu::Context context;

	OpenCLKernel *kernel = getKernel(context.cl());

	kernel->setArgs(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40);

	for (size_t offset_x = 0; offset_x < total_x; offset_x += part_x) {
		for (size_t offset_y = 0; offset_y < total_y; offset_y += part_y) {
			for (size_t offset_z = 0; offset_z < total_z; offset_z += part_z) {
				size_t offset[3];
				offset[0] = offset_x;
				offset[1] = offset_y;
				offset[2] = offset_z;

				size_t current_x = std::min(part_x, total_x - offset_x);
				size_t current_y = std::min(part_y, total_y - offset_y);
				size_t current_z = std::min(part_z, total_z - offset_z);

				gpu::WorkSize ws_part(local_x, local_y, local_z, current_x, current_y, current_z);

				// TODO: generalize this logic, apply it to CUDA, make so that ndRangeKernel is called only in one place in codebase, remove all get_group_id/get_num_groups calls, etc.
				context.cl()->ndRangeKernel(*kernel, 3, offset, ws_part.clGlobalSize(), ws_part.clLocalSize());
			}
		}
	}
}

void KernelSource::precompile(bool printLog) {
	gpu::Context context;

	precompile(context.cl(), printLog);
}

void KernelSource::precompile(const std::shared_ptr<OpenCLEngine> &cl, bool printLog) {
	getKernel(cl, printLog);
}

OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer &arg)
{
	is_null = false;
	size = sizeof(cl_mem);
	cl_mem_storage = arg.clmem();
	value = &cl_mem_storage;
	if (arg.cloffset() != 0) {
		ocl_exception("Offset is not zero, but ignored");
	}
}

template<typename T>
OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer_typed<T> &arg)
{
	is_null = false;
	size = sizeof(cl_mem);
	cl_mem_storage = arg.clmem();
	value = &cl_mem_storage;
	if (arg.cloffset() != 0) {
		ocl_exception("Offset is not zero, but ignored");
	}
}

template OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer_typed<char> &arg);
template OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer_typed<unsigned char> &arg);
template OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer_typed<short> &arg);
template OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer_typed<unsigned short> &arg);
template OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer_typed<int> &arg);
template OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer_typed<unsigned int> &arg);
template OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer_typed<float> &arg);
template OpenCLKernelArg::OpenCLKernelArg(const gpu::shared_device_buffer_typed<double> &arg);
