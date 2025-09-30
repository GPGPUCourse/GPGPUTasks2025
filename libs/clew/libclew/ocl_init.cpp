#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>

#ifdef _WIN32

#include <windows.h>

typedef HMODULE OclLibrary;

HMODULE oclLoadLibrary(void)
{
	return LoadLibraryW(L"OpenCL.dll");
}

FARPROC oclGetProcAddress(HMODULE hModule, LPCSTR lpProcName)
{
	return ::GetProcAddress(hModule, lpProcName);
}

#elif defined(__unix__) || defined(__APPLE__) || defined(__MACOSX)

#include <dlfcn.h>

typedef void * OclLibrary;

OclLibrary oclLoadLibrary(void)
{
#if defined(__APPLE__) || defined(__MACOSX)
	return dlopen("/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL", RTLD_NOW);
#else
	OclLibrary lib = dlopen("libOpenCL.so", RTLD_NOW);
	if (!lib) {
		lib = dlopen("libOpenCL.so.1", RTLD_NOW);
	}
	return lib;
#endif
}

void *oclGetProcAddress(void *handle, const char *symbol)
{
	return dlsym(handle, symbol);
}

#else
#error unsupported platform
#endif

// Platform API

typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetPlatformIDs)				(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetPlatformInfo)			(cl_platform_id, cl_platform_info, size_t, void *, size_t *);

// Device APIs

typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetDeviceIDs)				(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetDeviceInfo)				(cl_device_id, cl_device_info, size_t, void *, size_t *);

// Context APIs  

typedef cl_context			(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateContext)				(const cl_context_properties *, cl_uint, const cl_device_id *, void (CL_CALLBACK *)(const char *, const void *, size_t, void *), void *, cl_int *);
typedef cl_context			(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateContextFromType)		(const cl_context_properties *, cl_device_type, void (CL_CALLBACK *)(const char *, const void *, size_t, void *), void *, cl_int *);

typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clRetainContext)				(cl_context);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clReleaseContext)				(cl_context);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetContextInfo)				(cl_context, cl_context_info, size_t, void *, size_t *);

// Command Queue APIs

typedef cl_command_queue	(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateCommandQueue)			(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clRetainCommandQueue)			(cl_command_queue);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clReleaseCommandQueue)		(cl_command_queue);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetCommandQueueInfo)		(cl_command_queue, cl_command_queue_info, size_t, void *, size_t *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clSetCommandQueueProperty)	(cl_command_queue, cl_command_queue_properties, cl_bool, cl_command_queue_properties *);

// Memory Object APIs

typedef cl_mem				(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateBuffer)				(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_mem				(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateImage)				(cl_context, cl_mem_flags, const cl_image_format *, const cl_image_desc *, void *, cl_int *);
typedef cl_mem				(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateImage2D)				(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, void *, cl_int *);
typedef cl_mem				(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateImage3D)				(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, size_t, size_t, void *, cl_int *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clRetainMemObject)			(cl_mem);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clReleaseMemObject)			(cl_mem);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetSupportedImageFormats)	(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format *, cl_uint *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetMemObjectInfo)			(cl_mem, cl_mem_info, size_t, void *, size_t *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetImageInfo)				(cl_mem, cl_image_info, size_t, void *, size_t *);

// Sampler APIs

typedef cl_sampler			(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateSampler)				(cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, cl_int *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clRetainSampler)				(cl_sampler);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clReleaseSampler)				(cl_sampler);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetSamplerInfo)				(cl_sampler, cl_sampler_info, size_t, void *, size_t *);

// Program Object APIs

typedef cl_program			(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateProgramWithSource)	(cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_program			(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateProgramWithBinary)	(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *);
typedef cl_program			(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateProgramWithIL)		(cl_context, const void *, size_t, cl_int *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clRetainProgram)				(cl_program);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clReleaseProgram)				(cl_program);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clBuildProgram)				(cl_program, cl_uint, const cl_device_id *, const char *, void (CL_CALLBACK *)(cl_program, void *), void *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clUnloadCompiler)				(void);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetProgramInfo)				(cl_program, cl_program_info, size_t, void *, size_t *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetProgramBuildInfo)		(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);

// Kernel Object APIs

typedef cl_kernel			(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateKernel)				(cl_program, const char *, cl_int *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clCreateKernelsInProgram)		(cl_program, cl_uint, cl_kernel *, cl_uint *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clRetainKernel)				(cl_kernel);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clReleaseKernel)				(cl_kernel);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clSetKernelArg)				(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetKernelInfo)				(cl_kernel, cl_kernel_info, size_t, void *, size_t *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetKernelWorkGroupInfo)		(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);

// Event Object APIs

typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clWaitForEvents)				(cl_uint, const cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetEventInfo)				(cl_event, cl_event_info, size_t, void *, size_t *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clRetainEvent)				(cl_event);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clReleaseEvent)				(cl_event);

// Profiling APIs

typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetEventProfilingInfo)		(cl_event, cl_profiling_info, size_t, void *, size_t *);

// Flush and Finish APIs

typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clFlush)						(cl_command_queue);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clFinish)						(cl_command_queue);

// Enqueued Commands APIs

typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueReadBuffer)			(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueReadBufferRect)		(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueWriteBuffer)			(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueWriteBufferRect)		(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueCopyBuffer)			(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueReadImage)			(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueWriteImage)			(cl_command_queue, cl_mem, cl_bool, const size_t * [], const size_t * [], size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueCopyImage)			(cl_command_queue, cl_mem, cl_mem, const size_t * [], const size_t * [], const size_t * [], cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueCopyImageToBuffer)	(cl_command_queue, cl_mem, cl_mem, const size_t * [], const size_t * [], size_t, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueCopyBufferToImage)	(cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef void *				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueMapBuffer)			(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event *, cl_event *, cl_int *);
typedef void *				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueMapImage)			(cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t *, const size_t *, size_t *, size_t *, cl_uint, const cl_event *, cl_event *, cl_int *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueUnmapMemObject)		(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueNDRangeKernel)		(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueTask)				(cl_command_queue, cl_kernel, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueNativeKernel)		(cl_command_queue, void (CL_CALLBACK *)(void *), void *, size_t, cl_uint, const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueMarker)				(cl_command_queue, cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueWaitForEvents)		(cl_command_queue, cl_uint, const cl_event *);
typedef cl_int				(CL_API_ENTRY CL_API_CALL * p_pfn_clEnqueueBarrier)				(cl_command_queue);

// Extension function access
//
// Returns the extension function address for the given function name,
// or NULL if a valid function can not be found.  The client must
// check to make sure the address is not NULL, before using or 
// calling the returned function address.
//

typedef void *				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetExtensionFunctionAddress)(const char *);
typedef void *				(CL_API_ENTRY CL_API_CALL * p_pfn_clGetExtensionFunctionAddressForPlatform)(cl_platform_id, const char *);

p_pfn_clGetPlatformIDs				pfn_clGetPlatformIDs				= 0;
p_pfn_clGetPlatformInfo				pfn_clGetPlatformInfo				= 0;
p_pfn_clGetDeviceIDs				pfn_clGetDeviceIDs					= 0;
p_pfn_clGetDeviceInfo				pfn_clGetDeviceInfo					= 0;
p_pfn_clCreateContext				pfn_clCreateContext					= 0;
p_pfn_clCreateContextFromType		pfn_clCreateContextFromType			= 0;
p_pfn_clRetainContext				pfn_clRetainContext					= 0;
p_pfn_clReleaseContext				pfn_clReleaseContext				= 0;
p_pfn_clGetContextInfo				pfn_clGetContextInfo				= 0;
p_pfn_clCreateCommandQueue			pfn_clCreateCommandQueue			= 0;
p_pfn_clRetainCommandQueue			pfn_clRetainCommandQueue			= 0;
p_pfn_clReleaseCommandQueue			pfn_clReleaseCommandQueue			= 0;
p_pfn_clGetCommandQueueInfo			pfn_clGetCommandQueueInfo			= 0;
p_pfn_clSetCommandQueueProperty		pfn_clSetCommandQueueProperty		= 0;
p_pfn_clCreateBuffer				pfn_clCreateBuffer					= 0;
p_pfn_clCreateImage					pfn_clCreateImage					= 0;
p_pfn_clCreateImage2D				pfn_clCreateImage2D					= 0;
p_pfn_clCreateImage3D				pfn_clCreateImage3D					= 0;
p_pfn_clRetainMemObject				pfn_clRetainMemObject				= 0;
p_pfn_clReleaseMemObject			pfn_clReleaseMemObject				= 0;
p_pfn_clGetSupportedImageFormats	pfn_clGetSupportedImageFormats		= 0;
p_pfn_clGetMemObjectInfo			pfn_clGetMemObjectInfo				= 0;
p_pfn_clGetImageInfo				pfn_clGetImageInfo					= 0;
p_pfn_clCreateSampler				pfn_clCreateSampler					= 0;
p_pfn_clRetainSampler				pfn_clRetainSampler					= 0;
p_pfn_clReleaseSampler				pfn_clReleaseSampler				= 0;
p_pfn_clGetSamplerInfo				pfn_clGetSamplerInfo				= 0;
p_pfn_clCreateProgramWithSource		pfn_clCreateProgramWithSource		= 0;
p_pfn_clCreateProgramWithBinary		pfn_clCreateProgramWithBinary		= 0;
p_pfn_clCreateProgramWithIL			pfn_clCreateProgramWithIL			= 0;
p_pfn_clRetainProgram				pfn_clRetainProgram					= 0;
p_pfn_clReleaseProgram				pfn_clReleaseProgram				= 0;
p_pfn_clBuildProgram				pfn_clBuildProgram					= 0;
p_pfn_clUnloadCompiler				pfn_clUnloadCompiler				= 0;
p_pfn_clGetProgramInfo				pfn_clGetProgramInfo				= 0;
p_pfn_clGetProgramBuildInfo			pfn_clGetProgramBuildInfo			= 0;
p_pfn_clCreateKernel				pfn_clCreateKernel					= 0;
p_pfn_clCreateKernelsInProgram		pfn_clCreateKernelsInProgram		= 0;
p_pfn_clRetainKernel				pfn_clRetainKernel					= 0;
p_pfn_clReleaseKernel				pfn_clReleaseKernel					= 0;
p_pfn_clSetKernelArg				pfn_clSetKernelArg					= 0;
p_pfn_clGetKernelInfo				pfn_clGetKernelInfo					= 0;
p_pfn_clGetKernelWorkGroupInfo		pfn_clGetKernelWorkGroupInfo		= 0;
p_pfn_clWaitForEvents				pfn_clWaitForEvents					= 0;
p_pfn_clGetEventInfo				pfn_clGetEventInfo					= 0;
p_pfn_clRetainEvent					pfn_clRetainEvent					= 0;
p_pfn_clReleaseEvent				pfn_clReleaseEvent					= 0;
p_pfn_clGetEventProfilingInfo		pfn_clGetEventProfilingInfo			= 0;
p_pfn_clFlush						pfn_clFlush							= 0;
p_pfn_clFinish						pfn_clFinish						= 0;
p_pfn_clEnqueueReadBuffer			pfn_clEnqueueReadBuffer				= 0;
p_pfn_clEnqueueReadBufferRect		pfn_clEnqueueReadBufferRect			= 0;
p_pfn_clEnqueueWriteBuffer			pfn_clEnqueueWriteBuffer			= 0;
p_pfn_clEnqueueWriteBufferRect		pfn_clEnqueueWriteBufferRect		= 0;
p_pfn_clEnqueueCopyBuffer			pfn_clEnqueueCopyBuffer				= 0;
p_pfn_clEnqueueReadImage			pfn_clEnqueueReadImage				= 0;
p_pfn_clEnqueueWriteImage			pfn_clEnqueueWriteImage				= 0;
p_pfn_clEnqueueCopyImage			pfn_clEnqueueCopyImage				= 0;
p_pfn_clEnqueueCopyImageToBuffer	pfn_clEnqueueCopyImageToBuffer		= 0;
p_pfn_clEnqueueCopyBufferToImage	pfn_clEnqueueCopyBufferToImage		= 0;
p_pfn_clEnqueueMapBuffer			pfn_clEnqueueMapBuffer				= 0;
p_pfn_clEnqueueMapImage				pfn_clEnqueueMapImage				= 0;
p_pfn_clEnqueueUnmapMemObject		pfn_clEnqueueUnmapMemObject			= 0;
p_pfn_clEnqueueNDRangeKernel		pfn_clEnqueueNDRangeKernel			= 0;
p_pfn_clEnqueueTask					pfn_clEnqueueTask					= 0;
p_pfn_clEnqueueNativeKernel			pfn_clEnqueueNativeKernel			= 0;
p_pfn_clEnqueueMarker				pfn_clEnqueueMarker					= 0;
p_pfn_clEnqueueWaitForEvents		pfn_clEnqueueWaitForEvents			= 0;
p_pfn_clEnqueueBarrier				pfn_clEnqueueBarrier				= 0;
p_pfn_clGetExtensionFunctionAddress	pfn_clGetExtensionFunctionAddress	= 0;
p_pfn_clGetExtensionFunctionAddressForPlatform	pfn_clGetExtensionFunctionAddressForPlatform	= 0;

static bool has_ocl_version_1_0 = false;
static bool has_ocl_version_1_1 = false;
static bool has_ocl_version_1_2 = false;
static bool has_ocl_version_2_1 = false;

template <typename T>
inline void initFuncPtr(const OclLibrary &lib, bool &success, T &func_ptr, const char *name)
{
	func_ptr = (T) oclGetProcAddress(lib, name);
	if (!func_ptr)
		success = false;
}

#define INIT_FUNC_PTR(name) initFuncPtr(lib, result, pfn_##name, #name)

bool ocl_init_1_0(const OclLibrary &lib)
{
	bool result = true;
	INIT_FUNC_PTR(clGetPlatformIDs);
	INIT_FUNC_PTR(clGetPlatformInfo);
	INIT_FUNC_PTR(clGetDeviceIDs);
	INIT_FUNC_PTR(clGetDeviceInfo);
	INIT_FUNC_PTR(clCreateContext);
	INIT_FUNC_PTR(clCreateContextFromType);
	INIT_FUNC_PTR(clRetainContext);
	INIT_FUNC_PTR(clReleaseContext);
	INIT_FUNC_PTR(clGetContextInfo);
	INIT_FUNC_PTR(clCreateCommandQueue);
	INIT_FUNC_PTR(clRetainCommandQueue);
	INIT_FUNC_PTR(clReleaseCommandQueue);
	INIT_FUNC_PTR(clGetCommandQueueInfo);
	INIT_FUNC_PTR(clSetCommandQueueProperty);
	INIT_FUNC_PTR(clCreateBuffer);
	INIT_FUNC_PTR(clCreateImage);
	INIT_FUNC_PTR(clCreateImage2D);
	INIT_FUNC_PTR(clCreateImage3D);
	INIT_FUNC_PTR(clRetainMemObject);
	INIT_FUNC_PTR(clReleaseMemObject);
	INIT_FUNC_PTR(clGetSupportedImageFormats);
	INIT_FUNC_PTR(clGetMemObjectInfo);
	INIT_FUNC_PTR(clGetImageInfo);
	INIT_FUNC_PTR(clCreateSampler);
	INIT_FUNC_PTR(clRetainSampler);
	INIT_FUNC_PTR(clReleaseSampler);
	INIT_FUNC_PTR(clGetSamplerInfo);
	INIT_FUNC_PTR(clCreateProgramWithSource);
	INIT_FUNC_PTR(clCreateProgramWithBinary);
	INIT_FUNC_PTR(clRetainProgram);
	INIT_FUNC_PTR(clReleaseProgram);
	INIT_FUNC_PTR(clBuildProgram);
	INIT_FUNC_PTR(clUnloadCompiler);
	INIT_FUNC_PTR(clGetProgramInfo);
	INIT_FUNC_PTR(clGetProgramBuildInfo);
	INIT_FUNC_PTR(clCreateKernel);
	INIT_FUNC_PTR(clCreateKernelsInProgram);
	INIT_FUNC_PTR(clRetainKernel);
	INIT_FUNC_PTR(clReleaseKernel);
	INIT_FUNC_PTR(clSetKernelArg);
	INIT_FUNC_PTR(clGetKernelInfo);
	INIT_FUNC_PTR(clGetKernelWorkGroupInfo);
	INIT_FUNC_PTR(clWaitForEvents);
	INIT_FUNC_PTR(clGetEventInfo);
	INIT_FUNC_PTR(clRetainEvent);
	INIT_FUNC_PTR(clReleaseEvent);
	INIT_FUNC_PTR(clGetEventProfilingInfo);
	INIT_FUNC_PTR(clFlush);
	INIT_FUNC_PTR(clFinish);
	INIT_FUNC_PTR(clEnqueueReadBuffer);
	INIT_FUNC_PTR(clEnqueueWriteBuffer);
	INIT_FUNC_PTR(clEnqueueCopyBuffer);
	INIT_FUNC_PTR(clEnqueueReadImage);
	INIT_FUNC_PTR(clEnqueueWriteImage);
	INIT_FUNC_PTR(clEnqueueCopyImage);
	INIT_FUNC_PTR(clEnqueueCopyImageToBuffer);
	INIT_FUNC_PTR(clEnqueueCopyBufferToImage);
	INIT_FUNC_PTR(clEnqueueMapBuffer);
	INIT_FUNC_PTR(clEnqueueMapImage);
	INIT_FUNC_PTR(clEnqueueUnmapMemObject);
	INIT_FUNC_PTR(clEnqueueNDRangeKernel);
	INIT_FUNC_PTR(clEnqueueTask);
	INIT_FUNC_PTR(clEnqueueNativeKernel);
	INIT_FUNC_PTR(clEnqueueMarker);
	INIT_FUNC_PTR(clEnqueueWaitForEvents);
	INIT_FUNC_PTR(clEnqueueBarrier);
	INIT_FUNC_PTR(clGetExtensionFunctionAddress);
	return result;
}

bool ocl_init_1_1(const OclLibrary &lib)
{
	bool result = true;
	INIT_FUNC_PTR(clEnqueueReadBufferRect);
	INIT_FUNC_PTR(clEnqueueWriteBufferRect);
	return result;
}

bool ocl_init_1_2(const OclLibrary &lib)
{
	bool result = true;
	INIT_FUNC_PTR(clGetExtensionFunctionAddressForPlatform);
	return result;
}

bool ocl_init_2_1(const OclLibrary &lib)
{
	bool result = true;
	INIT_FUNC_PTR(clCreateProgramWithIL);
	return result;
}

bool ocl_init()
{
	if (pfn_clGetPlatformIDs)
		return true;

	OclLibrary lib = oclLoadLibrary();
	if (!lib)
		return false;

	has_ocl_version_1_0 = ocl_init_1_0(lib);
	has_ocl_version_1_1 = ocl_init_1_1(lib);
	has_ocl_version_1_2 = ocl_init_1_2(lib);
	has_ocl_version_2_1 = ocl_init_2_1(lib);
	return true;
}

bool ocl_version_1_2()
{
	return has_ocl_version_1_2;
}

bool ocl_version_2_1()
{
	return has_ocl_version_2_1;
}

// Platform API
extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint          num_entries,
                 cl_platform_id * platforms,
                 cl_uint *        num_platforms) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetPlatformIDs) return CL_INVALID_OPERATION;

	return pfn_clGetPlatformIDs(num_entries, platforms, num_platforms);
}

extern CL_API_ENTRY cl_int CL_API_CALL 
clGetPlatformInfo(cl_platform_id   platform,
                  cl_platform_info param_name,
                  size_t           param_value_size,
                  void *           param_value,
                  size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetPlatformInfo) return CL_INVALID_OPERATION;

	return pfn_clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

// Device APIs
extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id   platform,
               cl_device_type   device_type,
               cl_uint          num_entries,
               cl_device_id *   devices,
               cl_uint *        num_devices) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetDeviceIDs) return CL_INVALID_OPERATION;

	return pfn_clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id    device,
                cl_device_info  param_name, 
                size_t          param_value_size, 
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetDeviceInfo) return CL_INVALID_OPERATION;

	return pfn_clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
}

// Context APIs  
extern CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties * properties,
                cl_uint                 num_devices,
                const cl_device_id *    devices,
                void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                void *                  user_data,
                cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateContext) return 0;

	return pfn_clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

extern CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(const cl_context_properties * properties,
                        cl_device_type          device_type,
                        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                        void *                  user_data,
                        cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateContextFromType) return 0;

	return pfn_clCreateContextFromType(properties, device_type, pfn_notify, user_data, errcode_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clRetainContext) return CL_INVALID_OPERATION;

	return pfn_clRetainContext(context);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clReleaseContext) return CL_INVALID_OPERATION;

	return pfn_clReleaseContext(context);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetContextInfo(cl_context         context, 
                 cl_context_info    param_name, 
                 size_t             param_value_size, 
                 void *             param_value, 
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetContextInfo) return CL_INVALID_OPERATION;

	return pfn_clGetContextInfo(context, param_name, param_value_size, param_value, param_value_size_ret);
}

// Command Queue APIs
extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context                     context, 
                     cl_device_id                   device, 
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateCommandQueue) return 0;

	return pfn_clCreateCommandQueue(context, device, properties, errcode_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clRetainCommandQueue) return CL_INVALID_OPERATION;

	return pfn_clRetainCommandQueue(command_queue);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clReleaseCommandQueue) return CL_INVALID_OPERATION;

	return pfn_clReleaseCommandQueue(command_queue);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue      command_queue,
                      cl_command_queue_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetCommandQueueInfo) return CL_INVALID_OPERATION;

	return pfn_clGetCommandQueueInfo(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clSetCommandQueueProperty(cl_command_queue              command_queue,
                          cl_command_queue_properties   properties, 
                          cl_bool                        enable,
                          cl_command_queue_properties * old_properties) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clSetCommandQueueProperty) return CL_INVALID_OPERATION;

	return pfn_clSetCommandQueueProperty(command_queue, properties, enable, old_properties);
}

// Memory Object APIs
extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context   context,
               cl_mem_flags flags,
               size_t       size,
               void *       host_ptr,
               cl_int *     errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateBuffer) return 0;

	return pfn_clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                const cl_image_desc *   image_desc,
                void *                  host_ptr,
                cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
	if (!pfn_clCreateImage) return 0;

	return pfn_clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage2D(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                size_t                  image_width,
                size_t                  image_height,
                size_t                  image_row_pitch, 
                void *                  host_ptr,
                cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateImage2D) return 0;

	return pfn_clCreateImage2D(context, flags, image_format, image_width, image_height, image_row_pitch, host_ptr, errcode_ret);
}
                        
extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage3D(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                size_t                  image_width, 
                size_t                  image_height,
                size_t                  image_depth, 
                size_t                  image_row_pitch, 
                size_t                  image_slice_pitch, 
                void *                  host_ptr,
                cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateImage3D) return 0;

	return pfn_clCreateImage3D(context, flags, image_format, image_width, image_height, image_depth, image_row_pitch, image_slice_pitch, host_ptr, errcode_ret);
}
                        
extern CL_API_ENTRY cl_int CL_API_CALL
clRetainMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clRetainMemObject) return CL_INVALID_OPERATION;

	return pfn_clRetainMemObject(memobj);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clReleaseMemObject) return CL_INVALID_OPERATION;

	return pfn_clReleaseMemObject(memobj);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetSupportedImageFormats(cl_context           context,
                           cl_mem_flags         flags,
                           cl_mem_object_type   image_type,
                           cl_uint              num_entries,
                           cl_image_format *    image_formats,
                           cl_uint *            num_image_formats) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetSupportedImageFormats) return CL_INVALID_OPERATION;

	return pfn_clGetSupportedImageFormats(context, flags, image_type, num_entries, image_formats, num_image_formats);
}
                                    
extern CL_API_ENTRY cl_int CL_API_CALL
clGetMemObjectInfo(cl_mem           memobj,
                   cl_mem_info      param_name, 
                   size_t           param_value_size,
                   void *           param_value,
                   size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetMemObjectInfo) return CL_INVALID_OPERATION;

	return pfn_clGetMemObjectInfo(memobj, param_name, param_value_size, param_value, param_value_size_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetImageInfo(cl_mem           image,
               cl_image_info    param_name, 
               size_t           param_value_size,
               void *           param_value,
               size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetImageInfo) return CL_INVALID_OPERATION;

	return pfn_clGetImageInfo(image, param_name, param_value_size, param_value, param_value_size_ret);
}


// Sampler APIs
extern CL_API_ENTRY cl_sampler CL_API_CALL
clCreateSampler(cl_context          context,
                cl_bool             normalized_coords, 
                cl_addressing_mode  addressing_mode, 
                cl_filter_mode      filter_mode,
                cl_int *            errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateSampler) return 0;

	return pfn_clCreateSampler(context, normalized_coords, addressing_mode, filter_mode, errcode_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainSampler(cl_sampler sampler) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clRetainSampler) return CL_INVALID_OPERATION;

	return pfn_clRetainSampler(sampler);
}


extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseSampler(cl_sampler sampler) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clReleaseSampler) return CL_INVALID_OPERATION;

	return pfn_clReleaseSampler(sampler);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetSamplerInfo(cl_sampler         sampler,
                 cl_sampler_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetSamplerInfo) return CL_INVALID_OPERATION;

	return pfn_clGetSamplerInfo(sampler, param_name, param_value_size, param_value, param_value_size_ret);
}
                            
// Program Object APIs
extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateProgramWithSource) return 0;

	return pfn_clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
}

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary(cl_context                     context,
                          cl_uint                        num_devices,
                          const cl_device_id *           device_list,
                          const size_t *                 lengths,
                          const unsigned char **         binaries,
                          cl_int *                       binary_status,
                          cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateProgramWithBinary) return 0;

	return pfn_clCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
}

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithIL(cl_context             context,
                      const void *           il,
                      size_t                 length,
                      cl_int *               errcode_ret) CL_API_SUFFIX__VERSION_2_1
{
	if (!pfn_clCreateProgramWithIL) return 0;

	return pfn_clCreateProgramWithIL(context, il, length, errcode_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clRetainProgram) return CL_INVALID_OPERATION;

	return pfn_clRetainProgram(program);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clReleaseProgram) return CL_INVALID_OPERATION;

	return pfn_clReleaseProgram(program);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program           program,
               cl_uint              num_devices,
               const cl_device_id * device_list,
               const char *         options, 
               void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
               void *               user_data) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clBuildProgram) return CL_INVALID_OPERATION;

	return pfn_clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clUnloadCompiler(void) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clUnloadCompiler) return CL_INVALID_OPERATION;

	return pfn_clUnloadCompiler();
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetProgramInfo(cl_program         program,
                 cl_program_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetProgramInfo) return CL_INVALID_OPERATION;

	return pfn_clGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetProgramBuildInfo) return CL_INVALID_OPERATION;

	return pfn_clGetProgramBuildInfo(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

// Kernel Object APIs
extern CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateKernel) return 0;

	return pfn_clCreateKernel(program, kernel_name, errcode_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clCreateKernelsInProgram(cl_program     program,
                         cl_uint        num_kernels,
                         cl_kernel *    kernels,
                         cl_uint *      num_kernels_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clCreateKernelsInProgram) return CL_INVALID_OPERATION;

	return pfn_clCreateKernelsInProgram(program, num_kernels, kernels, num_kernels_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainKernel(cl_kernel    kernel) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clRetainKernel) return CL_INVALID_OPERATION;

	return pfn_clRetainKernel(kernel);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel   kernel) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clReleaseKernel) return CL_INVALID_OPERATION;

	return pfn_clReleaseKernel(kernel);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clSetKernelArg) return CL_INVALID_OPERATION;

	return pfn_clSetKernelArg(kernel, arg_index, arg_size, arg_value);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelInfo(cl_kernel       kernel,
                cl_kernel_info  param_name,
                size_t          param_value_size,
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetKernelInfo) return CL_INVALID_OPERATION;

	return pfn_clGetKernelInfo(kernel, param_name, param_value_size, param_value, param_value_size_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelWorkGroupInfo(cl_kernel                  kernel,
                         cl_device_id               device,
                         cl_kernel_work_group_info  param_name,
                         size_t                     param_value_size,
                         void *                     param_value,
                         size_t *                   param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetKernelWorkGroupInfo) return CL_INVALID_OPERATION;

	return pfn_clGetKernelWorkGroupInfo(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

// Event Object APIs
extern CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint             num_events,
                const cl_event *    event_list) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clWaitForEvents) return CL_INVALID_OPERATION;

	return pfn_clWaitForEvents(num_events, event_list);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetEventInfo(cl_event         event,
               cl_event_info    param_name,
               size_t           param_value_size,
               void *           param_value,
               size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetEventInfo) return CL_INVALID_OPERATION;

	return pfn_clGetEventInfo(event, param_name, param_value_size, param_value, param_value_size_ret);
}
                            
extern CL_API_ENTRY cl_int CL_API_CALL
clRetainEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clRetainEvent) return CL_INVALID_OPERATION;

	return pfn_clRetainEvent(event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clReleaseEvent) return CL_INVALID_OPERATION;

	return pfn_clReleaseEvent(event);
}

// Profiling APIs
extern CL_API_ENTRY cl_int CL_API_CALL
clGetEventProfilingInfo(cl_event            event,
                        cl_profiling_info   param_name,
                        size_t              param_value_size,
                        void *              param_value,
                        size_t *            param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetEventProfilingInfo) return CL_INVALID_OPERATION;

	return pfn_clGetEventProfilingInfo(event, param_name, param_value_size, param_value, param_value_size_ret);
}

// Flush and Finish APIs
extern CL_API_ENTRY cl_int CL_API_CALL
clFlush(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clFlush) return CL_INVALID_OPERATION;

	return pfn_clFlush(command_queue);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clFinish) return CL_INVALID_OPERATION;

	return pfn_clFinish(command_queue);
}

// Enqueued Commands APIs
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue    command_queue,
                    cl_mem              buffer,
                    cl_bool             blocking_read,
                    size_t              offset,
                    size_t              cb,
                    void *              ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueReadBuffer) return CL_INVALID_OPERATION;

	return pfn_clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBufferRect(cl_command_queue	command_queue,
						cl_mem				buffer,
						cl_bool				blocking_read,
						const size_t *		buffer_origin,
						const size_t *		host_origin,
						const size_t *		region,
						size_t				buffer_row_pitch,
						size_t				buffer_slice_pitch,
						size_t				host_row_pitch,
						size_t				host_slice_pitch,
						void *				ptr,
						cl_uint				num_events_in_wait_list,
						const cl_event *	event_wait_list,
						cl_event *			event) CL_API_SUFFIX__VERSION_1_1
{
	if (!pfn_clEnqueueReadBufferRect) return CL_INVALID_OPERATION;

	return pfn_clEnqueueReadBufferRect(command_queue, buffer, blocking_read, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}
                            
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue   command_queue,
                     cl_mem             buffer,
                     cl_bool            blocking_write,
                     size_t             offset,
                     size_t             cb,
                     const void *       ptr,
                     cl_uint            num_events_in_wait_list,
                     const cl_event *   event_wait_list,
                     cl_event *         event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueWriteBuffer) return CL_INVALID_OPERATION;

	return pfn_clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBufferRect(cl_command_queue	command_queue,
                         cl_mem				buffer,
                         cl_bool			blocking_write,
                         const size_t *		buffer_origin,
                         const size_t *		host_origin,
                         const size_t *		region,
                         size_t				buffer_row_pitch,
                         size_t				buffer_slice_pitch,
                         size_t				host_row_pitch,
                         size_t				host_slice_pitch,
                         const void *		ptr,
                         cl_uint			num_events_in_wait_list,
                         const cl_event *	event_wait_list,
                         cl_event *			event) CL_API_SUFFIX__VERSION_1_1
{
	if (!pfn_clEnqueueWriteBufferRect) return CL_INVALID_OPERATION;

	return pfn_clEnqueueWriteBufferRect(command_queue, buffer, blocking_write, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue    command_queue, 
                    cl_mem              src_buffer,
                    cl_mem              dst_buffer, 
                    size_t              src_offset,
                    size_t              dst_offset,
                    size_t              cb, 
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueCopyBuffer) return CL_INVALID_OPERATION;

	return pfn_clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, cb, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadImage(cl_command_queue     command_queue,
                   cl_mem               image,
                   cl_bool              blocking_read, 
                   const size_t *       origin,
                   const size_t *       region,
                   size_t               row_pitch,
                   size_t               slice_pitch, 
                   void *               ptr,
                   cl_uint              num_events_in_wait_list,
                   const cl_event *     event_wait_list,
                   cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueReadImage) return CL_INVALID_OPERATION;

	return pfn_clEnqueueReadImage(command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteImage(cl_command_queue    command_queue,
                    cl_mem              image,
                    cl_bool             blocking_write, 
                    const size_t *      origin[3],
                    const size_t *      region[3],
                    size_t              input_row_pitch,
                    size_t              input_slice_pitch, 
                    const void *        ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueWriteImage) return CL_INVALID_OPERATION;

	return pfn_clEnqueueWriteImage(command_queue, image, blocking_write, origin, region, input_row_pitch, input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImage(cl_command_queue     command_queue,
                   cl_mem               src_image,
                   cl_mem               dst_image, 
                   const size_t *       src_origin[3],
                   const size_t *       dst_origin[3],
                   const size_t *       region[3], 
                   cl_uint              num_events_in_wait_list,
                   const cl_event *     event_wait_list,
                   cl_event *           event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueCopyImage) return CL_INVALID_OPERATION;

	return pfn_clEnqueueCopyImage(command_queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImageToBuffer(cl_command_queue command_queue,
                           cl_mem           src_image,
                           cl_mem           dst_buffer, 
                           const size_t *   src_origin[3],
                           const size_t *   region[3], 
                           size_t           dst_offset,
                           cl_uint          num_events_in_wait_list,
                           const cl_event * event_wait_list,
                           cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueCopyImageToBuffer) return CL_INVALID_OPERATION;

	return pfn_clEnqueueCopyImageToBuffer(command_queue, src_image, dst_buffer, src_origin, region, dst_offset, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferToImage(cl_command_queue command_queue,
                           cl_mem           src_buffer,
                           cl_mem           dst_image, 
                           size_t           src_offset,
                           const size_t *   dst_origin,
                           const size_t *   region, 
                           cl_uint          num_events_in_wait_list,
                           const cl_event * event_wait_list,
                           cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueCopyBufferToImage) return CL_INVALID_OPERATION;

	return pfn_clEnqueueCopyBufferToImage(command_queue, src_buffer, dst_image, src_offset, dst_origin, region, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY void * CL_API_CALL
clEnqueueMapBuffer(cl_command_queue command_queue,
                   cl_mem           buffer,
                   cl_bool          blocking_map, 
                   cl_map_flags     map_flags,
                   size_t           offset,
                   size_t           cb,
                   cl_uint          num_events_in_wait_list,
                   const cl_event * event_wait_list,
                   cl_event *       event,
                   cl_int *         errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueMapBuffer) return 0;

	return pfn_clEnqueueMapBuffer(command_queue, buffer, blocking_map, map_flags, offset, cb, num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

extern CL_API_ENTRY void * CL_API_CALL
clEnqueueMapImage(cl_command_queue  command_queue,
                  cl_mem            image, 
                  cl_bool           blocking_map, 
                  cl_map_flags      map_flags, 
                  const size_t *    origin,
                  const size_t *    region,
                  size_t *          image_row_pitch,
                  size_t *          image_slice_pitch,
                  cl_uint           num_events_in_wait_list,
                  const cl_event *  event_wait_list,
                  cl_event *        event,
                  cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueMapImage) return 0;

	return pfn_clEnqueueMapImage(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch, num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueUnmapMemObject(cl_command_queue command_queue,
                        cl_mem           memobj,
                        void *           mapped_ptr,
                        cl_uint          num_events_in_wait_list,
                        const cl_event *  event_wait_list,
                        cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueUnmapMemObject) return CL_INVALID_OPERATION;

	return pfn_clEnqueueUnmapMemObject(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       cl_uint          num_events_in_wait_list,
                       const cl_event * event_wait_list,
                       cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueNDRangeKernel) return CL_INVALID_OPERATION;

	return pfn_clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueTask(cl_command_queue  command_queue,
              cl_kernel         kernel,
              cl_uint           num_events_in_wait_list,
              const cl_event *  event_wait_list,
              cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueTask) return CL_INVALID_OPERATION;

	return pfn_clEnqueueTask(command_queue, kernel, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNativeKernel(cl_command_queue  command_queue,
					  void (CL_CALLBACK *user_func)(void *), 
                      void *            args,
                      size_t            cb_args, 
                      cl_uint           num_mem_objects,
                      const cl_mem *    mem_list,
                      const void **     args_mem_loc,
                      cl_uint           num_events_in_wait_list,
                      const cl_event *  event_wait_list,
                      cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueNativeKernel) return CL_INVALID_OPERATION;

	return pfn_clEnqueueNativeKernel(command_queue, user_func, args, cb_args, num_mem_objects, mem_list, args_mem_loc, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMarker(cl_command_queue    command_queue,
                cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueMarker) return CL_INVALID_OPERATION;

	return pfn_clEnqueueMarker(command_queue, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWaitForEvents(cl_command_queue command_queue,
                       cl_uint          num_events,
                       const cl_event * event_list) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueWaitForEvents) return CL_INVALID_OPERATION;

	return pfn_clEnqueueWaitForEvents(command_queue, num_events, event_list);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueBarrier(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clEnqueueBarrier) return CL_INVALID_OPERATION;

	return pfn_clEnqueueBarrier(command_queue);
}

// Extension function access
//
// Returns the extension function address for the given function name,
// or NULL if a valid function can not be found.  The client must
// check to make sure the address is not NULL, before using or 
// calling the returned function address.
//
extern CL_API_ENTRY void * CL_API_CALL clGetExtensionFunctionAddress(const char * func_name) CL_API_SUFFIX__VERSION_1_0
{
	if (!pfn_clGetExtensionFunctionAddress) return 0;

	return pfn_clGetExtensionFunctionAddress(func_name);
}

extern CL_API_ENTRY void * CL_API_CALL clGetExtensionFunctionAddressForPlatform(cl_platform_id platform, const char * func_name) CL_API_SUFFIX__VERSION_1_2
{
	if (!pfn_clGetExtensionFunctionAddressForPlatform) return 0;

	return pfn_clGetExtensionFunctionAddressForPlatform(platform, func_name);
}
