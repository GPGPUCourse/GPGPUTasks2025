#define _SHORT_FILE_ "shared_device_buffer.cpp"

#include "shared_device_buffer.h"
#include "context.h"
#include <libbase/string_utils.h>
#include <libgpu/utils.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

#ifdef CUDA_SUPPORT
#include <libgpu/cuda/utils.h>
#include <cuda_runtime.h>
#endif

#include "vulkan/utils.h"
#include "vulkan/engine.h"
#include "vulkan/data_buffer.h"
#include "vulkan/vulkan_api_headers.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace gpu {

shared_device_buffer::shared_device_buffer()
{
	buffer_	= 0;
	data_	= 0;
	type_	= Context::TypeUndefined;
	size_	= 0;
	offset_	= 0;
	nbytes_guard_prefix_	= 0;
	nbytes_guard_suffix_	= 0;
}

shared_device_buffer::~shared_device_buffer()
{
	try {
		decref();
	}
	catch (std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}
}

shared_device_buffer::shared_device_buffer(const shared_device_buffer &other, size_t offset)
{
	buffer_	= other.buffer_;
	data_	= other.data_;
	type_	= other.type_;
	size_	= other.size_ - offset;
	offset_	= other.offset_ + offset;
	nbytes_guard_prefix_	= other.nbytes_guard_prefix_;
	nbytes_guard_suffix_	= other.nbytes_guard_suffix_;
	incref();
}

shared_device_buffer &shared_device_buffer::operator= (const shared_device_buffer &other)
{
	if (this != &other) {
		decref();
		buffer_	= other.buffer_;
		data_	= other.data_;
		type_	= other.type_;
		size_	= other.size_;
		offset_	= other.offset_;
		nbytes_guard_prefix_	= other.nbytes_guard_prefix_;
		nbytes_guard_suffix_	= other.nbytes_guard_suffix_;
		incref();
	}

	return *this;
}

void shared_device_buffer::swap(shared_device_buffer &other)
{
	std::swap(buffer_,	other.buffer_);
	std::swap(data_,	other.data_);
	std::swap(type_,	other.type_);
	std::swap(size_,	other.size_);
	std::swap(offset_,	other.offset_);
	std::swap(nbytes_guard_prefix_,	other.nbytes_guard_prefix_);
	std::swap(nbytes_guard_suffix_,	other.nbytes_guard_suffix_);
}

void shared_device_buffer::incref()
{
	if (!buffer_)
		return;

#if defined(_WIN64)
	InterlockedIncrement64((LONGLONG *) buffer_);
#elif defined(_WIN32)
	InterlockedIncrement((LONG *) buffer_);
#else
	__sync_add_and_fetch((long long *) buffer_, 1);
#endif
}

void shared_device_buffer::decref()
{
	if (!buffer_)
		return;

	long long count = 0;

#if defined(_WIN64)
	count = InterlockedDecrement64((LONGLONG *) buffer_);
#elif defined(_WIN32)
	count = InterlockedDecrement((LONG *) buffer_);
#else
	count = __sync_sub_and_fetch((long long *) buffer_, 1);
#endif

	gpu::Context context;
	if (context.type() != Context::TypeUndefined || (type_ == Context::TypeCUDA)) {
		// if destructor is called from the thread with gpu::Context - everything is fine, we have the context in thread local variable
		// and so we can read magic guards data,
		// but if destructor was called from another thread - we don't have the context and so (in case of OpenCL and Vulkan) we can't read magic guards data,
		// note that in CUDA we can read magic guards data in any way - it doesn't require any context
		checkMagicGuards("decref");
	}

	if (!count) {
		switch (type_) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			CUDA_SAFE_CALL(cudaFree(data_));
			break;
#endif
		case Context::TypeOpenCL:
			OCL_SAFE_CALL(clReleaseMemObject((cl_mem) data_));
			break;
		case Context::TypeVulkan:
		{
			avk2::raii::BufferData* vk_data = (avk2::raii::BufferData *) data_;
			delete vk_data;
			break;
		}
		default:
			gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
		}

		delete [] buffer_;
	}

	buffer_ = 0;
	data_	= 0;
	type_	= Context::TypeUndefined;
	size_	= 0;
	offset_	= 0;
	nbytes_guard_prefix_ = 0;
	nbytes_guard_suffix_ = 0;
}

shared_device_buffer shared_device_buffer::create(size_t size)
{
	shared_device_buffer res;
	res.resize(size);
	return res;
}

void *shared_device_buffer::cuptr() const
{
	if (type_ == Context::TypeOpenCL)
		throw gpu_exception("GPU buffer type mismatch");

	return (char *) data_ + offset_;
}

cl_mem shared_device_buffer::clmem() const
{
	if (type_ == Context::TypeCUDA)
		throw gpu_exception("GPU buffer type mismatch");

	return (cl_mem) data_;
}

avk2::raii::BufferData *shared_device_buffer::vkBufferData() const
{
	avk2::raii::BufferData* vk_data = (avk2::raii::BufferData *) data_;
	return vk_data;
}

size_t shared_device_buffer::cloffset() const
{
	if (type_ == Context::TypeCUDA)
		throw gpu_exception("GPU buffer type mismatch");

	return offset_;
}

size_t shared_device_buffer::vkoffset() const
{
	if (type_ != Context::TypeVulkan)
		throw gpu_exception("GPU buffer type mismatch");

	return offset_;
}

size_t shared_device_buffer::size() const
{
	return size_;
}

bool shared_device_buffer::isNull() const
{
	return data_ == NULL;
}

void shared_device_buffer::reset()
{
	decref();
}

void shared_device_buffer::resize(size_t size)
{
	if (size == size_)
		return;

	decref();

	Context context;
	Context::Type type = context.type();

	if (context.isMemoryGuardsEnabled()) { 
		if (size > GPU_BUFFER_BIG_MAGIC_GUARD_NBYTES) {
			nbytes_guard_prefix_ = GPU_BUFFER_BIG_MAGIC_GUARD_NBYTES;
			nbytes_guard_suffix_ = GPU_BUFFER_BIG_MAGIC_GUARD_NBYTES;
		} else {
			nbytes_guard_prefix_ = GPU_BUFFER_SMALL_MAGIC_GUARD_NBYTES;
			nbytes_guard_suffix_ = GPU_BUFFER_SMALL_MAGIC_GUARD_NBYTES;
		}

		if (type == Context::TypeOpenCL) {
			// OpenCL's cl_mem doesn't support pointers arithmetic,
			// so it is not easy to move cl_mem w.r.t. prefix guard
			// TODO add full support (prefix guard + suffix guard instead of just suffix guard) with clCreateSubBuffer(...) - https://registry.khronos.org/OpenCL/sdk/1.2/docs/man/xhtml/clCreateSubBuffer.html
			nbytes_guard_prefix_ = 0;
		} else if (type == Context::TypeVulkan) {
			rassert(nbytes_guard_prefix_ % context.vk()->device().min_storage_buffer_offset_alignment == 0, 339520659);
		}
	}

	size_t size_with_magic_bytes_guards = nbytes_guard_prefix_ + size + nbytes_guard_suffix_;

	switch (type) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL( cudaMalloc(&data_, size_with_magic_bytes_guards) );
		break;
#endif
	case Context::TypeOpenCL:
		data_ = context.cl()->createBuffer(CL_MEM_READ_WRITE, size_with_magic_bytes_guards);
		break;
	case Context::TypeVulkan:
		data_ = context.vk()->createBuffer(size_with_magic_bytes_guards);
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	buffer_	= new unsigned char [8];
	* (long long *) buffer_ = 0;
	incref();

	type_	= type;
	size_	= size;
	offset_	= nbytes_guard_prefix_;

	writeMagicGuards();
}

void shared_device_buffer::grow(size_t size, float reserveMultiplier)
{
	if (size > size_)
		resize(std::max(size, (size_t) (size * reserveMultiplier)));
}

void shared_device_buffer::write(const void *data, size_t size)
{
	if (size == 0)
		return;

	if (size > size_ + nbytes_guard_suffix_)
		throw gpu_exception("Too many data for this device buffer: " + to_string(size) + " > " + to_string(size_) + "+" + to_string(nbytes_guard_suffix_));

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaMemcpy(cuptr(), data, size, cudaMemcpyHostToDevice));
		break;
#endif
	case Context::TypeOpenCL:
		context.cl()->writeBuffer((cl_mem) data_, CL_TRUE, offset_, size, data);
		break;
	case Context::TypeVulkan:
	{
		context.vk()->writeBuffer(*((avk2::raii::BufferData*) data_), offset_, size, data);
		break;
	}
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}
}

void shared_device_buffer::write(const shared_device_buffer &buffer, size_t size)
{
	if (!size)
		return;

	if (size > size_)
		throw gpu_exception("Too many data for this device buffer: " + to_string(size) + " > " + to_string(size_));

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaMemcpy(cuptr(), buffer.cuptr(), size, cudaMemcpyDeviceToDevice));
		break;
#endif
	case Context::TypeOpenCL:
		context.cl()->copyBuffer(buffer.clmem(), clmem(), buffer.cloffset(), cloffset(), size);
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}
}

void shared_device_buffer::write(const shared_host_buffer &buffer, size_t size)
{
	if (!size)
		return;

	if (size > size_)
		throw gpu_exception("Too many data for this device buffer: " + to_string(size) + " > " + to_string(size_));

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaMemcpy(cuptr(), buffer.get(), size, cudaMemcpyHostToDevice));
		break;
#endif
	case Context::TypeOpenCL:
		context.cl()->writeBuffer((cl_mem) data_, CL_TRUE, offset_, size, buffer.get());
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}
}

void shared_device_buffer::write2D(size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
	if (spitch == width && dpitch == width) {
		write(src, width * height);
		return;
	}

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaMemcpy2D(cuptr(), dpitch, src, spitch, width, height, cudaMemcpyHostToDevice));
		break;
#endif
	case Context::TypeOpenCL:
		{
			size_t buffer_origin[3] = { offset_, 0, 0 };
			size_t host_origin[3] = { 0, 0, 0 };
			size_t region[3] = { width, height, 1 };
			context.cl()->writeBufferRect((cl_mem) data_, CL_TRUE, buffer_origin, host_origin, region, dpitch, 0, spitch, 0, src);
		}
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}
}

void shared_device_buffer::read(void *data, size_t size, ptrdiff_t offset) const
{
	if (size == 0)
		return;

	ptrdiff_t total_offset = (ptrdiff_t) offset_ + offset;
	rassert(total_offset >= 0, 621161921);

	if (size > size_ + nbytes_guard_suffix_)
		throw gpu_exception("Not enough data in this device buffer: " + to_string(offset) + "+" + to_string(size) + " > " + to_string(size_));

#ifdef CUDA_SUPPORT
	if (type_ == Context::TypeCUDA) {
		// we can read data from CUDA buffer even without context - this is helpful in case of destruction from other thread (calling decref() and checkMagicGuards())
		CUDA_SAFE_CALL(cudaMemcpy(data, (char *) cuptr() + offset, size, cudaMemcpyDeviceToHost));
	} else
#endif
	{
		Context context;
		switch (type_) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			// this is an impossible case - CUDA handled above without thread-local context usage
			rassert(type_ == Context::TypeCUDA, 344486037);
			break;
#endif
		case Context::TypeOpenCL:
			context.cl()->readBuffer((cl_mem) data_, CL_TRUE, offset_ + offset, size, data);
			break;
		case Context::TypeVulkan:
			context.vk()->readBuffer(*((avk2::raii::BufferData*) data_), offset_ + offset, size, data);
			break;
		default:
			gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
		}
	}
}

void shared_device_buffer::read2D(size_t spitch, void *dst, size_t dpitch, size_t width, size_t height) const
{
	if (spitch == width && dpitch == width) {
		read(dst, width * height);
		return;
	}

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			CUDA_SAFE_CALL(cudaMemcpy2D(dst, dpitch, cuptr(), spitch, width, height, cudaMemcpyDeviceToHost));
			break;
#endif
		case Context::TypeOpenCL:
		{
			size_t buffer_origin[3] = { offset_, 0, 0 };
			size_t host_origin[3] = { 0, 0, 0 };
			size_t region[3] = { width, height, 1 };
			context.cl()->readBufferRect((cl_mem) data_, CL_TRUE, buffer_origin, host_origin, region, spitch, 0, dpitch, 0, dst);
		}
			break;
		default:
			gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}
}

void shared_device_buffer::copyTo(shared_device_buffer &that, size_t size) const
{
	if (size == 0)
		return;
	if (size > size_)
		throw gpu_exception("Not enough data in this device buffer: " + to_string(size) + " > " + to_string(size_));

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			CUDA_SAFE_CALL(cudaMemcpy((char *) that.cuptr(), (char *) cuptr(), size, cudaMemcpyDeviceToDevice));
			break;
#endif
		case Context::TypeOpenCL:
			context.cl()->copyBuffer(clmem(), that.clmem(), offset_, that.offset_, size);
			break;
		default:
			gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}
}

const unsigned int* shared_device_buffer::getMagicGuardBytesHostPtr()
{
	static std::vector<unsigned int> guard_magic_bytes(GPU_BUFFER_BIG_MAGIC_GUARD_NVALUES, GPU_BUFFER_MAGIC_GUARD_VALUE);
	return guard_magic_bytes.data();
}

void shared_device_buffer::writeMagicGuards()
{
	rassert(offset_ == nbytes_guard_prefix_, 401744672);
	if (nbytes_guard_prefix_ > 0) {
		offset_	= 0;
		write(getMagicGuardBytesHostPtr(), nbytes_guard_prefix_);
	}
	if (nbytes_guard_suffix_ > 0) {
		offset_ = nbytes_guard_prefix_ + size_;
		write(getMagicGuardBytesHostPtr(), nbytes_guard_suffix_);
	}
	offset_ = nbytes_guard_prefix_;
}

bool shared_device_buffer::checkMagicGuards(const std::string &info) const
{
	if (nbytes_guard_prefix_ > 0) {
		std::vector<unsigned int> guard_magic_bytes_found(nbytes_guard_prefix_);
		read(guard_magic_bytes_found.data(), nbytes_guard_prefix_, -(ptrdiff_t) offset_);
		if (!checkMagicGuardBytes(guard_magic_bytes_found.data(), nbytes_guard_prefix_, "prefix " + info)) {
			return false;
		}
	}
	if (nbytes_guard_suffix_ > 0) {
		std::vector<unsigned int> guard_magic_bytes_found(nbytes_guard_suffix_);
		read(guard_magic_bytes_found.data(), nbytes_guard_suffix_, size_);
		if (!checkMagicGuardBytes(guard_magic_bytes_found.data(), nbytes_guard_suffix_, "suffix " + info)) {
			return false;
		}
	}
	return true;
}

bool shared_device_buffer::checkMagicGuardBytes(const unsigned int* found_data, size_t nbytes, const std::string &info) const
{
	rassert(nbytes <= GPU_BUFFER_BIG_MAGIC_GUARD_NBYTES, 224549017);
	size_t nvalues = (nbytes / sizeof(unsigned int));
	for (size_t i = 0; i < nvalues; ++i) {
		unsigned int found_value = found_data[i];
		unsigned int expected_value = getMagicGuardBytesHostPtr()[i];
		if (found_value != expected_value) {
			std::string context_type;
			Context context;
			if (context.type() == Context::TypeCUDA) context_type = "CUDA";
			else if (context.type() == Context::TypeOpenCL) context_type = "OpenCL";
			else if (context.type() == Context::TypeVulkan) context_type = "Vulkan";
			else rassert(false, 80907580);

			std::string message_text = context_type + " GPU buffer with size=" + to_string(size_) + " and offset=" + to_string(offset_)
					+ " has broken magic guard at index#" + to_string(i) + " (found value=" + to_string(found_value) + ")";
			if (!info.empty()) {
				message_text = message_text + ", info: " + info;
			}
			std::cerr << message_text << std::endl;
			return false;
		}
	}
	return true;
}

}
