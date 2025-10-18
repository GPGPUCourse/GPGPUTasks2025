#define _SHORT_FILE_ "shared_host_buffer.cpp"

#include "shared_host_buffer.h"
#include "context.h"
#include <algorithm>
#include <stdexcept>

#ifdef CUDA_SUPPORT
#include <libgpu/cuda/utils.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace gpu {

shared_host_buffer::shared_host_buffer()
{
	buffer_	= 0;
	data_	= 0;
	type_	= Context::TypeUndefined;
	size_	= 0;
}

shared_host_buffer::~shared_host_buffer()
{
	try {
		decref();
	}
	catch (std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}
}

shared_host_buffer::shared_host_buffer(const shared_host_buffer &other)
{
	buffer_	= other.buffer_;
	data_	= other.data_;
	type_	= other.type_;
	size_	= other.size_;
	incref();
}

shared_host_buffer &shared_host_buffer::operator= (const shared_host_buffer &other)
{
	if (this != &other) {
		decref();
		buffer_	= other.buffer_;
		data_	= other.data_;
		type_	= other.type_;
		size_	= other.size_;
		incref();
	}

	return *this;
}

void shared_host_buffer::swap(shared_host_buffer &other)
{
	std::swap(buffer_,	other.buffer_);
	std::swap(data_,	other.data_);
	std::swap(type_,	other.type_);
	std::swap(size_,	other.size_);
}

void shared_host_buffer::incref()
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

void shared_host_buffer::decref()
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

	if (count)
		return;

	switch (type_) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaFreeHost(data_));
		break;
#endif
	case Context::TypeOpenCL:
		free(data_);
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	delete [] buffer_;

	buffer_ = 0;
	data_	= 0;
	type_	= Context::TypeUndefined;
	size_	= 0;
}

shared_host_buffer shared_host_buffer::create(size_t size)
{
	shared_host_buffer res;
	res.resize(size);
	return res;
}

void *shared_host_buffer::get() const
{
	return data_;
}

size_t shared_host_buffer::size() const
{
	return size_;
}

void shared_host_buffer::resize(size_t size)
{
	if (size == size_)
		return;

	decref();

	buffer_	= new unsigned char [8];
	* (long long *) buffer_ = 0;
	incref();

	Context context;
	Context::Type type = context.type();

	switch (type) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL( cudaMallocHost(&data_, size) );
		break;
#endif
	case Context::TypeOpenCL:
		// TODO: implement pinned memory in opencl
		// currently we use a plain paged memory buffer
		data_ = malloc(size);
		if (!data_)
			throw std::bad_alloc();
		break;
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	type_ = type;
	size_ = size;
}

void shared_host_buffer::grow(size_t size)
{
	if (size > size_)
		resize(size);
}

template<typename T>
shared_host_buffer_typed<T> shared_host_buffer_typed<T>::createN(size_t number)
{
	shared_host_buffer_typed<T> res;
	res.resizeN(number);
	return res;
}

template <typename T>
void shared_host_buffer_typed<T>::resizeN(size_t number)
{
	this->resize(number * sizeof(T));
}

template <typename T>
T *shared_host_buffer_typed<T>::get() const
{
	return (T*) data_;
}

template<typename T>
size_t shared_host_buffer_typed<T>::number() const
{
	return this->size_ / sizeof(T);
}

template class shared_host_buffer_typed<char>;
template class shared_host_buffer_typed<unsigned char>;
template class shared_host_buffer_typed<short>;
template class shared_host_buffer_typed<unsigned short>;
template class shared_host_buffer_typed<int>;
template class shared_host_buffer_typed<unsigned int>;
template class shared_host_buffer_typed<float>;
template class shared_host_buffer_typed<double>;

}
