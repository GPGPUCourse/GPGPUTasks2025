#pragma once

#include <string>
#include <vector>
#include <cstddef>

#include "shared_host_buffer.h"

#include <libbase/point.h>
#include <libbase/runtime_assert.h>

#define GPU_BUFFER_MAGIC_GUARD_VALUE		1239239239
#define GPU_BUFFER_SMALL_MAGIC_GUARD_NBYTES	(64) // >=64 for Intel UHD 770, >=16 bytes for NVIDIA/AMD, to prevent Validation Error: [ VUID-VkWriteDescriptorSet-descriptorType-00328 ] | MessageID = 0xea08144e | vkUpdateDescriptorSets(): pDescriptorWrites[0].pBufferInfo[0].offset (4) must be a multiple of device limit minStorageBufferOffsetAlignment 16 when descriptor type is VK_DESCRIPTOR_TYPE_STORAGE_BUFFER. The Vulkan spec states: If descriptorType is VK_DESCRIPTOR_TYPE_STORAGE_BUFFER or VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, the offset member of each element of pBufferInfo must be a multiple of VkPhysicalDeviceLimits::minStorageBufferOffsetAlignment (https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkWriteDescriptorSet-descriptorType-00328)
#define GPU_BUFFER_BIG_MAGIC_GUARD_NBYTES	(2*128) // w.r.t. size of typical cache-line - 128 bytes
#define GPU_BUFFER_BIG_MAGIC_GUARD_NVALUES	(GPU_BUFFER_BIG_MAGIC_GUARD_NBYTES / sizeof(unsigned int))

typedef struct _cl_mem *cl_mem;

namespace avk2 {
	namespace raii {
		class BufferData;
	}
}

namespace gpu {

class shared_device_buffer {
public:
	shared_device_buffer();
	~shared_device_buffer();
	explicit shared_device_buffer(size_t size) : shared_device_buffer() { resize(size); }
	shared_device_buffer(const shared_device_buffer &other, size_t offset = 0);
	shared_device_buffer &operator= (const shared_device_buffer &other);

	void			swap(shared_device_buffer &other);
	void			reset();
	size_t			size() const;
	void			resize(size_t size);
	void			grow(size_t size, float reserveMultiplier=1.1f);
	bool 			isNull() const;

	void *					cuptr() const;
	cl_mem					clmem() const;
	size_t					cloffset() const;
	size_t					vkoffset() const;
	avk2::raii::BufferData	*vkBufferData() const;

	void 			write(const void *data, size_t size);
	void			write(const shared_device_buffer &buffer, size_t size);
	void			write(const shared_host_buffer &buffer, size_t size);
	void			write2D(size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);

	void			read(void *data, size_t size, ptrdiff_t offset = 0) const;
	void 			read2D(size_t spitch, void *dst, size_t dpitch, size_t width, size_t height) const;

	void 			copyTo(shared_device_buffer &that, size_t size) const;

	bool			checkMagicGuards(const std::string &info) const;

	static shared_device_buffer create(size_t size);

protected:
	void	incref();
	void	decref();

	static const unsigned int*	getMagicGuardBytesHostPtr();
	bool						checkMagicGuardBytes(const unsigned int* found_data, size_t nbytes, const std::string &info) const;
	void						writeMagicGuards();

	unsigned char *	buffer_;
	void *			data_;
	int				type_;
	size_t			size_;
	size_t			offset_;
	size_t			nbytes_guard_prefix_;
	size_t			nbytes_guard_suffix_;
};

template <typename T>
class shared_device_buffer_typed : public shared_device_buffer {
public:
	shared_device_buffer_typed() : shared_device_buffer() {}
	explicit shared_device_buffer_typed(size_t number) : shared_device_buffer() { resizeN(number); }
	shared_device_buffer_typed(const shared_device_buffer_typed &other, size_t offset) : shared_device_buffer(other, offset * sizeof(T)) {}
	explicit shared_device_buffer_typed(const shared_device_buffer &other) : shared_device_buffer(other) {}

	size_t			number() const		{ return size_ / elementSize(); }
	size_t			elementSize() const	{ return sizeof(T); }

	void			resizeN(size_t number)
	{
		this->resize(number * elementSize());
	}
	void			growN(size_t number, float reserveMultiplier=1.1f)
	{
		this->grow(number * elementSize(), reserveMultiplier);
	}

	T *				cuptr() const
	{
		return (T *) shared_device_buffer::cuptr();
	}

	void 			writeN(const T* data, size_t number)
	{
		this->write(data, number * elementSize());
	}

	void			readN(T* data, size_t number, size_t offset = 0) const
	{
		this->read(data, number * elementSize(), offset * elementSize());
	}

	std::vector<T>	readVector(size_t n = 0) const
	{
		if (n == 0) {
			n = number();
		}

		std::vector<T> data_cpu(n);
		readN(data_cpu.data(), n);
		return data_cpu;
	}

	void			copyToN(shared_device_buffer_typed<T> &that, size_t number) const {	this->copyTo(that, number * elementSize()); }

	void			fill(T value)
	{
		// TODO speedup this method with writing on GPU side (f.e. from kernel)
		std::vector<T> values(number(), value);
		this->writeN(values.data(), values.size());
	}

	static shared_device_buffer_typed<T> createN(size_t number)
	{
		shared_device_buffer_typed<T> res;
		res.resizeN(number);
		return res;
	}
};

typedef shared_device_buffer						gpu_mem_any;

typedef shared_device_buffer_typed<int8_t>			gpu_mem_8i;
typedef shared_device_buffer_typed<int16_t>			gpu_mem_16i;
typedef shared_device_buffer_typed<int32_t>			gpu_mem_32i;
typedef shared_device_buffer_typed<uint8_t>			gpu_mem_8u;
typedef shared_device_buffer_typed<uint16_t>		gpu_mem_16u;
typedef shared_device_buffer_typed<uint32_t>		gpu_mem_32u;
typedef shared_device_buffer_typed<float>			gpu_mem_32f;
typedef shared_device_buffer_typed<double>			gpu_mem_64f;

template <unsigned int NDim, unsigned int NFloat, unsigned int NUint>
class VertexGeneric {
public:
	static unsigned int getNDim()	{	return NDim;	}
	static unsigned int getNFloat()	{	return NFloat;	}
	static unsigned int getNUint()	{	return NUint;	}
private:
	// Note that we can't use std::vector - we need a plain structure
	// because of attributes' offsets calculations and Array-of-Structures memory transfers.
	// Note that we can't use plain float[], because C++ doesn't allow zero-sized plain-array.
	// Note that we can't use std::array<float, N> because sizeof(std::array<float, 0>) is non-zero (equal to 1 byte).
	// Note that we can't use struct Empty {}; + std::conditional<NFloat == 0, Empty, float[NFloat]>::type float_attributes; - because sizeof(Empty) > 0. Not so empty, huh?
	// We need to have no fields (for which array size is zero) - f.e. thanks to explicit declaration of specializations of VertexGeneric (i.e. <NDim, 0, 0>, <NDim, 0, NUint>, <NDim, NFloat, 0>, etc.).
	// Or we can use plain buffer array + reinterpreting its parts as three different arrays:
	unsigned char	plain_bytes_attributes_[NDim * sizeof(float) + NFloat * sizeof(float) + NUint * sizeof(unsigned int)];

	template <typename T, unsigned int N, unsigned int BytesOffset>
	T &attribute(size_t i)
	{
		rassert(i < N, 471563091);
		T *ptr = (T *) (plain_bytes_attributes_ + BytesOffset);
		return ptr[i];
	}
protected:
	float &			positionAttribute(size_t i)		{ return attribute<float,			NDim,	0>										(i);	}
	float &			floatAttribute(size_t i)		{ return attribute<float,			NFloat,	NDim*sizeof(float)>						(i);	}
	unsigned int &	uintAttribute(size_t i)			{ return attribute<unsigned int,	NUint,	NDim*sizeof(float)+NFloat*sizeof(float)>(i);	}
};

class Vertex3D : public VertexGeneric<3, 0, 0> {
public:
	void init(float x, float y, float z) {
		// location = 0
		this->positionAttribute(0) = x;
		this->positionAttribute(1) = y;
		this->positionAttribute(2) = z;
	}
};

class Vertex3DUV : public VertexGeneric<3, 2, 0> {
public:
	void init(float x, float y, float z, float u, float v) {
		// location = 0
		this->positionAttribute(0) = x;
		this->positionAttribute(1) = y;
		this->positionAttribute(2) = z;

		// location = 1
		this->floatAttribute(0) = u;
		this->floatAttribute(1) = v;
	}
};

typedef shared_device_buffer_typed<Vertex3D>		gpu_mem_vertices_xyz;
typedef shared_device_buffer_typed<Vertex3DUV>		gpu_mem_vertices_xyz_uv;
typedef shared_device_buffer_typed<point3u>			gpu_mem_faces_indices;

static_assert(sizeof(Vertex3D) == 3*sizeof(float), "1356123512");
static_assert(sizeof(Vertex3DUV) == 5*sizeof(float), "456341563245");

#define gpu_mem shared_device_buffer_typed

}
