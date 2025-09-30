#pragma once

#include "libbase/platform.h"
#include "../../../base/libbase/runtime_assert.h"

#include "../context.h"

#ifdef CPU_ARCH_X86
typedef uint64_t VkBuffer;
#else
typedef struct VkBuffer_T* VkBuffer;
#endif

typedef struct VmaAllocation_T* VmaAllocation;
struct VmaAllocationInfo;

namespace vk {
	class Buffer;
}

namespace avk2 {
	namespace raii {
		class BufferData {
		public:
			BufferData(VkBuffer buffer, VmaAllocation buffer_allocation);
			BufferData(VkBuffer buffer, VmaAllocation buffer_allocation, VmaAllocationInfo staging_alloc_info);
			~BufferData();

			vk::Buffer &getBuffer() const;
			void *getMappedDataPointer();

		protected:
			std::unique_ptr<vk::Buffer>			buffer_;
			VmaAllocation						allocation_;
			std::unique_ptr<VmaAllocationInfo>	staging_alloc_info_;
		};
	}
}