#include "data_buffer.h"

#include <utility>

#include "engine.h"

#include "../context.h"

#include "vulkan_api_headers.h"


avk2::raii::BufferData::BufferData(VkBuffer buffer, VmaAllocation buffer_allocation)
{
	this->buffer_ = std::unique_ptr<vk::Buffer>(new vk::Buffer(buffer));
	this->allocation_ = buffer_allocation;
	this->staging_alloc_info_.reset();
}

avk2::raii::BufferData::BufferData(VkBuffer buffer, VmaAllocation buffer_allocation, VmaAllocationInfo staging_alloc_info)
{
	this->buffer_ = std::unique_ptr<vk::Buffer>(new vk::Buffer(buffer));
	this->allocation_ = buffer_allocation;
	this->staging_alloc_info_ = std::unique_ptr<VmaAllocationInfo>(new VmaAllocationInfo(staging_alloc_info));
}

avk2::raii::BufferData::~BufferData()
{
	gpu::Context context;
	rassert(context.type() == gpu::Context::TypeVulkan, 439691531507893);
	vmaDestroyBuffer(context.vk()->getVma(), VkBuffer(*buffer_), allocation_);
}

vk::Buffer &avk2::raii::BufferData::getBuffer() const
{
	return *buffer_;
}

void *avk2::raii::BufferData::getMappedDataPointer()
{
	rassert(staging_alloc_info_, 155254647);
	return staging_alloc_info_->pMappedData;
}
