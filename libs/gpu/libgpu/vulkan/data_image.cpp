#include "data_image.h"

#include <utility>

#include "engine.h"

#include "../context.h"

#include "vulkan_api_headers.h"


avk2::raii::ImageData::ImageData(VkImage image, VmaAllocation image_allocation,
								 vk::raii::Sampler &&sampler,
								 vk::ImageLayout initial_layout, VkFlags aspect_flags, vk::Format format,
								 size_t width, size_t height, size_t channels)
{
	this->image_ = std::unique_ptr<vk::Image>(new vk::Image(image));
	this->allocation_ = image_allocation;
	this->sampler_ = std::unique_ptr<vk::raii::Sampler>(new vk::raii::Sampler(std::move(sampler)));
	this->current_layout_ = initial_layout;
	this->aspect_flags_ = aspect_flags;
	this->format_ = format;
	this->width_ = width;
	this->height_ = height;
	this->cn_ = channels;

	this->image_all_channels_2darray_view_ = createImageView(cn_, true);
	this->image_per_channel_2darray_view_.resize(cn_);
	this->image_per_channel_view_.resize(cn_);
	for (unsigned int c = 0; c < cn_; ++c) {
		image_per_channel_2darray_view_[c] = createImageView(c, true);
		image_per_channel_view_[c] = createImageView(c, false);
	}
}

avk2::raii::ImageData::~ImageData()
{
	gpu::Context context;
	rassert(context.type() == gpu::Context::TypeVulkan, 32151241223);
	vmaDestroyImage(context.vk()->getVma(), VkImage(*image_), allocation_);
}

std::unique_ptr<vk::raii::ImageView> avk2::raii::ImageData::createImageView(unsigned int c, bool image2DArray) {
	rassert(c < cn_ || c == cn_, 227386314);
	gpu::Context context;
	rassert(context.type() == gpu::Context::TypeVulkan, 213233292);
	vk::ImageSubresourceRange subresource;
	if (c == cn_) {
		subresource = vk::ImageSubresourceRange(vk::ImageAspectFlags(aspect_flags_), 0, 1, 0, cn_); // all channels
	} else {
		subresource = vk::ImageSubresourceRange(vk::ImageAspectFlags(aspect_flags_), 0, 1, c, 1); // only single channel
	}
	vk::ImageViewCreateInfo image_view_create_info(vk::ImageViewCreateFlags(), *image_,
												   image2DArray ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D,
												   format_,
												   {}, subresource);
	vk::raii::ImageView image_view = context.vk()->getDevice().createImageView(image_view_create_info);
	return std::unique_ptr<vk::raii::ImageView>(new vk::raii::ImageView(std::move(image_view)));
}

const vk::Image &avk2::raii::ImageData::getImage() const
{
	return *image_;
}

const vk::Sampler &avk2::raii::ImageData::getSampler()
{
	return *(*sampler_);
}

const vk::ImageView &avk2::raii::ImageData::getImageView()
{
	return *(*image_all_channels_2darray_view_);
}

const vk::ImageView &avk2::raii::ImageData::getImageChannelView(int c, bool image2DArray)
{
	if (image2DArray) {
		return *(*image_per_channel_2darray_view_[c]);
	} else {
		return *(*image_per_channel_view_[c]);
	}
}

VkFlags avk2::raii::ImageData::getAspectFlags() const
{
	return aspect_flags_;
}

vk::ImageLayout avk2::raii::ImageData::getCurrentLayout() const
{
	return current_layout_;
}

void avk2::raii::ImageData::transitionLayout(vk::ImageLayout old_layout, vk::ImageLayout new_layout)
{
	rassert(old_layout == getCurrentLayout(), 395634510);
	if (old_layout == new_layout) {
		return;
	}
	gpu::Context context;

	vk::raii::CommandBuffer command_buffer = context.vk()->createCommandBuffer();
	command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

	// see "Transition barrier masks" in https://vulkan-tutorial.com/resources/vulkan_tutorial_en.pdf
	vk::AccessFlags src_access_mask, dst_access_mask;
	vk::PipelineStageFlags src_stage_mask, dst_stage_mask;
	src_access_mask = vk::AccessFlagBits::eNone;
	dst_access_mask = vk::AccessFlagBits::eNone;
	src_stage_mask = vk::PipelineStageFlagBits::eAllCommands;
	dst_stage_mask = vk::PipelineStageFlagBits::eAllCommands;

	vk::ImageSubresourceRange subresource_range(vk::ImageAspectFlags(aspect_flags_), 0, 1, 0, nlayers());
	vk::ImageMemoryBarrier barrier(src_access_mask, dst_access_mask, old_layout, new_layout, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, *image_, subresource_range);

	command_buffer.pipelineBarrier(src_stage_mask, dst_stage_mask, vk::DependencyFlags(), nullptr, nullptr, barrier);

	command_buffer.end();

	context.vk()->submitCommandBuffer(command_buffer);

	current_layout_ = new_layout;
}

void avk2::raii::ImageData::updateLayout(vk::ImageLayout old_layout, vk::ImageLayout new_layout)
{
	rassert(current_layout_ == old_layout, 454856275);
	current_layout_ = new_layout;
}
