#pragma once

#include "libbase/platform.h"
#include "../../../base/libbase/runtime_assert.h"

#include "../context.h"

#ifdef CPU_ARCH_X86
typedef uint64_t VkImage;
#else
typedef struct VkImage_T* VkImage;
#endif

typedef struct VmaAllocation_T* VmaAllocation;
struct VmaAllocationInfo;

typedef uint32_t VkFlags;

namespace vk {
	class Image;
	class ImageView;
	class Sampler;
	enum class Format;
	enum class ImageLayout;
	enum class DescriptorType;
	namespace raii {
		class ImageView;
		class Sampler;
	}
}

namespace avk2 {
	namespace raii {
		// we use image2DArray to represent multi-channel image (including case cn=1)
		// note that ImageData can also represent image2D entity (that is more handful in case of working with depth/weight-images)
		// see also: VulkanEngine::createImage2D and VulkanEngine::createImage2DArray
		class ImageData {
		public:
			ImageData(VkImage image, VmaAllocation image_allocation,
					  vk::raii::Sampler &&sampler,
					  vk::ImageLayout initial_layout, VkFlags aspect_flags, vk::Format format,
					  size_t width, size_t height, size_t channels);
			~ImageData();

			const vk::Image &getImage() const;
			const vk::Sampler &getSampler();
			const vk::ImageView &getImageView();
			const vk::ImageView &getImageChannelView(int c, bool image2DArray=true);

			VkFlags getAspectFlags() const;
			vk::ImageLayout getCurrentLayout() const;

			size_t width() const		{	return width_;		}
			size_t height() const		{	return height_;		}
			size_t channels() const		{	return cn_;			}

			size_t nlayers() const		{	rassert(channels() >= 1, 720718172); return channels();	}

			// changes image layout via proper Vulkan call
			void transitionLayout(vk::ImageLayout old_layout, vk::ImageLayout new_layout);

			// doesn't change image layout - because it should be used if image layout already was changed as a side effect of previous calls
			// f.e. image can change its layout on Vulkan side because of using this image as attachment
			// in such case we should reflect this layout change on the CPU side (to make sanity checks possible) - using this method 
			void updateLayout(vk::ImageLayout old_layout, vk::ImageLayout new_layout);

		protected:
			std::unique_ptr<vk::raii::ImageView> createImageView(unsigned int c, bool image2DArray);

			std::unique_ptr<vk::Image>							image_;
			VmaAllocation										allocation_;
			std::unique_ptr<vk::raii::Sampler>					sampler_;
			std::unique_ptr<vk::raii::ImageView>				image_all_channels_2darray_view_;
			std::vector<std::unique_ptr<vk::raii::ImageView>>	image_per_channel_2darray_view_;
			std::vector<std::unique_ptr<vk::raii::ImageView>>	image_per_channel_view_;

			size_t									width_;
			size_t									height_;
			size_t									cn_;

			vk::ImageLayout							current_layout_;
			VkFlags									aspect_flags_;
			vk::Format								format_;
		};
	}
}