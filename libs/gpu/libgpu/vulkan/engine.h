#pragma once

#include "data_buffer.h"
#include "data_image.h"
#include "device.h"

#include <libbase/point.h>
#include <libbase/platform.h>
#include <libbase/string_utils.h>
#include <map>
#include <vector>
#include <memory>
#include <string.h>
#include <unordered_map>

#include "../work_size.h"
#include "../shared_device_buffer.h"
#include "../shared_device_image.h"

#include "spirv_reflect/shader_module_info.h"

#define CLEAR_DEPTH_FRAMEBUFFER_WITH_MAX_VALUE		1.0f

namespace vk {
	enum class DescriptorType;
	union ClearValue;
	struct VertexInputBindingDescription;
	struct VertexInputAttributeDescription;

	namespace raii {
		class Context;
		class Instance;
		class PhysicalDevice;
		class Device; // The same as OpenGL context, represents "I am running Vulkan on this GPU".
		class Queue; // In most cases it is single per device

		class DescriptorPool; // Used to allocate Descriptor
		class CommandPool; // Used to allocate VkCommandBuffer
		class CommandBuffer;

		class ShaderModule;
		class DescriptorSetLayout; // TODO ???
		class PipelineLayout; // TODO ???
		class Pipeline; // Bakes a lot of state, but viewport/stencil masks/blend constants/etc. can be changed dynamically

//		class DescriptorPool; // Used to allocate descriptors

//		class VkCommandBuffer; // They are submitted to VkQueue for execution

//		class VkImage; 
//		class VkImageView; // Thin wrapper around VkImage, a description of what array slices or mip levels are visible, and optionally a different (but compatible) format (like aliasing a UNORM texture as UINT)
//		class VkBuffer; // Plain memory buffer

//		class VkShaderModule; // Created from SPIR-V module, can include multiple entry-points, one particular entry point will be chosen on VkPipeline creation
//		class VkPipeline; // Bakes a lot of state, but viewport/stencil masks/blend constants/etc. can be changed dynamically

		class Fence;
	}
}
struct VmaAllocator_T;
typedef struct VmaAllocator_T* VmaAllocator;
struct VkDebugUtilsMessengerEXT_T;

struct RENDERDOC_API_1_6_0;

#ifdef CPU_ARCH_X86
typedef uint64_t VkDebugUtilsMessengerEXT;
#else
typedef VkDebugUtilsMessengerEXT_T *VkDebugUtilsMessengerEXT;
#endif

namespace vk {
	struct ApplicationInfo;
	class DescriptorSetLayout;
	enum class Format;
	namespace raii {
		class Context;
		class Instance;
		class DescriptorSet;
	}
}

namespace avk2 {
	vk::ApplicationInfo					createAppInfo();
	vk::raii::Instance					createInstance(const vk::raii::Context &context, bool enable_validation_layers=false);
	bool								isMoltenVK();

	class InstanceContext {
	public:
		InstanceContext(bool enable_validation_layers=false);
		~InstanceContext();

		size_t													getConstructionIndex()	{	return construction_index_;	}
		vk::raii::Context										&context()				{	return *context_;			}
		vk::raii::Instance										&instance()				{	return *instance_;			}

		bool													isDebugCallbackTriggered()					{	return is_debug_callback_triggered_;		}
		void													setDebugCallbackTriggered(bool triggered)	{	is_debug_callback_triggered_ = triggered;	}

		// vkContext and vkInstance should be re-used - so they are lazily inited in this Singleton, otherwise #45 call to avk2::createInstance is followed with empty instance.enumeratePhysicalDevices() on some systems
		static std::shared_ptr<InstanceContext>					getGlobalInstanceContext(bool enable_validation_layers=false);
		static void												clearGlobalInstanceContext();

		static void												renderDocStartCapture(const std::string &title);
		static void												renderDocEndCapture();

		// while we should be able to work with the same vkContext and vkInstance (i.e. with avk2::InstanceContext) from multiple threads
		// this still can lead to race-condition problems on some systems (f.e. we encountered vk::InitializationFailedError - PhysicalDevice::createDevice: ErrorInitializationFailed)
		// so in case of emergency - use this (and probably replace std::mutex with std::recursive_mutex):
		static std::unique_ptr<std::lock_guard<std::mutex>>		getGlobalLock();
	protected:
		static std::vector<std::shared_ptr<avk2::InstanceContext>>	global_instance_contexts_;

		static bool													global_rdoc_checked;
		static RENDERDOC_API_1_6_0*									global_rdoc_api;

		size_t										construction_index_;
		std::shared_ptr<vk::raii::Context>			context_;
		std::shared_ptr<vk::raii::Instance>			instance_;
		std::shared_ptr<VkDebugUtilsMessengerEXT>	debug_messenger_;
		bool										is_debug_callback_triggered_;
	};

	class VulkanEngine;
	class VkContext;

	typedef std::shared_ptr<VulkanEngine>	sh_ptr_vk_engine;

	class PushConstant {
	public:
		PushConstant()
		{
			size_ = 0;
			ptr_ = nullptr;
		}

		template <typename T>
		PushConstant(const T &value)
		{
			size_ = sizeof(T);

			// note that we don't know about lifetime of passed into RenderBuilder::params(T&) reference T &value,
			// so we need to copy its data so that data will be still alive when it will be used at render time
			data_.resize(size_);
			memcpy(data_.data(), &value, size_);

			ptr_ = (const void*) data_.data();
		}

		// trying to detect erroneous usage of KernelSource::exec()
		PushConstant(const gpu::WorkSize &value) = delete;

		bool isEmpty()		const { return size_ == 0; }
		const void *ptr()	const { return ptr_;       }
		size_t size()		const { return size_;      }

	protected:
		std::vector<char>	data_;
		const void			*ptr_;
		size_t				size_;
	};

	class ImageAttachment {
	public:
		// usage of this constructor means that attachment will not be cleared (will be used as is - i.e. with vk::AttachmentLoadOp::eLoad)
		ImageAttachment(gpu::shared_device_image &image, bool is_color);

		// see VkClearValue (union with VkClearColorValue and VkClearDepthStencilValue)
		ImageAttachment(gpu::shared_device_image &image, point4<float>		clear_color_value);
		ImageAttachment(gpu::shared_device_image &image, point4<int>		clear_color_value);
		ImageAttachment(gpu::shared_device_image &image, point4<unsigned>	clear_color_value);
		ImageAttachment(gpu::shared_device_image &image, float				clear_depth_value, unsigned int	clear_stencil_value=0);

		gpu::shared_device_image	&getImage() const				{ return image_;        }
		bool						hasClearValue() const			{ return (bool) clear_value_;  }
		vk::ClearValue				&getClearValue() const			{ rassert(hasClearValue(), 311187706); return *clear_value_; }
		vk::ClearValue				getChannelColorClearValue(unsigned int c) const;
		bool						isColorAttachment() const		{ return is_color_;     }
		bool						isDepthStencilAttachment() const{ return !is_color_;    }

	protected:
		gpu::shared_device_image			&image_;
		std::shared_ptr<vk::ClearValue>		clear_value_;
		bool								is_color_;
	};

	class DepthAttachment : public ImageAttachment {
	public:
		// usage of this constructor means that attachment will not be cleared (will be used as is - i.e. with vk::AttachmentLoadOp::eLoad)
		DepthAttachment(gpu::shared_device_image &image) : ImageAttachment(image, false) {}

		// this constructor specifies clear-on-load values for depth and stencil
		// in most cases 1.0 should be used for depth, because depth range is [0; 1] (due to low adoption of VK_EXT_depth_range_unrestricted extension at the moment)
		DepthAttachment(gpu::shared_device_image &image, float clear_depth_value, unsigned int clear_stencil_value=0) : ImageAttachment(image, clear_depth_value, clear_stencil_value) {}
	};

	class VertexInput {
	public:
		VertexInput() : stride_(0), attribute_ndim_(0), attribute_nfloat_(0), attribute_nuint_(0), n_(0)
		{}

		template <typename VertexGenericType>
		VertexInput(const gpu::shared_device_buffer_typed<VertexGenericType> &buffer)
			: buffer_(buffer), stride_(buffer.elementSize()), attribute_ndim_(VertexGenericType::getNDim()), attribute_nfloat_(VertexGenericType::getNFloat()), attribute_nuint_(VertexGenericType::getNUint()), n_(buffer.number())
		{}

		vk::VertexInputBindingDescription					buildBindingDescription() const;
		std::vector<vk::VertexInputAttributeDescription>	buildAttributeDescriptions() const;
		vk::Buffer											buffer() const;

		bool								isNull() const		{ return buffer_.isNull();         }
		size_t								stride() const		{ return stride_;                  }
		size_t								nvertices() const	{ return buffer_.size() / stride_; }
		size_t								offset() const		{ return buffer_.vkoffset();       }

	protected:
		gpu::shared_device_buffer			buffer_;
		size_t								stride_;
		unsigned int						attribute_ndim_;
		unsigned int						attribute_nfloat_;
		unsigned int						attribute_nuint_;
		size_t								n_;
	};

	class IndicesInput {
	public:
		IndicesInput() : with_adjacency_(false), n_indices_(0), n_faces_(0)
		{}

		IndicesInput(const gpu::gpu_mem_32u &buffer, bool with_adjacency=false) : buffer_(buffer), with_adjacency_(with_adjacency), n_indices_(buffer.number())
		{
			if (with_adjacency) {
				rassert(n_indices_ % 6 == 0, 318799886);
				n_faces_ = n_indices_ / 6;
			} else {
				rassert(n_indices_ % 3 == 0, 318799883);
				n_faces_ = n_indices_ / 3;
			}
		}

		IndicesInput(const gpu::gpu_mem_faces_indices &buffer) : buffer_(buffer), with_adjacency_(false), n_indices_(3 * buffer.number())
		{
			n_faces_ = buffer.number();
		}

		bool								isWithAdjacency() const	{ return with_adjacency_;  }
		bool								isNull() const			{ return buffer_.isNull(); }
		size_t								nindices() const		{ return n_indices_;       }
		size_t								nfaces() const			{ return n_faces_;         }
		size_t								offset() const			{ return buffer_.vkoffset();}
		vk::Buffer							buffer() const;

	protected:
		gpu::shared_device_buffer			buffer_;
		bool								with_adjacency_;
		size_t								n_indices_;
		size_t								n_faces_;
	};

	class VulkanKernelArg {
	public:
		VulkanKernelArg() : is_null(true), buffer(nullptr) { }

		// TODO support image2DArray, uimage2DArray, image2D, uimage2D, iimage2D, sampler2D

		VulkanKernelArg(const gpu::shared_device_buffer &arg);
		VulkanKernelArg(const gpu::shared_device_image &arg);

		void init();

		bool							is_null;
		const gpu::shared_device_buffer	*buffer;
		const gpu::shared_device_image	*image;
	};

	class VulkanKernel {
	public:
		VulkanKernel(const std::string &program_name)
			: program_name_(program_name)
		{}
		~VulkanKernel();

		void create(vk::raii::DescriptorSetLayout &&descriptor_set_layout, vk::raii::DescriptorSetLayout &&descriptor_set_layout_for_rassert, vk::raii::PipelineLayout &&pipeline_layout, vk::raii::Pipeline &&pipeline, avk2::ShaderModuleInfo &&shader_module_info);

		typedef VulkanKernelArg Arg;

		vk::raii::DescriptorSetLayout&			descriptorSetLayout()			{ return *descriptor_set_layout_.get();            }
		vk::raii::PipelineLayout&				pipelineLayout()				{ return *pipeline_layout_.get();                  }
		vk::raii::Pipeline&						pipeline()						{ return *pipeline_.get();                         }
		const avk2::ShaderModuleInfo&			shaderModuleInfo()				{ return *shader_module_info_.get();               }

		gpu::gpu_mem_32u&						rassertCodeAndLineBuffer()		{ return rassert_code_and_line_;                   }
		bool									isRassertUsed()					{ return is_rassert_used_;                         }
		void									checkRassertCode();
		const std::vector<vk::DescriptorType>	&getDescriptorTypes()			{ return descriptor_types_;                        }
		bool									isImageArrayed(unsigned int set, unsigned int binding);

	protected:
		std::shared_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout_;
		std::shared_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout_rassert_;
		std::shared_ptr<vk::raii::PipelineLayout> pipeline_layout_;
		std::shared_ptr<vk::raii::Pipeline> pipeline_;
		std::shared_ptr<ShaderModuleInfo> shader_module_info_;

		std::string									program_name_;
		bool										is_rassert_used_;
		gpu::gpu_mem_32u							rassert_code_and_line_;
		std::vector<vk::DescriptorType>				descriptor_types_;
	};

	class VulkanEngine {
	public:
		VulkanEngine();
		~VulkanEngine();

		void							init(uint64_t vk_device_id, bool enable_validation_layers);

		avk2::raii::BufferData*			createBuffer(size_t size);
		void							writeBuffer(const avk2::raii::BufferData &buffer_dst, size_t offset, size_t size, const void *src);
		void							readBuffer(const avk2::raii::BufferData &buffer_src, size_t offset, size_t size, void *dst);

		avk2::raii::ImageData*			createDepthImage(unsigned int width, unsigned int height);
		avk2::raii::ImageData*			createImage2DArray(unsigned int width, unsigned int height, size_t cn, DataType data_type);

		template <typename T>
		void							writeImage(const avk2::raii::ImageData &image_dst, const TypedImage<T> &src);
		void							writeImage(const avk2::raii::ImageData &image_dst, const AnyImage &src);
		template <typename T>
		void							readImage(const avk2::raii::ImageData &image_src, const TypedImage<T> &dst);
		void							readImage(const avk2::raii::ImageData &image_src, const AnyImage &dst);

		const Device &					device() const			{	return device_;		}
		std::map<int, VulkanKernel *> &	kernels()	{	return kernels_;	}

		VulkanKernel *					findKernel(int id) const;
		void							clearKernel(int id);
		void							clearKernels();
		void							clearStagingBuffers();

		std::shared_ptr<VkContext>		avk2_context_;

		vk::raii::Context &						getContext();
		vk::raii::Instance &					getInstance();
		vk::raii::PhysicalDevice &				getPhysicalDevice();
		vk::raii::Device &						getDevice();
		VmaAllocator &							getVma();
		const unsigned int &					getQueueFamilyIndex();
		vk::raii::Queue &						getQueue();

		vk::raii::CommandBuffer					createCommandBuffer();
		void									submitCommandBuffer(const vk::raii::CommandBuffer &command_buffer);
		std::shared_ptr<vk::raii::Fence>		submitCommandBufferAsync(const vk::raii::CommandBuffer &command_buffer);
		vk::raii::DescriptorSet					allocateDescriptor(vk::raii::DescriptorSetLayout& descriptor_set_layout, const std::vector<vk::DescriptorType> &descriptor_types);

	protected:
		avk2::raii::ImageData*			createImage2D(unsigned int width, unsigned int height, vk::Format format);
		avk2::raii::ImageData*			createImage2DArray(unsigned int width, unsigned int height, size_t cn, vk::Format format);

		void							allocateStagingWriteBuffers();
		void							allocateStagingReadBuffers();

		vk::raii::DescriptorPool &		getDescriptorPool();
		vk::raii::CommandPool &			getCommandPool();

		// TODO we can try to speedup it a bit more via triple-buffering
		std::unique_ptr<avk2::raii::BufferData>		staging_read_buffers_[2];
		std::unique_ptr<avk2::raii::BufferData>		staging_write_buffers_[2];
		// we are using context from the single thread, but let's add additional guarantees in case of multi-threading single-GPU processing
		// note that in such case it will be benefitial to use thread-local staging buffers
		Mutex										staging_read_buffers_mutex_;
		Mutex										staging_write_buffers_mutex_;

		std::map<int, VulkanKernel *>	kernels_;

		uint64_t						vk_device_id_;

		Device							device_;
	};

	class VersionedBinary {

	public:
		VersionedBinary(const char *data, const size_t size);

		const char *			data() const				{ return data_;					}
		size_t					size() const				{ return size_;					}

	protected:
		const char *			data_;
		const size_t			size_;
	};

	class ProgramBinaries {
	public:
		ProgramBinaries(std::vector<const VersionedBinary *> binaries, std::string program_name);

		int										id() const { return id_; }
		const VersionedBinary*					getBinary() const;
		const std::string &						programName() const { return program_name_; };
		bool									isProgramNameEndsWith(const std::string &suffix) const;

	protected:
		int										id_;
		std::vector<const VersionedBinary *>	binaries_;
		std::string								program_name_;
	};

	class KernelSource;

	class RenderBuilder {
		friend class KernelSource;
	public:
		RenderBuilder(KernelSource &kernel, size_t width, size_t height);

		template <typename VertexGenericType>
		RenderBuilder &geometry(const gpu::shared_device_buffer_typed<VertexGenericType> &vertices,
								const gpu::gpu_mem_32u &faces)
		{
			return geometryHelper(avk2::VertexInput(vertices), avk2::IndicesInput(faces));
		}

		template <typename VertexGenericType>
		RenderBuilder &geometry(const gpu::shared_device_buffer_typed<VertexGenericType> &vertices,
								const gpu::gpu_mem_faces_indices &faces)
		{
			return geometryHelper(avk2::VertexInput(vertices), avk2::IndicesInput(faces));
		}

		// in most cases clear_depth_value=CLEAR_DEPTH_FRAMEBUFFER_WITH_MAX_VALUE=1.0 should be used for clear_depth_value,
		// because depth range is [0; 1] (due to low adoption of VK_EXT_depth_range_unrestricted extension at the moment)
		// note that we assume that depth framebuffer attachment is always (if presented) in ZERO attachment slot, so to make it more explicit - depth() should be called BEFORE any addAttachment()
		// note that depth framebuffer is optional
		RenderBuilder &setDepthAttachment(gpu::shared_device_depth_image &depth_buffer, float clear_depth_value, unsigned int clear_stencil_value=0); // depth framebuffer attachment will be used with clear-on-load values for depth and stencil specified
		RenderBuilder &setDepthAttachment(gpu::shared_device_depth_image &depth_buffer, bool depth_write_enable=false); // depth framebuffer attachment will not be cleared (will be used as is - i.e. with vk::AttachmentLoadOp::eLoad) + we can disable depth updating

		// note that there can be any number of attachments - from zero and up to device/driver specific limit
		// note that we assume that depth framebuffer attachment is always (if presented) in ZERO attachment slot, so to make it more explicit - depth() should be called BEFORE any addAttachment()
		RenderBuilder &addAttachment(gpu::shared_device_image &color_attachment); // this image attachment will not be cleared (will be used as is - i.e. with vk::AttachmentLoadOp::eLoad)
		template <typename T>
		RenderBuilder &addAttachment(gpu::shared_device_image_typed<T> &color_attachment, const T &single_channel_value); // this single channel image attachment will be used with clear-on-load value
		template <typename T>
		RenderBuilder &addAttachment(gpu::shared_device_image_typed<T> &color_attachment, const point4<T> &clear_value); // this image attachment will be used with clear-on-load values
		RenderBuilder &addAttachment(gpu::shared_device_image_typed<unsigned char> &color_attachment, const point4uc &clear_value); // this image attachment will be used with clear-on-load values

		// note that blending can be enabled on per-attachment basis (and so can be a parameter in addAttachment method),
		// but it seems a bit niche use-case, so until it will be really needed - let's make it more trivial and explicit (enabling blending for all color attachment)
		RenderBuilder &setColorAttachmentsBlending(bool enabled) { color_attachments_blending_enabled_ = enabled; return *this; }

		RenderBuilder &setWireframeMode(bool enabled) { polygon_mode_wireframe_ = enabled; return *this; }

		template <typename T>
		RenderBuilder &params(const T &value)
		{
			rassert(push_constant_.isEmpty(), 176904413); // check for uniqueness of builder's geometry(...) call
			push_constant_ = avk2::PushConstant(value); // note that we don't know about lifetime of passed reference T &value, so we need to copy its data so that data will be still alive when it will be used at render time
			return *this;
		}

		typedef VulkanKernelArg Arg;

		void exec(const Arg &arg0 = Arg(), const Arg &arg1 = Arg(), const Arg &arg2 = Arg(), const Arg &arg3 = Arg(), const Arg &arg4 = Arg(), const Arg &arg5 = Arg(), const Arg &arg6 = Arg(), const Arg &arg7 = Arg(), const Arg &arg8 = Arg(), const Arg &arg9 = Arg(), const Arg &arg10 = Arg(), const Arg &arg11 = Arg(), const Arg &arg12 = Arg(), const Arg &arg13 = Arg(), const Arg &arg14 = Arg(), const Arg &arg15 = Arg(), const Arg &arg16 = Arg(), const Arg &arg17 = Arg(), const Arg &arg18 = Arg(), const Arg &arg19 = Arg(), const Arg &arg20 = Arg(), const Arg &arg21 = Arg(), const Arg &arg22 = Arg(), const Arg &arg23 = Arg(), const Arg &arg24 = Arg(), const Arg &arg25 = Arg(), const Arg &arg26 = Arg(), const Arg &arg27 = Arg(), const Arg &arg28 = Arg(), const Arg &arg29 = Arg(), const Arg &arg30 = Arg(), const Arg &arg31 = Arg(), const Arg &arg32 = Arg(), const Arg &arg33 = Arg(), const Arg &arg34 = Arg(), const Arg &arg35 = Arg(), const Arg &arg36 = Arg(), const Arg &arg37 = Arg(), const Arg &arg38 = Arg(), const Arg &arg39 = Arg(), const Arg &arg40 = Arg());

	protected:
		RenderBuilder &geometryHelper(const avk2::VertexInput &vertices, const avk2::IndicesInput &faces);
		RenderBuilder &addDepthAttachmentHelper(const avk2::DepthAttachment &depth_attachment);
		RenderBuilder &addColorAttachmentHelper(const avk2::ImageAttachment &color_attachment);

		KernelSource	&kernel_;

		unsigned int	viewport_width_;
		unsigned int	viewport_height_;

		bool			faces_with_adjacency_; // see https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#drawing-triangle-lists-with-adjacency
		bool 			polygon_mode_wireframe_;

		// note that sadly we can't use viewport depth with [0-MAX_FLT] range due to absence of support for VK_EXT_depth_range_unrestricted on MacOS
		// see https://github.com/KhronosGroup/MoltenVK/issues/1576 and https://vulkan.gpuinfo.org/listdevicescoverage.php?platform=macos&extension=VK_EXT_depth_range_unrestricted
		// also it is not supported on Intel Arc, f.e. on Intel A580 - https://vulkan.gpuinfo.org/displayreport.php?id=27375#device
		float			viewport_min_depth_;
		float			viewport_max_depth_;

		bool			depth_write_enable_; // it can be useful to disable writes to depth-buffer, while depth test is enabled (to test depth of each fragment VS pre-rendered depth-buffer)

		bool			color_attachments_blending_enabled_;

		avk2::VertexInput						geometry_vertices_;
		avk2::IndicesInput						geometry_indices_;

		avk2::PushConstant						push_constant_;

		std::vector<avk2::ImageAttachment>		depth_and_color_attachments_;
	};

	class KernelSource {
		friend class RenderBuilder;
	public:
		// TODO split into Compute and Render kernel
		KernelSource(const ProgramBinaries &compute_shader_program);
		KernelSource(const std::vector<const ProgramBinaries*> &graphic_shaders_programs);
		~KernelSource();

		typedef VulkanKernel::Arg Arg;

		bool isCompute() const			{ return shaders_programs_.size() == 1; }
		bool isRasterization() const	{ return shaders_programs_.size()  > 1; }

		// TODO split into Compute and Render kernel
		void exec(const PushConstant &params, const gpu::WorkSize &ws, const Arg &arg0 = Arg(), const Arg &arg1 = Arg(), const Arg &arg2 = Arg(), const Arg &arg3 = Arg(), const Arg &arg4 = Arg(), const Arg &arg5 = Arg(), const Arg &arg6 = Arg(), const Arg &arg7 = Arg(), const Arg &arg8 = Arg(), const Arg &arg9 = Arg(), const Arg &arg10 = Arg(), const Arg &arg11 = Arg(), const Arg &arg12 = Arg(), const Arg &arg13 = Arg(), const Arg &arg14 = Arg(), const Arg &arg15 = Arg(), const Arg &arg16 = Arg(), const Arg &arg17 = Arg(), const Arg &arg18 = Arg(), const Arg &arg19 = Arg(), const Arg &arg20 = Arg(), const Arg &arg21 = Arg(), const Arg &arg22 = Arg(), const Arg &arg23 = Arg(), const Arg &arg24 = Arg(), const Arg &arg25 = Arg(), const Arg &arg26 = Arg(), const Arg &arg27 = Arg(), const Arg &arg28 = Arg(), const Arg &arg29 = Arg(), const Arg &arg30 = Arg(), const Arg &arg31 = Arg(), const Arg &arg32 = Arg(), const Arg &arg33 = Arg(), const Arg &arg34 = Arg(), const Arg &arg35 = Arg(), const Arg &arg36 = Arg(), const Arg &arg37 = Arg(), const Arg &arg38 = Arg(), const Arg &arg39 = Arg(), const Arg &arg40 = Arg());
		RenderBuilder initRender(size_t width, size_t height);

		std::string getProgramName() const			{ return shaders_programs_[0]->programName(); }

		double getLastExecTotalTime() const			{ return last_exec_total_time_; }
		double getLastExecPrepairingTime() const	{ return last_exec_prepairing_time_; }
		double getLastExecGPUTime() const			{ return last_exec_gpu_time_; }

	protected:
		// launchRender(...) should be called via RenderBuilder constructed in public initRender(...)
		void launchRender(const RenderBuilder &params, const Arg &arg0 = Arg(), const Arg &arg1 = Arg(), const Arg &arg2 = Arg(), const Arg &arg3 = Arg(), const Arg &arg4 = Arg(), const Arg &arg5 = Arg(), const Arg &arg6 = Arg(), const Arg &arg7 = Arg(), const Arg &arg8 = Arg(), const Arg &arg9 = Arg(), const Arg &arg10 = Arg(), const Arg &arg11 = Arg(), const Arg &arg12 = Arg(), const Arg &arg13 = Arg(), const Arg &arg14 = Arg(), const Arg &arg15 = Arg(), const Arg &arg16 = Arg(), const Arg &arg17 = Arg(), const Arg &arg18 = Arg(), const Arg &arg19 = Arg(), const Arg &arg20 = Arg(), const Arg &arg21 = Arg(), const Arg &arg22 = Arg(), const Arg &arg23 = Arg(), const Arg &arg24 = Arg(), const Arg &arg25 = Arg(), const Arg &arg26 = Arg(), const Arg &arg27 = Arg(), const Arg &arg28 = Arg(), const Arg &arg29 = Arg(), const Arg &arg30 = Arg(), const Arg &arg31 = Arg(), const Arg &arg32 = Arg(), const Arg &arg33 = Arg(), const Arg &arg34 = Arg(), const Arg &arg35 = Arg(), const Arg &arg36 = Arg(), const Arg &arg37 = Arg(), const Arg &arg38 = Arg(), const Arg &arg39 = Arg(), const Arg &arg40 = Arg());

        void dispatchAutoSubdivided(const vk::raii::CommandBuffer &command_buffer, const gpu::WorkSize &ws) const;

		int getNextKernelId();

		void										init();

		static bool									parseArg(std::vector<avk2::KernelSource::Arg> &args, const Arg &arg);
		static std::vector<avk2::KernelSource::Arg>	parseArgs(const Arg &arg0, const Arg &arg1, const Arg &arg2, const Arg &arg3, const Arg &arg4, const Arg &arg5, const Arg &arg6, const Arg &arg7, const Arg &arg8, const Arg &arg9, const Arg &arg10, const Arg &arg11, const Arg &arg12, const Arg &arg13, const Arg &arg14, const Arg &arg15, const Arg &arg16, const Arg &arg17, const Arg &arg18, const Arg &arg19, const Arg &arg20, const Arg &arg21, const Arg &arg22, const Arg &arg23, const Arg &arg24, const Arg &arg25, const Arg &arg26, const Arg &arg27, const Arg &arg28, const Arg &arg29, const Arg &arg30, const Arg &arg31, const Arg &arg32, const Arg &arg33, const Arg &arg34, const Arg &arg35, const Arg &arg36, const Arg &arg37, const Arg &arg38, const Arg &arg39, const Arg &arg40);

		static vk::raii::ShaderModule				createShaderModule(const std::shared_ptr<VulkanEngine> &vk, const ProgramBinaries &program, avk2::ShaderModuleInfo *shader_module_info_output=nullptr);

		VulkanKernel								*getKernel(const std::shared_ptr<VulkanEngine> &vk);
		VulkanKernel								*compileComputeKernel(const std::shared_ptr<VulkanEngine> &vk);
		VulkanKernel								*compileRasterizationKernel(const std::shared_ptr<VulkanEngine> &vk);

		std::vector<const ProgramBinaries*>			shaders_programs_;

		double										last_exec_total_time_;
		double										last_exec_prepairing_time_;
		double										last_exec_gpu_time_;

		int				id_;
		std::string		name_;
	};

	class KernelsProfiler {
	public:
		KernelsProfiler(bool accumulate_per_kernel_times=false)
		: accumulate_per_kernel_times_(accumulate_per_kernel_times),
		  kernels_total_(0.0), kernels_prepairing_(0.0), kernels_gpu_(0.0)
		{}

		void addKernelProfilingTimes(const avk2::KernelSource &kernel);

		void printSlowestKernels(size_t top_k=5) const;

		double getTotal() const					{ return kernels_total_; }

		double getPrepairing() const			{ return kernels_prepairing_; }
		double getGPU() const					{ return kernels_gpu_; }

		std::string getPrepairingPercent() const{ return to_percent(getPrepairing(), getTotal()); }
		std::string getGPUPercent() const		{ return to_percent(getGPU(), getTotal()); }

	protected:
		std::unordered_map<std::string, double>	per_kernel_total_sum_by_name_;
		bool									accumulate_per_kernel_times_;

		double									kernels_total_;
		double									kernels_prepairing_;
		double									kernels_gpu_;
	};

}
