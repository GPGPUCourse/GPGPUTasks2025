Tutorials/Links
=======

- [How to setup Validation Layers](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers)
- [How to Learn Vulkan](http://web.archive.org/web/20240204071340/https://www.jeremyong.com/c++/vulkan/graphics/rendering/2018/03/26/how-to-learn-vulkan/)
- [Vulkan in 30 minutes](http://web.archive.org/web/20231123204000/https://renderdoc.org/vulkan-in-30-minutes.html)
- [Vulkan Memory Types on PC and How to Use Them](http://web.archive.org/web/20221118221047/https://asawicki.info/news_1740_vulkan_memory_types_on_pc_and_how_to_use_them)
- [Vulkan RAII programming guide](https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/vk_raii_ProgrammingGuide.md)
- [Vulkan RAII samples](https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/RAII_Samples)
- [About VkQueue families](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families#page_Queue-families)
- [Vulkan tutorial](https://vulkan-tutorial.com/resources/vulkan_tutorial_en.pdf)
- [Vulkan compute code example](https://github.com/mcleary/VulkanHpp-Compute-Sample) ([and article](https://bakedbits.dev/posts/vulkan-compute-example/))
- [VMA - Vulkan Memory Allocator](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/quick_start.html)
- [Vulkan RAII samples](https://github.com/jherico/VulkanExamples/tree/cpp/examples)
- [About Push Constants](https://vkguide.dev/docs/new_chapter_2/vulkan_pushconstants/) ([and another one](https://vkguide.dev/docs/chapter-3/push_constants/))
- [Structs Alignment Requirements](https://vulkan-tutorial.com/Uniform_buffers/Descriptor_pool_and_sets#page_Alignment-requirements)
- [Using Debug Printf](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/debug_printf.md)

Rasterization:

- []

Vulkan C API vs C++ RAII API examples comparison
======

Example 1:
------

```c++
// Vulkan C++ API:
vk::BufferCopy region_cpp;
command_buffer.copyBuffer(staging_buffer, vk_data->buffer, region_cpp);
```

```c
// Vulkan C API:
VkBufferCopy region = {0, 0, size};
VKF.vkCmdCopyBuffer(vk::CommandBuffer(command_buffer), staging_buffer, vk_data->buffer, 1, &region);
```

Note that ```VkBufferCopy``` is a struct, so it is easy to make a mistake and initialize it partially (leaving some fields with uninitialized trash):

In such case we initialized ```size``` field, but all other fields (offsets) were left uninitialized:

```
VkBufferCopy region;
region.size = size;
```

**Improvement 1:** This is not the case with C++ API, because this code ```vk::BufferCopy region_cpp;``` will call default constructor initializing all fields with zeros.

Note that in case of C API it is enough to write ```VkBufferCopy region = {};``` (but it is still more error-prone).

Example 2:
------

```c++
// Vulkan C++ API:
vk::raii::Instance temporary_instance = vk::raii::Instance(context, instance_create_info);
std::vector<vk::raii::PhysicalDevice> devices = temporary_instance.enumeratePhysicalDevices();
```

```c
// Vulkan C API:
VkInstance temporary_instance;
assert(VKF.vkCreateInstance(&instance_create_info, nullptr, &temporary_instance) == VK_SUCCESS);
unsigned int ndevices = 0;
assert(VKF.vkEnumeratePhysicalDevices(vk::Instance(temporary_instance), &ndevices, nullptr) == VK_SUCCESS);
std::vector<VkPhysicalDevice> devices(ndevices);
assert(VKF.vkEnumeratePhysicalDevices(vk::Instance(temporary_instance), &ndevices, devices.data()) == VK_SUCCESS);
VKF.vkDestroyInstance(temporary_instance);
```

**Improvement 2:** In case of C++ RAII API we don't need to call ```vkDestroyInstance(...)``` manually. Also it supports std containers like ```std::vector``` - see devices result from ```vkEnumeratePhysicalDevices```.

Interesting implementation details:
------

These three lines are equal thanks to syntax sugar inside Vulkan C++ API:

```c++
vk::raii::DescriptorSet descriptor_sets;
...
descriptor_writes[i] = vk::WriteDescriptorSet(descriptor_sets, binding, 0, 1, vk::DescriptorType::eStorageBuffer);
descriptor_writes[i] = vk::WriteDescriptorSet(descriptor_sets.operator vk::DescriptorSet(), binding, 0, 1, vk::DescriptorType::eStorageBuffer);
descriptor_writes[i] = vk::WriteDescriptorSet(vk::DescriptorSet(descriptor_sets), binding, 0, 1, vk::DescriptorType::eStorageBuffer);
```

I.e. ```vk::raii::DescriptorSet``` will be implicitly converted into ```vk::DescriptorSet```. So RAII boxes are-just-works! (without need of manual conversions)

**Improvement 3:** More pretty-looking and expressive enums: ```VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER``` -> ```vk::DescriptorType::eStorageBuffer```.

But there are also a case when some parameter is a pointer to multiple values, f.e.:

```c
// C API function:
void vkCmdBindDescriptorSets(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            layout,
    uint32_t                                    firstSet,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet*                      pDescriptorSets, // <- this argument is a pointer to multiple VkDescriptorSet values
    uint32_t                                    dynamicOffsetCount,
    const uint32_t*                             pDynamicOffsets);
```

**Improvement 4:** In such cases you can even pass ```std::vector<vk::DescriptorSet>``` as an argument:

```c++
vk::raii::DescriptorSet descriptor_sets_a = ...;
vk::raii::DescriptorSet descriptor_sets_b = ...;
std::vector<vk::DescriptorSet> descriptors;
descriptors.push_back(descriptor_sets_a);
descriptors.push_back(descriptor_sets_b);
command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, descriptors, nullptr);
```

And you can pass even a single (**non-RAII**) descriptor (it will be automatically boxed into ```vk::ArrayProxy```):

```c++
vk::raii::DescriptorSet descriptor_sets;
vk::DescriptorSet descriptor_sets_non_raii = descriptor_sets;
// These four lines are equal, but the 4-th one DOESN'T COMPILE
command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, vk::PipelineLayout(kernel->pipelineLayout()), 0, descriptor_sets_non_raii, nullptr);
command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, vk::PipelineLayout(kernel->pipelineLayout()), 0, vk::ArrayProxy<vk::DescriptorSet>(descriptor_sets_non_raii), nullptr);
command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, vk::PipelineLayout(kernel->pipelineLayout()), 0, vk::ArrayProxy<vk::DescriptorSet>(descriptor_sets), nullptr);
command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, vk::PipelineLayout(kernel->pipelineLayout()), 0, descriptor_sets, nullptr); // this line does not compile!
```

The last one line doesn't compile, because it requires two implicit conversions: ```vk::raii::DescriptorSet``` -> ```vk::DescriptorSet``` -> ```vk::ArrayProxy```.

Example of SPIRV-Reflect output from kernel
=======

```glsl
#version 450

#include <libgpu/vulkan/vk/common.vk>

#include "defines.h"

layout (local_size_x = VK_GROUP_SIZE) in;

layout (std430, binding = 0) readonly buffer AsIn {
	uint as[];
};

layout (std430, binding = 1) readonly buffer BsIn {
	uint bs[];
};

layout (std430, binding = 2) writeonly buffer CsOut {
	uint cs[];
};

layout (push_constant) uniform PushConstants {
	uint n;
	vec3 tmp;
} params;

void main()
{
	const uint i = gl_GlobalInvocationID.x;
	if (i < params.n) {
		cs[i] = as[i] + bs[i];
	}
}
```

If we will compile this kernel into SPIRV and then launch SPIRV-Reflect:

```/opt/bin/spirv-reflect .../source/libgpu/vulkan/tests/kernels/generated_kernels/aplusb_spirv_vulkan.spir```

```
generator       : Google Shaderc over Glslang
source lang     : Unknown
source lang ver : 0
source file     : 
entry point     : main (stage=CS)
local size      : (256, 1, 1)


  Input variables: 1

    0:
      spirv id  : 11
      location  : (built-in) GlobalInvocationId
      type      : uint3
      semantic  : 
      name      : 
      qualifier : 


  Push constant blocks: 1

    0:
      spirv id : 21
      name     : <unnamed>
          // size = 32, padded size = 32
          struct <unnamed> {
              uint   ; // abs offset =  0, rel offset =  0, size =  4, padded size = 16
              float3 ; // abs offset = 16, rel offset = 16, size = 12, padded size = 16 UNUSED
          } <unnamed>;



  Descriptor bindings: 3

    Binding 0.0
      spirv id : 39
      set      : 0
      binding  : 0
      type     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER (SRV)
      count    : 1
      accessed : true
      name     : <unnamed>
          // size = 0, padded size = 0
          struct <unnamed> {
              uint ; // abs offset = 0, rel offset = 0, size = 0, padded size = 0
          } <unnamed>;


    Binding 0.1
      spirv id : 47
      set      : 0
      binding  : 1
      type     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER (SRV)
      count    : 1
      accessed : true
      name     : <unnamed>
          // size = 0, padded size = 0
          struct <unnamed> {
              uint ; // abs offset = 0, rel offset = 0, size = 0, padded size = 0
          } <unnamed>;


    Binding 0.2
      spirv id : 34
      set      : 0
      binding  : 2
      type     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER (UAV)
      count    : 1
      accessed : true
      name     : <unnamed>
          // size = 0, padded size = 0
          struct <unnamed> {
              uint ; // abs offset = 0, rel offset = 0, size = 0, padded size = 0
          } <unnamed>;
```

Notes
=======

- ```VkInstance``` - each is isolated and don't know anything about any another VkInstance. 
It is constructed with list of required ```layers``` and ```extensions```. Can enumerate available GPUs (```VkPhysicalDevice```).
- ```VkDevice``` - can be created from the ```VkPhysicalDevice```. Represents "I am running Vulkan on this GPU". The same as OpenGL context.

Images and Buffers
------

- ```VkImage``` - it's usage should be specified at construction: is it a color attachment, or sampled image in shader, or image load/store, etc. Tiling of ```VkImage``` can be ```Linear``` (plain linear data layout in memory, limited support of image types) or ```Optimal``` (affects is it directly readable/writable, available only in ```device-local``` memory). Used via ```VkImageView```.
- ```VkImageView``` - thin wrapper around ```VkImage```. A description of what array slices or mip levels are visible, and optionally a different (but compatible) format (like aliasing a UNORM texture as UINT).
- ```VkBuffer``` - plain memory buffer (just specify size and a usage). Used directly. 

Allocating GPU Memory
------

```vkGetPhysicalDeviceMemoryProperties()``` - reports one or more available ```memory heaps```. F.e. two ```heaps``` can be **system RAM** and **GPU VRAM**. Each heap contains one or more ```memory types```, each with different properties:
- is it ```host-visible```? (f.e. ```staging resources``` will need to be in host visible memory)
- is it ```device-local```? (required for ```Optimal tiling``` of ```VkImage```)
- is it coherent between GPU and CPU access? (means that when CPU changes data - GPU will see these changes if synchronized properly, i.e. explicit flushing with ```vkFlushMappedMemoryRanges/vkInvalidateMappedMemoryRanges``` is not required)
- is it cached or uncached? (only ```host-visible``` memory can be ```cached```)

```vkAllocateMemory()``` - allocates memory on ```VkDevice``` (with specified ```heap``` and ```type```) and returns ```VkDeviceMemory```.

```vkMapMemory()/vkUnmapMemory()``` - can be used on ```host-visible``` memory to map for update. Mapped data can be in-use (for changes too) by GPU, but you need to have proper synchronization. CPU shouldn't write to memory when it is used by GPU.

Binding Memory
------

- ```vkGet{Buffer|Image}MemoryRequirements``` - will report memory requirements of ```VkBuffer```/```VkImage``` w.r.t. its properties like ```Optimal tiling```, required padding, alignment between mips, etc. Also it will report bitmask of ```memory types``` requirements (f.e. ```Optimal tiling``` can be allocated only on ```device-local```).
- ```vkBind{Buffer|Image}Memory``` - should be done before usage.

Command buffers
------

- ```VkQueue``` - we use queue that supports graphics commands - see [about VkQueue families](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families#page_Queue-families)
- ```VkCommandBuffer``` - allocated from ```VkCommandPool``` (```vkAllocateCommandBuffers()``` can be used from thread's pool without synchronization). They are submitted to ```VkQueue``` (in most cases it is single per device).
- ```vkQueueSubmit``` - can submit several ```VkCommandBuffer``` into ```VkQueue``` at once. Their execution can be reordered and/or overlap (w.r.t. synchronization).

Shaders Compilation and Pipeline State Objects
------

- ```VkPipeline``` - bakes a lot of state, but viewport/stencil masks/blend constants/etc. can be changed dynamically. In ```vkCreateGraphicsPipelines()``` you specify which states are dynamic and others are baked from values of PSO creation info.
- ```VkPipelineCache``` - can be used to save some data from compiled ```VkPipeline``` to compile it faster in the future. Requires custom **versioning**.
- ```VkShaderModule``` - created from SPIR-V module. Can include multiple entry-points, one particular entry point will be chosen on ```VkPipeline``` creation.

[glslang](https://github.com/KhronosGroup/glslang) can be used to compile SPIR-V from GLSL.

Binding Model
------

- each shader has its one namespace - pixel shader texture binding 0 is not vertex shader texture binding 0
- the base binding unit is a ```descriptor``` - one bind that can be an image, a sampler, a uniform/constant buffer, etc. (could also be arrayed - so you can have an array of images that can be different sizes)
- ```descriptors``` are bound in blocks in a ```VkDescriptorSet``` (and each type is described with ```VkDescriptorSetLayout```)
- ```descriptors``` are allocated from ```VkDescriptorPool``` (useful to have a pool-per-thread like with ```VkCommandPool```)

```c++
VkDescriptorSetLayoutBinding bindings[] = {
	// binding 0 is a UBO, array size 1, visible to all stages
	{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL_GRAPHICS, NULL },
	// binding 1 is a sampler, array size 1, visible to all stages
	{ 1, VK_DESCRIPTOR_TYPE_SAMPLER,        1, VK_SHADER_STAGE_ALL_GRAPHICS, NULL },
	// binding 5 is an image, array size 10, visible only to fragment shader
	{ 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 10, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
};
```

```glsl
#version 430
layout(set = 0, binding = 0) uniform MyUniformBufferType {
    // ...
} MyUniformBufferInstance;
layout(set = 0, binding = 1) sampler MySampler;
layout(set = 0, binding = 5) uniform image2D MyImages[10];
```

Synchronisation
------

- ```VkMemoryBarrier``` - global memory barrier
- ```Vk{Buffer|Image}MemoryBarrier``` - apply to specific resources (or their subsections), example: ```VkImageMemoryBarrier``` has ```srcAccessMask = ACCESS_COLOR_ATTACHMENT_WRITE``` and ```dstAccessMask = ACCESS_SHADER_READ``` - i.e. all color writes should finish before any shader reads begin

Image Barriers and Layouts
------

- ```VkImageMemoryBarrier``` - can specify transition of image from one layout to another
- ```GENERAL layout``` - is legal to use for anything but might not be optimal (and there are optimal layouts for color attachment, depth attachment, shader sampling, etc.)
- ```UNDEFINED layout``` - is used as initial one, should be changed before any GPU usage (there are also alternative initial layout - ```PREINITIALIZED layout```)
- ```UNDEFINED layout``` - can be specified as previous layout for transition (i.e. "I don't care what the image was like before, throw it away and use it like this")

Kernel compilation and execution
------

- Kernel is compiled lazily (at the first launch w.r.t. passed arguments)
- How descriptors are specified? Descriptors list is built w.r.t. passed arguments
- What if on later launches different arguments (in terms of their descriptors type) will be passed? This is, probably, an incorrect usage - so error will be raised

- ```KernelSource::getKernel``` loads SPIR-V bytecode (via bytes gzip-unpacking)
- ```VulkanKernel``` compiles kernel from SPIR-V bytecode w.r.t. descriptors list
- TODO WHERE DESCRIPTORS LIST GENERATED?

Render passes
------

- ```VkFramebuffer``` - is a set of ```VkImageViews```
- each subpass from ```VkRenderPass``` also specify an action both for loading and storing each attachment (the depth should be cleared to 1.0, but the color can be initialised to garbage for all I care - I'm going to fully overwrite the screen in this pass)

Rendering
------

 - https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#renderpass
 - https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#drawing

GLSL structs and alignment
------

 - Matrix constructors - https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)#Matrix_constructors
 - We use ```std430``` instead of ```std140``` - because in ```std140``` float array (used in FourierCorrectionVk) is x4 times bigger due to 16 bytes strides - https://www.reddit.com/r/vulkan/comments/u5jiws/glsl_std140_layout_for_arrays_of_scalars/
 - Matrix (```mat3/mat4/mat3x4```) - column-major format, so we need to transpose data on GPU matrices initialization from CPU matrices - https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)#Matrices
 - ```mat3x3``` is in fact in size like 3x4-floats, so we need to mimic this with ```matrix3x4``` (with zero last column)
 - ```vec3``` has auto-alignment to 16-byte boundaries, so it is handful to add dummy ```float``` field (or use it as another field, note that this is impossible with ```std140```)

TODOs
------

- ```TODO``` add rasserts for GPU
- ```TODO``` compile_vulkan and compile_opencl should migrate into libgpu/...
- ```TODO``` opencl_program.h.in and vulkan_program.h.in should migrate into libgpu/...
- ```TODO``` use [SPIRV reflect](https://github.com/KhronosGroup/SPIRV-Reflect) to validate that everything is fine (like number of uniforms/bindings/etc.)
- ```TODO``` VulkanKernel/VulkanKernelArg: add support for image2DArray, uimage2DArray, image2D, uimage2D, iimage2D, sampler2D
- ```TODO``` implement automatic subdivision in avk2::KernelSource::exec()
- ```TODO``` add caching of kernels compilations (including persistent cache, just like for OpenCL)
- ```TODO``` make VULKAN_TIMEOUT_NANOSECS configurable via tweak, try to make it small/big, etc.
- ```TODO``` VulkanEngine::readBuffer and VulkanEngine::writeBuffer are still ~10-20% slower than in OpenCL - we can try to speedup them with triple-buffering
- ```TODO``` implement non-blocking async method returning custom fence class that can be used for blocking wait - i.e. we will use submitCommandBufferAsync in methods like readBuffer/writeBuffer and readImage/writeImage
- ```TODO``` KernelSource::execRender - vk::raii::Pipeline should be cached (assuming any property haven't changed) - and it should be easy to enable "re-create Pipeline on each call" for debug purposes
- ```TODO``` try to improve performance with non-general layout, but it seems to be not important on NVIDIA and AMD desktop GPUs
- ```TODO``` try to use vk::CullModeFlagBits to speedup rendering when back-faces can be skipped (see also https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/ and https://github.com/gpuweb/gpuweb/issues/416)
- ```TODO``` we can try to use persistent vk::raii::PipelineCache for faster first rendering - can be important for multiprocess cluster processing
- ```TODO``` add info in readme about screen (and texture) coordinate systems - see also https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/ and https://github.com/gpuweb/gpuweb/issues/416
- ```TODO``` add info in readme about int8<->float conversion rules - see https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap3.html#fundamentals-fixedfpconv
