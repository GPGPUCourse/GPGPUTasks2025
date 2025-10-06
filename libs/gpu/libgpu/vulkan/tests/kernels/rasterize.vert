#version 450

#include <libgpu/vulkan/vk/common.vk>

// layout(binding  = 0)	uniform		sampler2DArray	inputImage; // TODO try

layout(location = 0)	in			vec3			inPos; // vertex attributes

layout(location = 0)	out			vec3			outProjection;

void main()
{
	vec2 pt0 = vec2(0.5f, 0.5f); // the upper-left pixel center
//	rassert(239.1f == texture(inputImage, vec3(pt0 / textureSize(input_image, 0).xy, 0)).r, 47464908);
	vec3 screenPos = inPos;
	outProjection = screenPos; // screen position is in [0, 1] range (+X is directed to the right, +Y is looking down, i.e. origin is in the top-left corner)
	gl_Position = vec4(toGLPositionRange(screenPos.xy), screenPos.z, 1.0);
	// gl_FragDepth = gl_Position.z; - note that gl_Position.z will be automatically assigned to gl_FragDepth
}
