#version 450

#define NCHANNELS 3
#define REQUIRE_TEMPLATE_NCHANNELS // but in this case we specify NCHANNELS with exact value (in general we generate kernel with each possible NCHANNELS value)
#include <libgpu/vulkan/vk/common.vk>

#include "defines.h"

//layout(binding = 0)	uniform		sampler2DArray	inputImage; // TODO try

layout(location = 0)	in			vec3			inProjection;

LAYOUT_NCHANNELS(0,									outColor);
layout(location = 0+NCHANNELS)	out	int				outFaceIdx;

layout (push_constant) uniform PushConstants {
		int	nfaces;
} params;

void main()
{
	int face_id = gl_PrimitiveID;
	rassert(face_id >= 0 && face_id < params.nfaces, 675459179);

	vec2 pt0 = vec2(0.5f, 0.5f); // the upper-left pixel center
//	rassert(239.1f == texture(inputImage, vec3(pt0 / textureSize(inputImage, 0).xy, 0)).r, 57464908);
	float res[NCHANNELS];
	for (int c = 0; c < NCHANNELS; ++c) {
		res[c] = (SOME_COLOR_VALUE + c) / 255.0f; // about int8<->float conversion rules - see https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap3.html#fundamentals-fixedfpconv
	}
	ASSIGN_NCHANNELS(outColor, res);
	outFaceIdx = face_id;
}