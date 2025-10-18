#version 450

#define NCHANNELS 4
#define REQUIRE_TEMPLATE_NCHANNELS // but in this case we specify NCHANNELS with exact value (in general we generate kernel with each possible NCHANNELS value)
#include <libgpu/vulkan/vk/common.vk>

#include "defines.h"

//layout(binding = 0)	uniform		sampler2DArray	inputImage; // TODO try

layout(location = 0)	in			vec3			inProjection;

LAYOUT_NCHANNELS(0,									outColor);

void main()
{
	float res[NCHANNELS];
	res[0] = BLENDING_RED_VALUE;
	res[1] = BLENDING_GREEN_VALUE;
	res[2] = BLENDING_BLUE_VALUE;
	res[3] = BLENDING_ALPHA_VALUE;
	ASSIGN_NCHANNELS(outColor, res);
}