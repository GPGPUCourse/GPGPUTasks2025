#include <libgpu/opencl/cl/common.cl>

__kernel void aplusb(
	__global const int *a,
	__global const int *b,
	__global       int *c,
	      unsigned int  n)
{
	unsigned int i = get_global_id(0);
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}