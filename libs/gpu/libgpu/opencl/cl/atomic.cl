#ifndef atomic_cl // pragma once
#define atomic_cl

#ifndef HOST_CODE
#include "clion_defines.cl"
#endif

STATIC_KEYWORD void atomic_add_f32_local(volatile __local float *address, float value)
{
	float old = value;
	while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);
}

STATIC_KEYWORD void atomic_add_f32(volatile __global float *address, float value)
{
	float old = value;
	while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);
}

STATIC_KEYWORD float atomic_cmpxchg_f32(volatile __global float *p, float cmp, float val) {
	union {
		unsigned int	u32;
		float			f32;
	} cmp_union, val_union, old_union;

	cmp_union.f32 = cmp;
	val_union.f32 = val;
	old_union.u32 = atomic_cmpxchg((volatile __global unsigned int *) p, cmp_union.u32, val_union.u32);
	return old_union.f32;
}

STATIC_KEYWORD float atomic_cmpxchg_float(volatile __global float *p, float cmp, float val) {
	return atomic_cmpxchg_f32(p, cmp, val);
}

STATIC_KEYWORD unsigned int atomic_cmpxchg_uint(volatile __global uint *p, uint cmp, uint val) {
	return atomic_cmpxchg(p, cmp, val);
}

STATIC_KEYWORD void atomic_max_f32(volatile __global float *p, const float val) {
	float cmp = *p;
	while (val > cmp) {
		float old = atomic_cmpxchg_float(p, cmp, val);
		if (old == cmp) {
			break;
		} else {
			cmp = old;
		}
	}
}

STATIC_KEYWORD void atomic_min_f32(volatile __global float *p, const float val) {
	float cmp = *p;
	while (val < cmp) {
		float old = atomic_cmpxchg_float(p, cmp, val);
		if (old == cmp) {
			break;
		} else {
			cmp = old;
		}
	}
}

#endif // pragma once
