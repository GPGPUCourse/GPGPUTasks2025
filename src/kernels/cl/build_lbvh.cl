#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "helpers/rassert.cl"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
build_lbvh(
    __global const uint *mortons_codes,
    uint nfaces,
    __global BVHNodeGPU *lbvh)
{
    const int index = get_global_id(0);
    if (index > nfaces - 2)
    {
        return;
    }
    int diff = common_bits_from(mortons_codes, nfaces, index, index + 1) - common_bits_from(mortons_codes, nfaces, index, index - 1);
    rassert(diff != 0, 42);
    int direction = diff > 0 ? 1 : -1;
    int dmin = common_bits_from(mortons_codes, nfaces, index, index - direction);
    int lmax = 2;
    while (common_bits_from(mortons_codes, nfaces, index, index + lmax * direction) > dmin)
    {
        lmax *= 2;
    }
    uint l = 0;
    for (int t = lmax / 2; t > 0; t /= 2)
    {
        if (common_bits_from(mortons_codes, nfaces, index, index + (l + t) * direction) > dmin)
        {
            l += t;
        }
    }
    int j = index + l * direction;
    rassert(j < nfaces, 12345);
    int dnode = common_bits_from(mortons_codes, nfaces, index, j);
    int s = 0;
    for (uint w = 2; ((l + w - 1) / w) > 0; w *= 2)
    {
        uint t = ((l + w - 1) / w);
        if (common_bits_from(mortons_codes, nfaces, index, index + (s + t) * direction) > dnode)
        {
            s += t;
        }
    }
    int y = index + s * direction + min(direction, 0);
    rassert(y < nfaces, 1234);
    int left, right;
    if (min(index, j) == y)
    {
        left = (nfaces - 1) + y;
    }
    else
    {
        left = y;
    }

    if (max(index, j) == y + 1)
    {
        right = (nfaces - 1) + y + 1;
    }
    else
    {
        right = y + 1;
    }
    rassert(left < 2 * nfaces - 1, 123456);
    rassert(right < 2 * nfaces - 1, 123457);
    lbvh[index].leftChildIndex = left;
    lbvh[index].rightChildIndex = right;
}
