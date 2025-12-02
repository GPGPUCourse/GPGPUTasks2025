#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

static inline int clz32(uint x)
{
    if (x == 0u) return 32;
    return clz(x);
}

static inline int common_prefix(__global const uint* input_data, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;

    uint ci = input_data[i * 2];
    uint cj = input_data[j * 2];

    if (ci == cj) {
        uint di = i;
        uint dj = j;
        uint diff = di ^ dj;
        return 32 + clz32(diff);
    } else {
        uint diff = ci ^ cj;
        return clz32(diff);
    }
}

static inline int find_split(__global const uint* input_data,
    int first, int last, int N)
{
    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last)
        return first;

    // Prefix between first and last (уже с учётом индекса, если коды равны)
    int commonPrefix = common_prefix(input_data, N, first, last);

    int split = first;
    int step  = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = common_prefix(input_data, N, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

static inline AABBGPU get_simple_aabb(
    __global const uint* faces,
    __global const uint* vertices,
    int index)
{
    uint3 face = loadFace(faces, index);
    float3 v0 = loadVertex(vertices, face.x);
    float3 v1 = loadVertex(vertices, face.y);
    float3 v2 = loadVertex(vertices, face.z);

    AABBGPU aabb;
    aabb.min_x = min(v0.x, min(v1.x, v2.x));
    aabb.min_y = min(v0.y, min(v1.y, v2.y));
    aabb.min_z = min(v0.z, min(v1.z, v2.z));
    aabb.max_x = max(v0.x, max(v1.x, v2.x));
    aabb.max_y = max(v0.y, max(v1.y, v2.y));
    aabb.max_z = max(v0.z, max(v1.z, v2.z));
    return aabb;
}

__kernel void build_bvh_gpu(
    __global const uint* input_data,
    __global const uint* faces,
    __global const uint* vertices,
    __global       uint* output_triIndex,
    __global       uint* parents,
    __global       BVHNodeGPU* output_nodes,
                   int nfaces)
{
    // 2 * i value of element is morton code, 2 * i + 1 value is triIndex
    const unsigned int i = get_global_id(0);
    int nodes_cnt = nfaces * 2 - 1;
    if (i >= nodes_cnt) {
        return;
    }
    if (i >= nfaces - 1) {
        int original_index = i - (nfaces - 1);
        output_triIndex[original_index] = input_data[2 * original_index + 1];
        output_nodes[i].aabb = get_simple_aabb(faces, vertices, output_triIndex[original_index]);
        output_nodes[i].leftChildIndex = -1;
        output_nodes[i].rightChildIndex = -1;
        return;
    }
    int cpL = common_prefix(input_data, nfaces, i, i - 1);
    int cpR = common_prefix(input_data, nfaces, i, i + 1);

    // Direction of the range
    int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    int deltaMin = common_prefix(input_data, nfaces, i, i - d);
    int lmax = 2;

    while (common_prefix(input_data, nfaces, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    // Binary search to find exact range length
    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(input_data, nfaces, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j_index = i + l * d;
    int i_index = i;
    int first = min(i_index, j_index);
    int last  = max(i_index, j_index);
    int split = find_split(input_data, first, last, nfaces);
    int leftIndex;
    if (split == first) {
        // Range [first, split] has one primitive -> leaf
        leftIndex = (nfaces - 1) + split;
    } else {
        // Internal node
        leftIndex = split;
    }
    // Right child
    int rightIndex;
    if (split + 1 == last) {
        // Range [split+1, last] has one primitive -> leaf
        rightIndex = (nfaces - 1) + split + 1;
    } else {
        // Internal node
        rightIndex = split + 1;
    }
    output_nodes[i].leftChildIndex  = leftIndex;
    output_nodes[i].rightChildIndex = rightIndex;
    parents[leftIndex] = i;
    parents[rightIndex] = i;
}
