#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

static inline int delta(int i, int j, __global const PrimGPU* prims, const uint nFaces) {
    if (j < 0 || j >= nFaces) return -1;
    uint ci = prims[i].mortonCode;
    uint cj = prims[j].mortonCode;

    if (ci == cj) {
        return 32 + clz(i ^ j);
    } else {
        return clz(ci ^ cj);
    }
}

static inline AABBGPU prim_aabb(
    const PrimGPU prim,
    __global const float* vertices,
    __global const uint* faces) {
    uint3 face = loadFace(faces, prim.faceIndex);
    float3 u = loadVertex(vertices, face.x);
    float3 v = loadVertex(vertices, face.y);
    float3 w = loadVertex(vertices, face.z);
    AABBGPU result;
    result.min_x = min(min(u.x, v.x), w.x);
    result.min_y = min(min(u.y, v.y), w.y);
    result.min_z = min(min(u.z, v.z), w.z);
    result.max_x = max(max(u.x, v.x), w.x);
    result.max_y = max(max(u.y, v.y), w.y);
    result.max_z = max(max(u.z, v.z), w.z);
    return result;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void build_lbvh(
    __global const float* vertices,
    __global const uint* faces,
    const uint nFaces,
    __global const PrimGPU* sortedPrims,
    __global BVHNodeGPU* nodes,
    __global uint* leafIndices
) {
    const int nNodes = 2 * nFaces - 1;
    const int leafStart = nFaces - 1;
    const int i = get_global_id(0);
    if (i >= nNodes) return;
    if (i >= leafStart) {
        nodes[i].leftChildIndex = 0xFFFFFFFFu;
        nodes[i].rightChildIndex = 0xFFFFFFFFu;
        nodes[i].aabb = prim_aabb(sortedPrims[i - leafStart], vertices, faces);
        leafIndices[i - leafStart] = sortedPrims[i - leafStart].faceIndex;
        return;
    }

    const int deltaLeft = delta(i, i - 1, sortedPrims, nFaces);
    const int deltaRight = delta(i, i + 1, sortedPrims, nFaces);
    int dir = deltaLeft < deltaRight ? 1 : -1;
    const int deltaMin = delta(i, i - dir, sortedPrims, nFaces);
    int lUpperBound = 2;

    while (delta(i, i + lUpperBound * dir, sortedPrims, nFaces) > deltaMin) {
        lUpperBound *= 2;
    }

    int l = 0;
    for (int t = lUpperBound >> 1; t > 0; t >>= 1) {
        if (delta(i, i + (l + t) * dir, sortedPrims, nFaces) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * dir;

    int rangeLeft = min(i, j);
    int rangeRight = max(i, j);
    int deltaRange = delta(rangeLeft, rangeRight, sortedPrims, nFaces);

    int split = rangeLeft;
    int step = rangeRight - rangeLeft;

    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < rangeRight) {
            int splitDelta = delta(rangeLeft, newSplit, sortedPrims, nFaces);
            if (splitDelta > deltaRange) {
                split = newSplit;
            }
        }
    } while (step > 1);

    nodes[i].leftChildIndex = split + (split == rangeLeft ? leafStart : 0);
    nodes[i].rightChildIndex = split + 1 + (split + 1 == rangeRight ? leafStart : 0);
    nodes[nodes[i].leftChildIndex].parentIndex = i;
    nodes[nodes[i].rightChildIndex].parentIndex = i;
}