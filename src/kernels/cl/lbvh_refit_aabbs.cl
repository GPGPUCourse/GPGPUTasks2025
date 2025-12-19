#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

static inline AABBGPU aabb_union(AABBGPU a, AABBGPU b)
{
    AABBGPU r;
    r.min_x = fmin(a.min_x, b.min_x);
    r.min_y = fmin(a.min_y, b.min_y);
    r.min_z = fmin(a.min_z, b.min_z);
    r.max_x = fmax(a.max_x, b.max_x);
    r.max_y = fmax(a.max_y, b.max_y);
    r.max_z = fmax(a.max_z, b.max_z);
    return r;
}

__kernel void lbvh_refit_aabbs(
    __global BVHNodeGPU* nodes,
    __global const int* parent,
    __global uint* visit,
    uint nfaces)
{
    int i = (int)get_global_id(0);
    int n = (int)nfaces;
    if (i >= n) return;

    int leafStart = n - 1;
    int node = leafStart + i;

    int p = parent[node];
    while (p >= 0) {
        uint old = atomic_inc(&visit[p]);
        if (old == 0) return;

        int lc = (int)nodes[p].leftChildIndex;
        int rc = (int)nodes[p].rightChildIndex;

        nodes[p].aabb = aabb_union(nodes[lc].aabb, nodes[rc].aabb);

        p = parent[p];
    }
}
