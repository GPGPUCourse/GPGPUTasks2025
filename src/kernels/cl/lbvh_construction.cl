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

inline int common_pref_len(
    __global const uint*       morton_codes,
    int l,
    int r, 
    int nfaces
) 
{
    if (l > r) {
        int tmp = r;
        r = l;
        l = tmp;
    }

    rassert(r == l + 1, 4358904);

    if (l < 0)
        return -1;

    if (r >= nfaces)
        return -1;

    int bit = 31;
    while (bit >= 0 && (((morton_codes[l] + l) >> bit) & 1) == (((morton_codes[r] + r) >> bit) & 1))
        bit--;
    
    return 32 - bit - 1;
}

__kernel void lbvh_construction(
    __global const uint*       morton_codes,
    __global uint*             face_indexes,
    __global const uint*       faces,
    __global const float*      vertices,
    __global BVHNodeGPU*       nodes,
    __global int*              parent,
    uint                       nfaces
   )
{
    int i = get_global_id(0);
    const int leafStart = (int)nfaces - 1; 

    int d, pref_len, j, min_pref, min_pref_ind;
    if (i < leafStart) {
        d = 0, pref_len = -1;

        if (i == 0)
            d = +1;
        else
            d = (common_pref_len(morton_codes, i, i + 1, nfaces) > common_pref_len(morton_codes, i - 1, i, nfaces) ? +1 : -1);

        pref_len = common_pref_len(morton_codes, i, i - d, nfaces);

        j = i;
        min_pref = 100, min_pref_ind = -1;
        while (j + d < nfaces &&
               j + d >= 0     &&
               pref_len < common_pref_len(morton_codes, j, j + d, nfaces)) { 
            if (min_pref > common_pref_len(morton_codes, j, j + d, nfaces)) {
                min_pref = common_pref_len(morton_codes, j, j + d, nfaces);
                min_pref_ind = j;
            }
            j += d;
        }
        // min_pref_ind, min_pref_ind + d - max on segment

        if (i == 0) {
            parent[0] = -1;
        }
        if (j == i + d) {
            nodes[i].leftChildIndex  = face_indexes[min(i, j)];
            nodes[i].rightChildIndex = face_indexes[max(i, j)];
        }
        else if (i == min_pref_ind) {
            if (d == +1) {
                nodes[i].leftChildIndex  = face_indexes[i];
                nodes[i].rightChildIndex = i + d;
            }
            if (d == -1) {
                nodes[i].leftChildIndex  = i + d;
                nodes[i].rightChildIndex = face_indexes[i];
            }
        }
        else if (min_pref_ind + d == j) { 
            if (d == +1) {
                nodes[i].leftChildIndex  = j - d;
                nodes[i].rightChildIndex = face_indexes[j];
            }
            if (d == -1) {
                nodes[i].leftChildIndex  = face_indexes[j];
                nodes[i].rightChildIndex = j - d;
            }
        }
        else {
            nodes[i].leftChildIndex  = min(min_pref_ind, min_pref_ind + d);
            nodes[i].rightChildIndex = max(min_pref_ind, min_pref_ind + d);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (i < leafStart) { 
        parent[nodes[i].leftChildIndex] = i;
        parent[nodes[i].rightChildIndex] = i;

        // atomic_add(&counter[nodes[i].leftChildIndex], 1);
        // atomic_add(&counter[nodes[i].rightChildIndex], 1);
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (i == 0)
        printf("tree build done\n");
}
