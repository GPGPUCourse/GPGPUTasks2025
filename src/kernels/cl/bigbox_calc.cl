#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

// BVH traversal: closest hit along ray
__kernel void bigbox_calc(    
    uint                      nfaces,
    __global const float*     vertices,
    __global const uint*      faces,
    __global AABBGPU*         bigbox)
{
    uint i = get_global_id(0);

    if (i == 0) {
        bigbox[0].min_x = +INFINITY;
        bigbox[0].min_y = +INFINITY;
        bigbox[0].min_z = +INFINITY;
        bigbox[0].max_x = -INFINITY;
        bigbox[0].max_y = -INFINITY;
        bigbox[0].max_z = -INFINITY;

        for (int j = 0; j < nfaces; j++) {
            uint3  f  = loadFace(faces, j);
            float3 v0 = loadVertex(vertices, f.x);
            float3 v1 = loadVertex(vertices, f.y);
            float3 v2 = loadVertex(vertices, f.z);

            bigbox[0].min_x = min(bigbox[0].min_x, min(v0.x, min(v1.x, v2.x)));
            bigbox[0].min_y = min(bigbox[0].min_y, min(v0.y, min(v1.y, v2.y)));
            bigbox[0].min_z = min(bigbox[0].min_z, min(v0.z, min(v1.z, v2.z)));
            bigbox[0].max_x = max(bigbox[0].max_x, min(v0.x, min(v1.x, v2.x)));
            bigbox[0].max_y = max(bigbox[0].max_y, min(v0.y, min(v1.y, v2.y)));
            bigbox[0].max_z = max(bigbox[0].max_z, min(v0.z, min(v1.z, v2.z)));
        }

        
        printf("Biggest box : (%f %f %f) -- (%f %f %f)\n", bigbox[0].min_x,
                                             bigbox[0].min_y,
                                             bigbox[0].min_z,
                                             bigbox[0].max_x,
                                             bigbox[0].max_y,
                                             bigbox[0].max_z);
    }
}
