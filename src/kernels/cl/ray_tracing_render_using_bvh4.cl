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

#ifndef NULL
#define NULL 0L
#endif

%:ifndef __GFX11__
// BVH traversal: closest hit along ray
static inline bool bvh_closest_hit(
    const float3              orig,
    const float3              dir,
    const size_t              bvh_base,
    const unsigned            root,
    const unsigned            bvh_stack_base,
    float                     tMin,
    __private float*          outT, // сюда нужно записать t рассчитанный в intersect_ray_triangle(..., t, u, v)
    __global BVHTriangleGPU* __private* outTri)
{
    unsigned stack[64];
    unsigned size = 1;
    stack[0] = root;
    float tOut = FLT_MAX;
    size_t max_iters = 1000;
    while(size && max_iters --> 0) {
        unsigned at = stack[--size];
        if(at % 8 == 5) {
            __global BVHBoxGPU* box = (__global BVHBoxGPU*)(bvh_base + (at - 5) * 8);
            for(size_t i = 0; i < 4; i++) {
                if(box->children[i] == -1u)
                    continue;
                AABBGPU aabb = box->coords[i];
                float tNear, tFar;
                if(intersect_ray_aabb(orig, dir, aabb, tMin, tOut, &tNear, &tFar)) {
                    stack[size++] = box->children[i];
                }
            }
        } else {
            __global BVHTriangleGPU* tri = (__global BVHTriangleGPU*)(bvh_base + at * 8);
            float3 a = loadVertex(tri->a, 0);
            float3 b = loadVertex(tri->b, 0);
            float3 c = loadVertex(tri->c, 0);
            float t, u, v;
            if(intersect_ray_triangle(orig, dir, a, b, c, tMin, tOut, false, &t, &u, &v))
            {
                tOut = t;
                *outTri = tri;
            }
        }
    }
    *outT = tOut;
    return tOut != FLT_MAX;
}

// Cast a single ray and report if ANY occluder is hit (for ambient occlusion)
static inline bool any_hit_from(
    const float3              orig,
    const float3              dir,
    const size_t              bvh_base,
    const unsigned            root,
    const unsigned            bvh_stack_base)
{
    unsigned stack[64];
    unsigned size = 1;
    stack[0] = root;
    size_t max_iters = 1000;
    while(size && max_iters --> 0) {
        unsigned at = stack[--size];
        if(at % 8 == 5) {
            __global BVHBoxGPU* box = (__global BVHBoxGPU*)(bvh_base + (at - 5) * 8);
            for(size_t i = 0; i < 4; i++) {
                if(box->children[i] == -1u)
                    continue;
                AABBGPU aabb = box->coords[i];
                float tNear, tFar;
                if(intersect_ray_aabb_any(orig, dir, aabb, &tNear, &tFar))
                    stack[size++] = box->children[i];
            }
        } else {
            __global BVHTriangleGPU* tri = (__global BVHTriangleGPU*)(bvh_base + at * 8);
            float3 a = loadVertex(tri->a, 0);
            float3 b = loadVertex(tri->b, 0);
            float3 c = loadVertex(tri->c, 0);
            float t, u, v;
            if(intersect_ray_triangle_any(orig, dir, a, b, c, false, &t, &u, &v))
                return true;
        }
    }
    return false;
}

%:else

uint4 get_texture(size_t bvh, unsigned root, bool nearest) {
    unsigned long long descr = 0; // 128-bit in opencl
    rassert(bvh % 256 == 0, 1337);
    rassert(root % 8 == 5, 1338);
    descr = bvh >> 8;
    unsigned long long sort = nearest ? 0 : 2; // undocumented parameter: box sorting heuristic (0/1/2)
    descr |= sort << 53; 
    descr |= 0ull << 55; // 0 ULP box intersection allowance (up to 255)
    descr |= 1ull << 63; // sort AABBs in output
    descr |= ((unsigned long long)root / 8 + 1) << 64; // size. docs explicitly say it should be size-1 but it breaks in that case
    descr |= 0ull << 119; // ??? related to handling instance nodes ???
    descr |= 0ull << 120; // no barycentric coordinates for triangle returns (we don't use them)
    descr |= 0ull << 121; // ??? something cache-related ???
    descr |= 0ull << 123; // 64kb pages not guaranteed
    descr |= 8ull << 124; // texture type: BVH
    return as_uint4(descr);
}

unsigned get_parent(const size_t bvh_base, unsigned elt) {
    if(elt % 8 == 5) {
        __global BVHBoxGPU* box = (__global BVHBoxGPU*)(bvh_base + 8 * (elt - 5));
        return box->reserved[2];
    }
    else {
        __global BVHTriangleGPU* box = (__global BVHTriangleGPU*)(bvh_base + 8 * elt);
        return box->id;
    }
}

// BVH traversal: closest hit along ray
static inline bool bvh_closest_hit(
    float3                    orig,
    const float3              dir,
    const size_t              bvh_base,
    const unsigned            root,
    const unsigned            bvh_stack_base,
    float                     tMin,
    __private float*          outT, // сюда нужно записать t рассчитанный в intersect_ray_triangle(..., t, u, v)
    __global BVHTriangleGPU* __private* outTri) // тут ещё были outU и outV, но мы ими не пользуемся. (по идее для их вычисления достаточно битового флага в дескрипторе)
{
    orig += dir * tMin;
    uint4 descr = get_texture(bvh_base, root, true);
    float4 origp = (float4)(orig, 0); // почему-то llvm требует 4 компонента и игнорирует последний
    float4 dirp = (float4)(dir, 0);
    float4 inv_dirp = (float4)(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z, 0);
    
    unsigned stk = bvh_stack_base;
    const unsigned NO_PUSH = 0xfffffffe; // undocumented magic value -- disables pushing when passed as last_visited
    float tOut = FLT_MAX;
    unsigned at = root;
    unsigned i;
    unsigned best_at = 0;
    uint last_visited = 0xffffFFFF;
    for(i = 0; i < 10000; i++) {
        // XXX: decreasing tOut when shorter matches are found causes problems if we recompute a node due to stack overflow
        // - if a sorting mode varies based on on tOut, we may skip the same AABB twice
        // - if the AABB is now too far to reach, we apparently get varying outcomes:
        //   - sometimes bvh_stack_rtn treats an unknown last_visited as -1, and we effectively restart processing the node
        //     (this will eventually halt, as tOut and the number of remaining AABBs have decreased)
        //   - sometimes bvh_stack_rtn ignores the remaining AABBs and we leave the node without recursing into them
        //   (it is likely that bvh_stack_rtn is not random and I'm misinterpreting something)
        // it appears that only sorting mode 0 (closest) avoids these issues at depth 8 and 16
        // how AMD drivers deal with this remains a mystery
        uint4 hw = __builtin_amdgcn_image_bvh_intersect_ray(at, tOut, origp, dirp, inv_dirp, descr);
        if(at % 8 == 0) {
            //printf("tri %d res %d\n", at, hw.w);
            float tNum = as_float(hw.x), tDenom = as_float(hw.y);
            if(hw.w && (tOut * tDenom < tNum) == (tDenom < 0)) {
                tOut = tNum / tDenom;
                best_at = at;
            }
            last_visited = NO_PUSH;
        } else {
            //printf("box %d res %d %d %d %d\n", at, hw.x, hw.y, hw.z, hw.w);
        }
        hw = (uint4)(hw.w, hw.z, hw.y, hw.x); // why AMD drivers do not need this remains a mystery
        //printf("at %u stk %d lv %d: hw %d %d %d %d\n", at, stk - bvh_stack_base, last_visited, hw.x, hw.y, hw.z, hw.w);
        unsigned old_stk = stk;
        uint2 r = __builtin_amdgcn_ds_bvh_stack_rtn(stk, last_visited, hw, STACK_SHIFT << 12);
        
        if(r.x == -1) {
            r.x = get_parent(bvh_base, at);
            if(r.x == 0) // root
                break;
            r.y++;
            // XXX ^ looks like sometimes the decrement is needed? we have stk continuously increasing during the traversal on small stacks
            // apparently this is what AMD drivers do though
            // if this somehow happens 2^18 times in one traversal and it overflows into stack address, we may have memory corruption
            last_visited = at;
        } else {
            last_visited = 0xffffFFFF;
        }
        stk = r.y;
        at = r.x;
        if(at == NO_PUSH || stk < bvh_stack_base)
            break;
    }

    *outT = tOut;
    *outTri = (__global BVHTriangleGPU*)(bvh_base + best_at * 8);
    return tOut != FLT_MAX;
}

// Cast a single ray and report if ANY occluder is hit (for ambient occlusion)
static inline bool any_hit_from(
    float3                    orig,
    const float3              dir,
    const size_t              bvh_base,
    const unsigned            root,
    const unsigned            bvh_stack_base) // здесь ещё было ignore_face, но оно не нужно, т.к. у нас orig и так сдвинут по нормали
{
    /*
    float outT, outU, outV;
    __global BVHTriangleGPU* outTri;
    return bvh_closest_hit(orig,dir,bvh_base,root,bvh_stack_base, 0.0f, &outT, &outTri, &outU, &outV);
    */
    uint4 descr = get_texture(bvh_base, root, false);
    float4 origp = (float4)(orig, 0); // почему-то llvm требует 4 компонента и игнорирует последний
    float4 dirp = (float4)(dir, 0);
    float4 inv_dirp = (float4)(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z, 0);

    unsigned stk = bvh_stack_base;
    const unsigned NO_PUSH = 0xfffffffe; // undocumented magic value -- disables pushing when passed as last_visited
    float tOut = FLT_MAX;
    unsigned at = root;
    size_t i;
    uint last_visited = 0xffffFFFF;
    for(i = 0; i < 10000; i++) {
        uint4 hw = __builtin_amdgcn_image_bvh_intersect_ray(at, tOut, origp, dirp, inv_dirp, descr);
        if(at % 8 == 0) {
            if(hw.w)
                return true;
            last_visited = NO_PUSH;
        }
        hw = (uint4)(hw.w, hw.z, hw.y, hw.x);
        uint2 r = __builtin_amdgcn_ds_bvh_stack_rtn(stk, last_visited, hw, STACK_SHIFT << 12);
        stk = r.y;
        if(r.x == -1) {
            r.x = get_parent(bvh_base, at);
            if(r.x == 0) // root
                break;
            stk++;
            last_visited = at;
        } else {
            last_visited = 0xffffFFFF;
        }
        at = r.x;
        if(at == NO_PUSH || stk < bvh_stack_base)
            break;
    }
    return false;
}

%:endif

// Helper: build tangent basis for a given normal
static inline void make_basis(const float3 n,
                              __private float3* t,
                              __private float3* b)
{
    // pick a non-parallel vector
    float3 up = (fabs(n.z) < 0.999f)
        ? (float3)(0.0f, 0.0f, 1.0f)
        : (float3)(0.0f, 1.0f, 0.0f);

    *t = normalize_f3(cross_f3(up, n));
    *b = cross_f3(n, *t);
}


__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void ray_tracing_render_using_bvh4(
    __global const char* bvh,
    __global int*              framebuffer_face_id,
    __global float*            framebuffer_ambient_occlusion,
    __global const CameraViewGPU* camera,
    unsigned root)
{
    // stack layout: stacks of 8 to 64 uint4s (configurable, we use 16), each with stride 128 (for 32-element warps, which we hopefully get)
    // since 16 is not sufficient for the entire traversal, we detect overflows and recover through parent pointers
    // there is no dedicated parent pointer field in the BVH structures, but using the reserved fields doesn't seem to break anything
    __local uint bvh_stack[GROUP_SIZE_X * GROUP_SIZE_Y * (8 << STACK_SHIFT)]; // size of local memory on RDNA3 is 65536
    // ds_bvh_stack_rtn requires the base pointer to be an offset relative to start of local storage allocated to this wave. apparently the start is at 0
    const uint li = get_local_id(0) + get_local_id(1) * GROUP_SIZE_X;
    unsigned bvh_stack_base = (size_t)bvh_stack / 4 + li % 32 + li / 32 * 32 * (8 << STACK_SHIFT);
    bvh_stack_base = bvh_stack_base << 18 | /*3 << 16 |*/ 0;
    
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    rassert(camera->magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 786435342);
    if (i >= camera->K.width || j >= camera->K.height)
        return;

    float3 ray_origin;
    float3 ray_direction;
    make_primary_ray(camera,
                     (float)i + 0.5f,
                     (float)j + 0.5f,
                     &ray_origin,
                     &ray_direction);

    float tMin      = 1e-6f;
    float tBest     = FLT_MAX;
    BVHTriangleGPU __global* faceBest = NULL;

    // Use BVH traversal instead of brute-force loop
    bool hit = bvh_closest_hit(
        ray_origin,
        ray_direction,
        (size_t)bvh, root, bvh_stack_base,
        tMin,
        &tBest,
        &faceBest);
    

    const uint idx = j * camera->K.width + i;
    framebuffer_face_id[idx] = hit ? faceBest->triangle_id : -1;
    

    float ao = 1.0f; // background stays white

    if (hit) {
        float3 a = loadVertex(faceBest->a, 0);
        float3 b = loadVertex(faceBest->b, 0);
        float3 c = loadVertex(faceBest->c, 0);

        float3 e1 = (float3)(b.x - a.x, b.y - a.y, b.z - a.z);
        float3 e2 = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float3 n  = normalize_f3(cross_f3(e1, e2));

        // ensure hemisphere is "outside" relative to the camera ray
        if (n.x * ray_direction.x +
            n.y * ray_direction.y +
            n.z * ray_direction.z > 0.0f)
        {
            n = (float3)(-n.x, -n.y, -n.z);
        }

        float3 P = (float3)(ray_origin.x + tBest * ray_direction.x,
                            ray_origin.y + tBest * ray_direction.y,
                            ray_origin.z + tBest * ray_direction.z);

        float3 ac = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float  scale = fmax(fmax(length_f3(e1), length_f3(e2)),
                            length_f3(ac));

        float  eps = 1e-3f * fmax(1.0f, scale);
        float3 Po  = (float3)(P.x + n.x * eps,
                              P.y + n.y * eps,
                              P.z + n.z * eps);

        // build tangent basis
        float3 T;
        float3 B;
        make_basis(n, &T, &B);

        // per-pixel seed (stable)
        union {
            float f32;
            uint  u32;
        } tBestUnion;
        tBestUnion.f32 = tBest; // not stable, tBest depends on computation method
        //uint rng = (uint)(0x9E3779B9u ^ idx ^ tBestUnion.u32);
        uint rng = (uint)(0x9E3779B9u ^ idx * 13371337);

        int hits = 0;
        for (int s = 0; s < AO_SAMPLES; ++s) {
            // uniform hemisphere sampling (solid angle)
            float u1  = random01(&rng);
            float u2  = random01(&rng);
            float z   = u1;                      // z in [0,1]
            float phi = 6.28318530718f * u2;     // 2*pi*u2
            float r   = sqrt(fmax(0.0f, 1.0f - z * z));
            float3 d_local = (float3)(r * cos(phi),
                                      r * sin(phi),
                                      z);

            // transform to world space
            float3 d = (float3)(
                T.x * d_local.x + B.x * d_local.y + n.x * d_local.z,
                T.y * d_local.x + B.y * d_local.y + n.y * d_local.z,
                T.z * d_local.x + B.z * d_local.y + n.z * d_local.z
            );

            if (any_hit_from(Po, d, (size_t)bvh, root, bvh_stack_base)) {
                ++hits;
            }
        }

        ao = 1.0f - (float)hits / (float)AO_SAMPLES; // [0,1]
    }
    framebuffer_ambient_occlusion[idx] = ao;
}
