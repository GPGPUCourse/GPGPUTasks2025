#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "helpers/rassert.cu"

#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"

// my modified code from HW about merge_sort
// which also sorts values (indices and AABBs) along with keys
__global__ void merge_sort_key_value_kernel(
    const MortonCode* input_keys,
    const unsigned int* input_indices,
    const AABBGPU* input_aabbs,
    MortonCode* output_keys,
    unsigned int* output_indices,
    AABBGPU* output_aabbs,
    int sorted_k,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    const MortonCode key = input_keys[idx];
    const unsigned int value_idx = input_indices[idx];
    const AABBGPU value_aabb = input_aabbs[idx];

    // bucket = pair of sorted arrays
    const int bucket_idx = idx / (2 * sorted_k);
    const int bucket_start = bucket_idx * 2 * sorted_k;

    // half intervals
    const int left_start = bucket_start;
    const int left_end = (left_start + sorted_k < n) ? (left_start + sorted_k) : n;

    const int right_start = left_end;
    const int right_end = (right_start + sorted_k < n) ? (right_start + sorted_k) : n;

    const bool is_in_right = (idx >= right_start);

    // corner case at the end of the array (no right part)
    if (right_start >= n) {
        output_keys[idx] = key;
        output_indices[idx] = value_idx;
        output_aabbs[idx] = value_aabb;
        return;
    }

    int pos_to_put = INT_MAX;

    if (is_in_right) {
        int l = left_start;
        int r = left_end;

        while (l < r) {
            int m = (l + r) / 2;
            if (input_keys[m] <= key) { // <= for stability
                l = m + 1;
            } else {
                r = m;
            }
        }
        const int indice_in_left = l - left_start;
        const int indice_in_right = idx - right_start;

        pos_to_put = left_start + indice_in_left + indice_in_right;
    } else {
        int l = right_start;
        int r = right_end;

        while (l < r) {
            int m = (l + r) / 2;
            if (input_keys[m] < key) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        const int indice_in_left = idx - left_start;
        const int indice_in_right = l - right_start;

        pos_to_put = left_start + indice_in_left + indice_in_right;
    }

    output_keys[pos_to_put] = key;
    output_indices[pos_to_put] = value_idx;
    output_aabbs[pos_to_put] = value_aabb;
}

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
__device__  unsigned int expandBits_device(unsigned int v)
{
    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
__device__  MortonCode morton3D_device(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    const unsigned int ix = min(max((int)(x * 1024.0f), 0), 1023);
    const unsigned int iy = min(max((int)(y * 1024.0f), 0), 1023);
    const unsigned int iz = min(max((int)(z * 1024.0f), 0), 1023);

    const unsigned int xx = expandBits_device(ix);
    const unsigned int yy = expandBits_device(iy);
    const unsigned int zz = expandBits_device(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__global__ void calc_triangle_aabb_and_centroids(
    const float* vertices,
    const unsigned int* faces,
    AABBGPU* triangle_aabbs,
    float3* centroids,
    unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces) return;

    const uint3 face = loadFace(faces, i);
    const float3 v0 = loadVertex(vertices, face.x);
    const float3 v1 = loadVertex(vertices, face.y);
    const float3 v2 = loadVertex(vertices, face.z);

    {
        AABBGPU aabb;
        aabb.min_x = fminf(fminf(v0.x, v1.x), v2.x);
        aabb.min_y = fminf(fminf(v0.y, v1.y), v2.y);
        aabb.min_z = fminf(fminf(v0.z, v1.z), v2.z);
        aabb.max_x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
        aabb.max_y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
        aabb.max_z = fmaxf(fmaxf(v0.z, v1.z), v2.z);

        triangle_aabbs[i] = aabb;
    }
    
    {
        float3 centroid;
        centroid.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
        centroid.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
        centroid.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

        centroids[i] = centroid;
    }
}

// Global centroid bounds (for morton codes later)
__global__ void calc_centroid_bounds_reduce(
    const float3* centroids,
    float3* partial_mins,
    float3* partial_maxs,
    unsigned int nfaces)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float3 shared_mins[GROUP_SIZE];
    __shared__ float3 shared_maxs[GROUP_SIZE];

    // init
    if (i < nfaces) {
        shared_mins[tid] = centroids[i];
        shared_maxs[tid] = centroids[i];
    } else {
        shared_mins[tid] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        shared_maxs[tid] = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    __syncthreads();

    // reduction in shared
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mins[tid].x = fminf(shared_mins[tid].x, shared_mins[tid + s].x);
            shared_mins[tid].y = fminf(shared_mins[tid].y, shared_mins[tid + s].y);
            shared_mins[tid].z = fminf(shared_mins[tid].z, shared_mins[tid + s].z);

            shared_maxs[tid].x = fmaxf(shared_maxs[tid].x, shared_maxs[tid + s].x);
            shared_maxs[tid].y = fmaxf(shared_maxs[tid].y, shared_maxs[tid + s].y);
            shared_maxs[tid].z = fmaxf(shared_maxs[tid].z, shared_maxs[tid + s].z);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_mins[blockIdx.x] = shared_mins[0];
        partial_maxs[blockIdx.x] = shared_maxs[0];
    }
}

__global__ void calc_morton_codes(
    const float3* centroids,
    const float3* centroid_min,
    const float3* centroid_max,
    MortonCode* morton_codes,
    unsigned int* original_indices,
    unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces) return;

    const float3 cMin = centroid_min[0];
    const float3 cMax = centroid_max[0];

    const float eps = 1e-9f;
    const float dx = fmaxf(cMax.x - cMin.x, eps);
    const float dy = fmaxf(cMax.y - cMin.y, eps);
    const float dz = fmaxf(cMax.z - cMin.z, eps);

    const float3 c = centroids[i];
    float nx = (c.x - cMin.x) / dx;
    float ny = (c.y - cMin.y) / dy;
    float nz = (c.z - cMin.z) / dz;

    // Clamp to [0,1]
    nx = fminf(fmaxf(nx, 0.0f), 1.0f);
    ny = fminf(fmaxf(ny, 0.0f), 1.0f);
    nz = fminf(fmaxf(nz, 0.0f), 1.0f);

    morton_codes[i] = morton3D_device(nx, ny, nz);
    original_indices[i] = i;
}

__device__ int clz32(uint32_t x)
{
    if (x == 0u) return 32;
    return __clz(x);
}

// compute common prefix length between sorted Morton codes at indices i and j
__device__ int common_prefix(const MortonCode* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;

    const MortonCode ci = codes[i];
    const MortonCode cj = codes[j];

    if (ci == cj) {
        uint32_t di = static_cast<uint32_t>(i);
        uint32_t dj = static_cast<uint32_t>(j);
        uint32_t diff = di ^ dj;
        return 32 + clz32(diff);
    } else {
        uint32_t diff = ci ^ cj;
        return clz32(diff);
    }
}

__global__ void build_internal_nodes(
    const MortonCode* morton_codes,
    BVHNodeGPU* nodes,
    unsigned int* parent_indices,
    unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = (int)nfaces;
    
    if (i >= N - 1) return;

    // direction
    const int cpL = common_prefix(morton_codes, N, i, i - 1);
    const int cpR = common_prefix(morton_codes, N, i, i + 1);
    const int dir = (cpR > cpL) ? 1 : -1;

    // find upper bound on the length
    int deltaMin = common_prefix(morton_codes, N, i, i - dir);
    int lmax = 2;
    while (common_prefix(morton_codes, N, i, i + lmax * dir) > deltaMin) {
        lmax <<= 1;
    }

    // find exact range with gallopping
    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(morton_codes, N, i, i + (l + t) * dir) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * dir;
    int first = min(i, j);
    int last = max(i, j);

    // split pos
    int commonPrefix = common_prefix(morton_codes, N, first, last);
    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = common_prefix(morton_codes, N, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    // find child indices
    const int leafStart = N - 1;
    
    int leftIndex;
    if (split == first) {
        leftIndex = leafStart + split;
    } else {
        leftIndex = split;
    }

    int rightIndex;
    if (split + 1 == last) {
        rightIndex = leafStart + split + 1;
    } else {
        rightIndex = split + 1;
    }

    BVHNodeGPU& node = nodes[i];
    node.leftChildIndex = (uint32_t)(leftIndex);
    node.rightChildIndex = (uint32_t)(rightIndex);
    
    // write on the forehead of children that we are their parent
    parent_indices[leftIndex] = i;
    parent_indices[rightIndex] = i;
}

__global__ void initialize_leaf_nodes(
    const AABBGPU* triangle_aabbs,
    BVHNodeGPU* nodes,
    unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces) return;

    const unsigned int leafIndex = (nfaces - 1) + i;
    BVHNodeGPU& leaf = nodes[leafIndex];

    leaf.aabb = triangle_aabbs[i];
    leaf.leftChildIndex = 0xFFFFFFFFu;
    leaf.rightChildIndex = 0xFFFFFFFFu;
}

__global__ void calc_internal_aabbs(
    BVHNodeGPU* nodes,
    const unsigned int* parent_indices,
    int* atomic_flags,
    unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces) return;

    const int N = (int)nfaces;
    const int leafStart = N - 1;
    
    int currentIndex = leafStart + i;

    // go to root
    while (true) {
        const unsigned int parentIndex = parent_indices[currentIndex];

        if (parentIndex == 0xFFFFFFFFu || parentIndex == -1) {
            break;
        }

        const int old = atomicAdd(&atomic_flags[parentIndex], 1); // <- old = which was read
        
        if (old == 0) { // first child that enters here exits
            break;
        } else {
            // this is a second arrived child, make him compute AABB for parent
            BVHNodeGPU& parent = nodes[parentIndex];
            const BVHNodeGPU& left = nodes[parent.leftChildIndex];
            const BVHNodeGPU& right = nodes[parent.rightChildIndex];

            // TODO: would be great to move aabb operations to dedicated functions
            AABBGPU aabb;
            aabb.min_x = fminf(left.aabb.min_x, right.aabb.min_x);
            aabb.min_y = fminf(left.aabb.min_y, right.aabb.min_y);
            aabb.min_z = fminf(left.aabb.min_z, right.aabb.min_z);
            aabb.max_x = fmaxf(left.aabb.max_x, right.aabb.max_x);
            aabb.max_y = fmaxf(left.aabb.max_y, right.aabb.max_y);
            aabb.max_z = fmaxf(left.aabb.max_z, right.aabb.max_z);

            parent.aabb = aabb;

            currentIndex = parentIndex;
        }
    }
}

namespace cuda {
void lbvh_build_gpu(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& bvhNodes,
    gpu::gpu_mem_32u& leafTriIndices,
    unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    if (nfaces == 0) return;

    if (nfaces == 1) {
        if (bvhNodes.size() < 1) {
            bvhNodes.resize(1);
        }
        
        std::vector<BVHNodeGPU> single_node(1);
        std::vector<AABBGPU> single_aabb(1);
        
        gpu::shared_device_buffer_typed<AABBGPU> triangle_aabbs(1);
        gpu::shared_device_buffer_typed<float3> centroids(1);
        
        ::calc_triangle_aabb_and_centroids<<<1, 1, 0, stream>>>(
            vertices.cuptr(), faces.cuptr(),
            triangle_aabbs.cuptr(), centroids.cuptr(),
            1);
        CUDA_CHECK_KERNEL(stream);
        
        triangle_aabbs.readN(single_aabb.data(), 1);
        single_node[0].aabb = single_aabb[0];
        single_node[0].leftChildIndex = 0xFFFFFFFFu;
        single_node[0].rightChildIndex = 0xFFFFFFFFu;
        
        bvhNodes.writeN(single_node.data(), 1);
        
        std::vector<uint32_t> single_index = {0};
        leafTriIndices.writeN(single_index.data(), 1);
        return;
    }

    gpu::shared_device_buffer_typed<AABBGPU> triangle_aabbs(nfaces);
    gpu::shared_device_buffer_typed<float3> centroids(nfaces);

    // 1) calc AABB and centroids for all faces
    ::calc_triangle_aabb_and_centroids<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(),
        triangle_aabbs.cuptr(), centroids.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);

    // 2) calculate global centroid bounds
    unsigned int num_blocks = (nfaces + GROUP_SIZE - 1) / GROUP_SIZE;
    gpu::shared_device_buffer_typed<float3> partial_mins(num_blocks);
    gpu::shared_device_buffer_typed<float3> partial_maxs(num_blocks);

    ::calc_centroid_bounds_reduce<<<num_blocks, GROUP_SIZE, 0, stream>>>(
        centroids.cuptr(),
        partial_mins.cuptr(), partial_maxs.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);

    // final reduction on CPU
    std::vector<float3> mins_host(num_blocks);
    std::vector<float3> maxs_host(num_blocks);
    partial_mins.readN(mins_host.data(), num_blocks);
    partial_maxs.readN(maxs_host.data(), num_blocks);

    float3 global_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 global_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (unsigned int b = 0; b < num_blocks; ++b) {
        global_min.x = fminf(global_min.x, mins_host[b].x);
        global_min.y = fminf(global_min.y, mins_host[b].y);
        global_min.z = fminf(global_min.z, mins_host[b].z);
        global_max.x = fmaxf(global_max.x, maxs_host[b].x);
        global_max.y = fmaxf(global_max.y, maxs_host[b].y);
        global_max.z = fmaxf(global_max.z, maxs_host[b].z);
    }

    gpu::shared_device_buffer_typed<float3> centroid_min(1);
    gpu::shared_device_buffer_typed<float3> centroid_max(1);
    centroid_min.writeN(&global_min, 1);
    centroid_max.writeN(&global_max, 1);

    gpu::gpu_mem_32u morton_codes(nfaces);
    gpu::gpu_mem_32u original_indices(nfaces);

    // 3) calc Morton codes
    ::calc_morton_codes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        centroids.cuptr(),
        centroid_min.cuptr(), centroid_max.cuptr(),
        morton_codes.cuptr(), original_indices.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);

    gpu::gpu_mem_32u morton_codes_sorted(nfaces);
    gpu::shared_device_buffer_typed<AABBGPU> triangle_aabbs_sorted(nfaces);
    gpu::gpu_mem_32u sorted_indices(nfaces);

    // 4) Sort by Morton codes with payload (indices and AABBs)
    MortonCode* current_keys_input = morton_codes.cuptr();
    MortonCode* current_keys_output = morton_codes_sorted.cuptr();
    unsigned int* current_indices_input = original_indices.cuptr();
    unsigned int* current_indices_output = sorted_indices.cuptr();
    AABBGPU* current_aabbs_input = triangle_aabbs.cuptr();
    AABBGPU* current_aabbs_output = triangle_aabbs_sorted.cuptr();

    int current_k = 1;

    while (current_k < (int)nfaces) {
        ::merge_sort_key_value_kernel<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            current_keys_input,
            current_indices_input,
            current_aabbs_input,
            current_keys_output,
            current_indices_output,
            current_aabbs_output,
            current_k,
            nfaces);
        CUDA_CHECK_KERNEL(stream);

        // Swap pointers for next iteration
        MortonCode* tmp_keys = current_keys_input;
        current_keys_input = current_keys_output;
        current_keys_output = tmp_keys;

        unsigned int* tmp_indices = current_indices_input;
        current_indices_input = current_indices_output;
        current_indices_output = tmp_indices;

        AABBGPU* tmp_aabbs = current_aabbs_input;
        current_aabbs_input = current_aabbs_output;
        current_aabbs_output = tmp_aabbs;

        current_k *= 2;
    }

    if (current_keys_input != morton_codes_sorted.cuptr()) {
        cudaMemcpyAsync(morton_codes_sorted.cuptr(), current_keys_input,
                        nfaces * sizeof(MortonCode), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(sorted_indices.cuptr(), current_indices_input,
                        nfaces * sizeof(unsigned int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(triangle_aabbs_sorted.cuptr(), current_aabbs_input,
                        nfaces * sizeof(AABBGPU), cudaMemcpyDeviceToDevice, stream);
        CUDA_CHECK_KERNEL(stream);
    }

    cudaMemcpyAsync(leafTriIndices.cuptr(), sorted_indices.cuptr(),
                    nfaces * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
    CUDA_CHECK_KERNEL(stream);

    // 5) init leaf nodes
    const unsigned int num_nodes = 2 * nfaces - 1;
    if (bvhNodes.size() < num_nodes) {
        bvhNodes.resize(num_nodes);
    }

    gpu::gpu_mem_32u parent_indices(num_nodes);
    parent_indices.fill(0xFFFFFFFFu);

    ::initialize_leaf_nodes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        triangle_aabbs_sorted.cuptr(),
        bvhNodes.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);

    // 6) build internal nodes
    ::build_internal_nodes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        morton_codes_sorted.cuptr(),
        bvhNodes.cuptr(),
        parent_indices.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);

    // 7) AABBs for internal nodes
    gpu::gpu_mem_32i atomic_flags(nfaces - 1);
    atomic_flags.fill(0);

    ::calc_internal_aabbs<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        bvhNodes.cuptr(),
        parent_indices.cuptr(),
        atomic_flags.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
