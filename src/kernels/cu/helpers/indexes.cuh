#pragma once


// OpenCL: get_global_id(0)
__device__ __forceinline__ int global_index_axis_x() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// OpenCL: get_global_id(1)
__device__ __forceinline__ int global_index_axis_y() {
    return blockIdx.y * blockDim.y + threadIdx.y;
}

// OpenCL: get_global_id(2)
__device__ __forceinline__ int global_index_axis_z() {
    return blockIdx.z * blockDim.z + threadIdx.z;
}

// OpenCL: get_local_id(0)
__device__ __forceinline__ int thread_index_axis_x() {
    return threadIdx.x;
}

// OpenCL: get_local_id(1)
__device__ __forceinline__ int thread_index_axis_y() {
    return threadIdx.y;
}

// OpenCL: get_local_id(2)
__device__ __forceinline__ int thread_index_axis_z() {
    return threadIdx.z;
}

// OpenCL: get_group_id(0)
__device__ __forceinline__ int work_group_index_axis_x() {
    return blockIdx.x;
}

// OpenCL: get_group_id(1)
__device__ __forceinline__ int work_group_index_axis_y() {
    return blockIdx.y;
}

// OpenCL: get_group_id(2)
__device__ __forceinline__ int work_group_index_axis_z() {
    return blockIdx.z;
}
