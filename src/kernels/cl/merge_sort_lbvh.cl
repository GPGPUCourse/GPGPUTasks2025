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

__kernel void merge_sort_lbvh(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  n,
                   int  chunk_size)
{
    // 2 * i value of element is morton code, 2 * i + 1 value is triIndex
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    int first_part_index = i - i % (chunk_size * 2);
    int second_part_index = first_part_index + chunk_size;
    int local_shift = i % chunk_size;
    if (second_part_index >= n) {
        output_data[i * 2] = input_data[i * 2];
        output_data[i * 2 + 1] = input_data[i * 2 + 1];
        return;
    }
    int l = -1;
    int r = chunk_size;
    int addition;
    bool strict_less;
    if (i < second_part_index) {
        addition = second_part_index;
        strict_less = true;
    }
    else {
        addition = first_part_index;
        strict_less = false;
    }
    while (r - l > 1) {
        int m = (r + l) / 2;
        int index_m = m + addition;
        if (index_m < n && ((input_data[index_m * 2] < input_data[i * 2] && strict_less) || (input_data[index_m * 2] <= input_data[i * 2] && !strict_less))) {
            l = m;
        }
        else {
            r = m;
        }
    }
    uint output_index = first_part_index + local_shift + r;
    output_data[output_index * 2] = input_data[i * 2];
    output_data[output_index * 2 + 1] = input_data[i * 2 + 1];
}
