#pragma once

// BVH debug helpers: dump LBVH nodes as PLY boxes/points for Meshlab.

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <limits>

// #include "scene_reader.h"
#include "../kernels/shared_structs/bvh_node_gpu_shared.h"

namespace debug {
namespace bvh_detail {

    // Write simple ASCII PLY header with vertices and triangle faces
    inline void write_ply_header(std::ofstream& out,
        std::uint32_t vertex_count,
        std::uint32_t face_count)
    {
        // ASCII PLY, vertices with (x,y,z), triangle faces
        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "element vertex " << vertex_count << "\n";
        out << "property float x\n";
        out << "property float y\n";
        out << "property float z\n";
        out << "element face " << face_count << "\n";
        out << "property list uchar int vertex_indices\n";
        out << "end_header\n";
    }

    // BFS from root (index 0) to compute depths of all reachable nodes
    inline std::vector<std::uint32_t> compute_depths(const std::vector<BVHNodeGPU>& nodes)
    {
        const std::size_t n = nodes.size();
        std::vector<std::uint32_t> depth(n,
            std::numeric_limits<std::uint32_t>::max());

        if (n == 0)
            return depth;

        std::vector<std::uint32_t> queue;
        queue.reserve(n);

        depth[0] = 0;
        queue.push_back(0);

        std::size_t head = 0;
        while (head < queue.size()) {
            std::uint32_t idx = queue[head++];
            const BVHNodeGPU& node = nodes[idx];

            std::uint32_t d_next = depth[idx] + 1;

            std::uint32_t lc = node.leftChildIndex;
            std::uint32_t rc = node.rightChildIndex;

            if (lc < n && depth[lc] == std::numeric_limits<std::uint32_t>::max()) {
                depth[lc] = d_next;
                queue.push_back(lc);
            }
            if (rc < n && depth[rc] == std::numeric_limits<std::uint32_t>::max()) {
                depth[rc] = d_next;
                queue.push_back(rc);
            }
        }

        return depth;
    }

    // Write 8 vertices of a box for given AABB
    inline void write_box_vertices(std::ofstream& out, const AABBGPU& aabb)
    {
        const float min_x = aabb.min_x;
        const float min_y = aabb.min_y;
        const float min_z = aabb.min_z;

        const float max_x = aabb.max_x;
        const float max_y = aabb.max_y;
        const float max_z = aabb.max_z;

        // Vertex order:
        // 0: (min_x, min_y, min_z)
        // 1: (max_x, min_y, min_z)
        // 2: (max_x, max_y, min_z)
        // 3: (min_x, max_y, min_z)
        // 4: (min_x, min_y, max_z)
        // 5: (max_x, min_y, max_z)
        // 6: (max_x, max_y, max_z)
        // 7: (min_x, max_y, max_z)

        out << min_x << ' ' << min_y << ' ' << min_z << '\n'; // 0
        out << max_x << ' ' << min_y << ' ' << min_z << '\n'; // 1
        out << max_x << ' ' << max_y << ' ' << min_z << '\n'; // 2
        out << min_x << ' ' << max_y << ' ' << min_z << '\n'; // 3
        out << min_x << ' ' << min_y << ' ' << max_z << '\n'; // 4
        out << max_x << ' ' << min_y << ' ' << max_z << '\n'; // 5
        out << max_x << ' ' << max_y << ' ' << max_z << '\n'; // 6
        out << min_x << ' ' << max_y << ' ' << max_z << '\n'; // 7
    }

    // Write 12 triangle faces for a box that uses vertices [base, base+7]
    inline void write_box_faces(std::ofstream& out, std::uint32_t base)
    {
        // Helper lambda to get absolute vertex index
        auto v = [base](std::uint32_t local_index) {
            return base + local_index;
        };

        // Bottom (z = min)
        out << "3 " << v(0) << ' ' << v(1) << ' ' << v(2) << '\n';
        out << "3 " << v(0) << ' ' << v(2) << ' ' << v(3) << '\n';

        // Top (z = max)
        out << "3 " << v(4) << ' ' << v(5) << ' ' << v(6) << '\n';
        out << "3 " << v(4) << ' ' << v(6) << ' ' << v(7) << '\n';

        // Front (y = min)
        out << "3 " << v(0) << ' ' << v(1) << ' ' << v(5) << '\n';
        out << "3 " << v(0) << ' ' << v(5) << ' ' << v(4) << '\n';

        // Right (x = max)
        out << "3 " << v(1) << ' ' << v(2) << ' ' << v(6) << '\n';
        out << "3 " << v(1) << ' ' << v(6) << ' ' << v(5) << '\n';

        // Back (y = max)
        out << "3 " << v(2) << ' ' << v(3) << ' ' << v(7) << '\n';
        out << "3 " << v(2) << ' ' << v(7) << ' ' << v(6) << '\n';

        // Left (x = min)
        out << "3 " << v(3) << ' ' << v(0) << ' ' << v(4) << '\n';
        out << "3 " << v(3) << ' ' << v(4) << ' ' << v(7) << '\n';
    }

} // namespace bvh_detail

// Dump original scene geometry (as triangles) into a PLY file.
inline void dump_scene_geometry_ply(const std::string& filename,
    const SceneGeometry& scene)
{
    const std::size_t nverts = scene.vertices.size();
    const std::size_t nfaces = scene.faces.size();

    if (nverts == 0 || nfaces == 0)
        return;

    // Reuse the same layout assumption as GPU upload:
    // SceneGeometry::vertices: packed [x y z] float per vertex
    // SceneGeometry::faces:    packed [i0 i1 i2] uint32 per triangle
    const float* vdata = reinterpret_cast<const float*>(scene.vertices.data());
    const std::uint32_t* fdata =
        reinterpret_cast<const std::uint32_t*>(scene.faces.data());

    std::ofstream out(filename.c_str());
    if (!out.is_open())
        return;

    bvh_detail::write_ply_header(out,
        static_cast<std::uint32_t>(nverts),
        static_cast<std::uint32_t>(nfaces));

    // Vertices
    for (std::size_t i = 0; i < nverts; ++i) {
        const float x = vdata[3 * i + 0];
        const float y = vdata[3 * i + 1];
        const float z = vdata[3 * i + 2];
        out << x << ' ' << y << ' ' << z << '\n';
    }

    // Faces
    for (std::size_t i = 0; i < nfaces; ++i) {
        const std::uint32_t i0 = fdata[3 * i + 0];
        const std::uint32_t i1 = fdata[3 * i + 1];
        const std::uint32_t i2 = fdata[3 * i + 2];
        out << "3 " << i0 << ' ' << i1 << ' ' << i2 << '\n';
    }
}

// Dump BVH nodes as boxes (triangulated cubes) for nodes with depth in [min_depth, max_depth].
// Depth is computed from node 0 as root.
inline void dump_bvh_boxes_ply(const std::string& filename,
    const std::vector<BVHNodeGPU>& nodes,
    std::uint32_t min_depth,
    std::uint32_t max_depth)
{
    if (nodes.empty())
        return;

    const std::size_t n = nodes.size();
    std::vector<std::uint32_t> depth = bvh_detail::compute_depths(nodes);

    // Count how many nodes we are going to export
    std::size_t selected_nodes = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t d = depth[i];
        if (d == std::numeric_limits<std::uint32_t>::max())
            continue; // unreachable from root
        if (d < min_depth || d > max_depth)
            continue;
        ++selected_nodes;
    }

    std::cerr << "debug BVH: in " << selected_nodes << " nodes depth is from [" << min_depth << "; " << max_depth << "]" << std::endl;
    if (selected_nodes == 0)
        return;

    const std::uint32_t verts_per_node = 8u;
    const std::uint32_t faces_per_node = 12u;

    std::uint64_t vert_count64 = selected_nodes * verts_per_node;
    std::uint64_t face_count64 = selected_nodes * faces_per_node;

    // For debug purposes we assume counts fit into uint32_t
    if (vert_count64 > std::numeric_limits<std::uint32_t>::max() ||
        face_count64 > std::numeric_limits<std::uint32_t>::max())
    {
        // Clamp just in case; resulting file will be incomplete but still valid.
        vert_count64 = std::min<std::uint64_t>(
            vert_count64, std::numeric_limits<std::uint32_t>::max());
        face_count64 = std::min<std::uint64_t>(
            face_count64, std::numeric_limits<std::uint32_t>::max());
    }

    std::ofstream out(filename.c_str());
    if (!out.is_open())
        return;

    bvh_detail::write_ply_header(
        out,
        static_cast<std::uint32_t>(vert_count64),
        static_cast<std::uint32_t>(face_count64));

    // First pass: write all vertices in the same order as faces will expect
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t d = depth[i];
        if (d == std::numeric_limits<std::uint32_t>::max())
            continue;
        if (d < min_depth || d > max_depth)
            continue;

        const BVHNodeGPU& node = nodes[i];
        bvh_detail::write_box_vertices(out, node.aabb);
    }

    // Second pass: write faces, keeping the same order and tracking base index
    std::uint32_t base_index = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t d = depth[i];
        if (d == std::numeric_limits<std::uint32_t>::max())
            continue;
        if (d < min_depth || d > max_depth)
            continue;

        bvh_detail::write_box_faces(out, base_index);
        base_index += verts_per_node;
    }
}

// Convenience wrapper: dump all reachable BVH nodes as boxes
inline void dump_bvh_all_boxes_ply(const std::string& filename,
    const std::vector<BVHNodeGPU>& nodes)
{
    dump_bvh_boxes_ply(
        filename,
        nodes,
        0u,
        std::numeric_limits<std::uint32_t>::max());
}

// Dump BVH node centers as points (no faces) for nodes with depth in [min_depth, max_depth]
inline void dump_bvh_centers_ply(const std::string& filename,
    const std::vector<BVHNodeGPU>& nodes,
    std::uint32_t min_depth,
    std::uint32_t max_depth)
{
    if (nodes.empty())
        return;

    const std::size_t n = nodes.size();
    std::vector<std::uint32_t> depth = bvh_detail::compute_depths(nodes);

    // Count selected nodes
    std::size_t selected_nodes = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t d = depth[i];
        if (d == std::numeric_limits<std::uint32_t>::max())
            continue;
        if (d < min_depth || d > max_depth)
            continue;
        ++selected_nodes;
    }

    if (selected_nodes == 0)
        return;

    std::ofstream out(filename.c_str());
    if (!out.is_open())
        return;

    const std::uint32_t vertex_count =
        static_cast<std::uint32_t>(selected_nodes);
    const std::uint32_t face_count = 0;

    bvh_detail::write_ply_header(out, vertex_count, face_count);

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t d = depth[i];
        if (d == std::numeric_limits<std::uint32_t>::max())
            continue;
        if (d < min_depth || d > max_depth)
            continue;

        const AABBGPU& aabb = nodes[i].aabb;
        const float cx = 0.5f * (aabb.min_x + aabb.max_x);
        const float cy = 0.5f * (aabb.min_y + aabb.max_y);
        const float cz = 0.5f * (aabb.min_z + aabb.max_z);
        out << cx << ' ' << cy << ' ' << cz << '\n';
    }
}

} // namespace debug
