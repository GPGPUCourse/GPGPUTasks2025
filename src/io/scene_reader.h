#pragma once

#include <vector>
#include <string>

#include "libbase/point.h"

// Minimal scene container: triangles by vertex indices + positions
struct SceneGeometry {
    std::vector<point3f> vertices; // each vertex is (x, y, z)
    std::vector<point3u> faces;    // each face is (v0, v1, v2) - indices of three vertices used in a triangle
    point3f min;
    point3f max;
};

// Loads geometry from Wavefront OBJ (supports v, f; polygons are fan-triangulated)
SceneGeometry loadOBJ(const std::string &path);

// Loads geometry from PLY (ascii or binary_little_endian 1.0; x/y/z + face index list)
SceneGeometry loadPLY(const std::string &path);

// Dispatch by file extension (.ply / .obj)
SceneGeometry loadScene(const std::string &path);
