#include "debug_io.h"

#include <libbase/math.h>
#include <libbase/bbox2.h>
#include <libbase/fast_random.h>
#include <libbase/string_utils.h>

#include <map>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>

#define LOOP_XY(img) for (int y = 0; y < (int) (img).height(); ++y) \
							for (int x = 0; x < (int) (img).width(); ++x)

namespace {

	template <typename T>
	point3<T> min3(const point3<T> &lhs, const point3<T> &rhs)
	{
		point3<T> result;

		result.x = std::min(lhs.x, rhs.x);
		result.y = std::min(lhs.y, rhs.y);
		result.z = std::min(lhs.z, rhs.z);

		return result;
	}

	template <typename T>
	point3<T> max3(const point3<T> &lhs, const point3<T> &rhs)
	{
		point3<T> result;

		result.x = std::max(lhs.x, rhs.x);
		result.y = std::max(lhs.y, rhs.y);
		result.z = std::max(lhs.z, rhs.z);

		return result;
	}

	point3uc getAngleColor(double angle_rad)
	{
		point3uc colors[4] = {
				point3uc(0, 255, 255),
				point3uc(255, 255, 0),
				point3uc(255, 0, 255),
				point3uc(255, 255, 0)};

		double sin = std::sin(angle_rad);
		double cos = std::cos(angle_rad);

		point3d c1;
		point3d c2;
		double coef;

		if (sin >= 0 && cos >= 0) {
			c1 = colors[0];
			c2 = colors[1];
			coef = sin * sin;
		}
		else
		if (sin >= 0 && cos <= 0) {
			c1 = colors[1];
			c2 = colors[2];
			coef = cos * cos;
		}
		else
		if (sin <= 0 && cos <= 0) {
			c1 = colors[2];
			c2 = colors[3];
			coef =  sin * sin;
		}
		else
		{
			c1 = colors[3];
			c2 = colors[0];
			coef = cos * cos;
		}

		point3d color = c1 * (1 - coef) + c2 * coef;
		color.x = std::clamp(color.x, 0.0, 255.0);
		color.y = std::clamp(color.y, 0.0, 255.0);
		color.z = std::clamp(color.z, 0.0, 255.0);

		return point3uc(color);
	}

	template <typename T>
	image8u colorMappingImpl(const TypedImage<T> &image, bool force)
	{
		if (!DEBUG_IO_IMAGE_SAVE_ENABLED && !(DEBUG_IO_FORCE_IMAGE_SAVE_ENABLED && force)) {
			return image8u();
		}

		point3<T> min_v(std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max());
		point3<T> max_v(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest());

		LOOP_XY(image) {

			if (image.channels() == 4 && image.ptr(y)[x * image.channels() + 3] == T(0)) {
				continue;
			}

			point3<T> v;
			v.x = image.ptr(y)[x * image.channels() + 0];
			v.y = image.ptr(y)[x * image.channels() + 1];
			v.z = image.ptr(y)[x * image.channels() + 2];

			min_v = min3(min_v, v);
			max_v = max3(max_v, v);
		}

		image8u result(image.width(), image.height(), 3);
		result.fillZero();

		LOOP_XY(result) {

			if (image.channels() == 4 && image.ptr(y)[x * image.channels() + 3] == T(0)) {
				continue;
			}

			point3<T> v;
			v.x = image.ptr(y)[x * image.channels() + 0];
			v.y = image.ptr(y)[x * image.channels() + 1];
			v.z = image.ptr(y)[x * image.channels() + 2];

			point3<T> res = (v - min_v) * T(255);
			res.x /= (max_v - min_v).x;
			res.y /= (max_v - min_v).y;
			res.z /= (max_v - min_v).z;

			res.x = std::clamp(res.x, T(0), T(255));
			res.y = std::clamp(res.y, T(0), T(255));
			res.z = std::clamp(res.z, T(0), T(255));

			result.ptr(y)[x * 3 + 0] = (unsigned char) res.x;
			result.ptr(y)[x * 3 + 1] = (unsigned char) res.y;
			result.ptr(y)[x * 3 + 2] = (unsigned char) res.z;
		}

		return result;
	}


	template <typename T>
	image8u randomMappingImpl(const TypedImage<T> &ids_map32, const T nodata_value, const std::string &bg_img_path, bool force)
	{

		if (!DEBUG_IO_IMAGE_SAVE_ENABLED && !(DEBUG_IO_FORCE_IMAGE_SAVE_ENABLED && force)) return image8u();

		image8u random_mapping_img;
		if (!bg_img_path.empty()) {
			random_mapping_img = image8u(bg_img_path);
			rassert(random_mapping_img.channels() == 3, 959300738); // ensuring that RGB
		} else {
			random_mapping_img = image8u(ids_map32.width(), ids_map32.height(), 3);
			random_mapping_img.fillZero();
		}

		std::unordered_map<int32_t, point3uc> random_mapping;

		point3uc nodata_color = point3uc(0, 255, 0); // green color
		FastRandom r(239); // seed is fixed for reproducibility
		LOOP_XY(ids_map32) {
			int32_t id = ids_map32(y, x);

			point3uc color;
			if (id == nodata_value) {
				color = nodata_color;
			} else {
				if (random_mapping.find(id) == random_mapping.end()) {
					random_mapping[id] = r.nextColor();
				}
				color = random_mapping[id];
			}
 
			if (!bg_img_path.empty()) {
				float factor = 0.5f;
				point3uc bg_color(&random_mapping_img(y, x));
				point3f blended_color = point3f(color) * factor + point3f(bg_color) * (1.0f - factor);
				for (int c = 0; c < 3; ++c) {
					color[c] = narrow_cast<unsigned char>(blended_color[c]);
				}
			}
			for (int c = 0; c < 3; ++c) {
				random_mapping_img(y, x, c) = color[c];
			}
		}

		return random_mapping_img;
	}

}

image8u debug_io::randomMapping(const image32i &ids_map, int nodata_value, const std::string &bg_img_path, bool force)
{
	return randomMappingImpl(ids_map, nodata_value, bg_img_path, force);
}

image8u debug_io::randomMapping(const image32u &ids_map, unsigned int nodata_value, const std::string &bg_img_path, bool force)
{
	return randomMappingImpl(ids_map, nodata_value, bg_img_path, force);
}

image8u debug_io::randomMapping(const image32i &ids_map, const std::string &bg_img_path, bool force)
{
	return randomMapping(ids_map, std::numeric_limits<int>::max(), bg_img_path, force);
}

image8u debug_io::randomMapping(const image32u &ids_map, const std::string &bg_img_path, bool force)
{
	return randomMapping(ids_map, std::numeric_limits<unsigned int>::max(), bg_img_path, force);
}

image8u debug_io::colorMapping(const image32f &image, bool force)
{
	return colorMappingImpl(image, force);
}

static std::string getPathFromName(const std::string &path)
{
	rassert(starts_with(path, "/"), 123008026814); // path is already global

	std::string parendDir = debug_io::getParentDir(path);
	debug_io::ensureDirExists(parendDir);

	return path;
}

void debug_io::dumpImage(const std::string &name, const AnyImage &image, bool verbose, bool force)
{
	if (!DEBUG_IO_IMAGE_SAVE_ENABLED && !(DEBUG_IO_FORCE_IMAGE_SAVE_ENABLED && force)) {
		return;
	}

	if (verbose) { 
		std::cout << "saving " << name << ", " << image.width() << " x " << image.height() << ", " << image.channels() << " channels..." << std::endl;
	}

	auto name_parts = split(name, ".");
	rassert(name_parts.size() >= 1, 617100897);
	std::string name_ext = tolower(name_parts[name_parts.size() - 1]);

	rassert(image.channels() == 1 || image.channels() == 3, image.channels(), 779908144);
	if (name_ext == "png") {
		image.savePNG(name);
	} else if (name_ext == "jpg" || name_ext == "jpeg") {
		image.saveJPEG(name);
	} else {
		throw std::runtime_error("unsupported image extension '" + name_ext + "' (name: " + name + ")");
	}
}


std::string debug_io::getParentDir(const std::string &path)
{
	std::string delimeter = "/";
	size_t pos = path.find_last_of(delimeter);
	std::string dirPath;

	if (pos == std::string::npos) {
		dirPath = ""; // this means that path doesn't contain any '/', f.e. it looks like 'image.jpg'
	} else {
		dirPath = path.substr(0, pos + 1); // including last slash
	}

	return dirPath;
}

void debug_io::ensureDirExists(const std::string &dirPath)
{
	if (dirPath.empty()) { 
		return;
	}

#ifndef __APPLE__
	std::filesystem::create_directories(dirPath);
#else
	// we can't use std::filesystem::create_directories on Mac OS because of this compilation error:
	// > error: 'create_directories' is unavailable: introduced in macOS 10.15
	// but this is used only on developers' PC for debug purposes, and we don't use Mac OS
	// so we can just throw exception if this method is suddenly used
	throw std::runtime_error("ensureDirExists was called but it is unsupported on macOS");
#endif
}

image8u debug_io::angleMapping(const image32f &angles, bool force)
{
	if (!DEBUG_IO_IMAGE_SAVE_ENABLED && !(DEBUG_IO_FORCE_IMAGE_SAVE_ENABLED && force)) {
		return image8u();
	}

	image8u result(angles.width(), angles.height(), 3);

	LOOP_XY(result) {

		float angle = angles(y, x);

		point3uc pixelResult;
		if (angle >= 0) {
			pixelResult = getAngleColor(angle);
		}

		for (int c = 0; c < 3; ++c) {
			result(y, x, c) = pixelResult[c];
		}

	}

	return result;
}

void debug_io::dumpPointCloud(const std::string &name, const std::vector<point3d> &points, const std::vector<point3uc> &colors)
{
	std::ofstream ofs(getPathFromName(name), std::ios::out);

	ofs <<  "ply\n"
			"format ascii 1.0\n" // TODO why ascii? may be use binary like in dumpModel?
			"element vertex " << points.size() << "\n"
			"property float x\n"
			"property float y\n"
			"property float z\n"
			"property uint8 red\n"
			"property uint8 green\n"
			"property uint8 blue\n"
			"end_header\n";

	for (int i = 0; i < (int) points.size(); ++i) {
		const point3d &point = points[i];
		const point3i &color = colors.size() ? colors[i] : point3uc(128, 128, 128);
		ofs << format("%1 %2 %3 %4 %5 %6\n", point.x, point.y, point.z, color.x, color.y, color.z);
	}
}

void debug_io::dumpModel(const std::string &name, const std::vector<point3f> &vertices, const std::vector<point3u> &faces)
{
	std::ofstream ofs(getPathFromName(name), std::ios::out);

	ofs <<  "ply\n"
			"format binary_little_endian 1.0\n"
			"element vertex " << vertices.size() << "\n"
			"property float x\n"
			"property float y\n"
			"property float z\n"
			"element face " << faces.size() << "\n"
			"property list uchar int vertex_indices\n"
			"end_header\n";

	for (size_t vi = 0; vi < vertices.size(); ++vi) {
		ofs.write((char *) (&vertices[vi]), 3 * sizeof(float));
	}

	for (size_t fi = 0; fi < faces.size(); ++fi) {
		unsigned char nv = 3;
		ofs.write((char *) &nv, sizeof(unsigned char));
		ofs.write((char *) (&faces[fi]), 3 * sizeof(unsigned int));
	}

	ofs.close();
	if (!ofs) {
		throw std::runtime_error("Can't dump model " + name);
	}
}

void debug_io::dumpModel(const std::string &name, const vertices_and_faces_t &vertices_and_faces)
{
	auto& [vertices, faces] = vertices_and_faces;
	dumpModel(name, vertices, faces);
}

image8u debug_io::depthMapping(const image32f &depth, float nodata_value, bool force)
{
	image8u img(depth.width(), depth.height(), 3);
	img.fillZero();

	const float MAX_NO_DATA = -std::numeric_limits<float>::max();
	const float MIN_NO_DATA =  std::numeric_limits<float>::max();
	float max_depth = MAX_NO_DATA;
	float min_depth = MIN_NO_DATA;

	LOOP_XY(depth) {
		float d = depth(y, x);
		if (d == nodata_value) {
			continue;
		}
		min_depth = std::min(min_depth, d);
		max_depth = std::max(max_depth, d);
	}
	if (min_depth > max_depth) { // protection from case when all pixels are equal to ignoredValue
		rassert(max_depth == MAX_NO_DATA, 205896686);
		rassert(min_depth == MIN_NO_DATA, 20589686);
		max_depth = min_depth;
	}
	if (min_depth == max_depth) { // protection from zero-division
		max_depth = min_depth + 1.0f;
	}

	point3uc nodata_color = point3uc(0, 255, 0); // green color
	LOOP_XY(depth) {
		float d = depth(y, x);
		point3uc color;
		if (d == nodata_value) {
			color = nodata_color;
		} else {
			auto intensity = narrow_cast<unsigned char>(std::round(255.0 * (d - min_depth) / (max_depth - min_depth)));
			color = point3uc(intensity, intensity, intensity);
		}
		for (int c = 0; c < 3; ++c) {
			img(y, x, c) = color[c];
		}
	}

	return img;
}

image8u debug_io::upscaleNearestNeighbor(const image8u &image, int k) {
	image8u upscaled_image(k * image.width(), k * image.height(), image.channels());
	#pragma omp parallel for
	LOOP_XY(upscaled_image) {
		for (int c = 0; c < upscaled_image.channels(); ++c) {
			upscaled_image(y, x, c) = image(y / k, x / k, c);
		}
	}
	return upscaled_image;
}

std::tuple<std::vector<point3f>, std::vector<point3u>> debug_io::representTexmappingAs3DModel(
		const std::vector<point2f> &texmapping_vertices, const std::vector<point4u> &texmapping_faces, size_t atlas_size)
{
	size_t nvertices = texmapping_vertices.size();
	size_t nfaces = texmapping_faces.size();

	std::vector<point3f> vertices(nvertices);
	std::vector<point3u> faces(nfaces);

	for (size_t vi = 0; vi < nvertices; ++vi) {
		point2f uv = texmapping_vertices[vi] * (float) atlas_size; // from [0, 1] range to [0, atlas_size]
		vertices[vi] = point3f(uv.x, uv.y, 0.0f);
	}
	for (size_t fi = 0; fi < nfaces; ++fi) {
		faces[fi] = texmapping_faces[fi].xyz(); // note that input faces are point4u: {v0, v1, v2, pageId}
	}

	const unsigned int NO_PAGE_ID = std::numeric_limits<unsigned int>::max();
	std::vector<unsigned int> vertices_page_id(nvertices, NO_PAGE_ID);
	std::map<unsigned int, bbox2f> pages_bboxes; // we use map instead of unordered_map to have guarantees about iteration order

	// saving per-vertex page_id info
	// estimating pages_bboxes
	for (size_t fi = 0; fi < nfaces; ++fi) {
		unsigned int page_id = texmapping_faces[fi].w; // note that input faces are point4u: {v0, v1, v2, pageId}
		for (size_t k = 0; k < 3; ++k) {
			unsigned int vi = faces[fi][k];
			if (vertices_page_id[vi] == NO_PAGE_ID) {
				vertices_page_id[vi] = page_id;
			} else {
				rassert(vertices_page_id[vi] == page_id, 827287796);
			}
			pages_bboxes[page_id].grow(vertices[k].xy());
		}
	}

	// estimate pages_y_offset for each page w.r.t. estimated pages_bboxes,
	// so that pages will not overlap (and have proper padding between them for visual clarity)
	std::unordered_map<unsigned int, float> pages_y_offset;
	float bboxes_offset_prefix = 0.0f;
	for (auto page_bbox: pages_bboxes) { // we use map instead of unordered_map to have guarantees about iteration order
		auto [page_id, bbox] = page_bbox;
		pages_y_offset[page_id] = bboxes_offset_prefix;
		bboxes_offset_prefix += bbox.height() * 1.1f;
	}

	// apply pages_y_offset for each page - adding offset to each vertex 
	for (size_t vi = 0; vi < nvertices; ++vi) {
		vertices[vi].y += pages_y_offset[vertices_page_id[vi]];
	}

	// add borders around each atlas page
	for (auto page_bbox: pages_bboxes) { // we use map instead of unordered_map to have guarantees about iteration order
		float page_y_offset = pages_y_offset[page_bbox.first];
		std::vector<point2f> atlas_corners = {{0, 0}, {1.0f*atlas_size, 0}, {1.0f*atlas_size, 1.0f*atlas_size}, {0, 1.0f*atlas_size}};
		for (size_t e = 0; e < 4; ++e) {
			point2f v0 = point2f(0.0f, page_y_offset) + atlas_corners[e];
			point2f v1 = point2f(0.0f, page_y_offset) + atlas_corners[(e + 1) % 4];
			unsigned int vi_prev = vertices.size(); vertices.push_back(point3f(v0.x, v0.y, 0.0f));
			unsigned int vi_next = vertices.size(); vertices.push_back(point3f(v1.x, v1.y, 0.0f));
			unsigned int vi_nxt2 = vertices.size(); vertices.push_back(point3f(v1.x, v1.y, 0.0f));
			faces.push_back({vi_prev, vi_next, vi_nxt2});
		}
	}

	return {vertices, faces};
}

std::string debug_io::padint(int i)
{
	return to_string_pad_zeros(i, 5);
}
