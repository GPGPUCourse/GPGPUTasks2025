#pragma once

#include <libimages/images.h>

#include <libbase/point.h>

#include <vector>
#include <limits>

#define DEBUG_IO_IMAGE_SAVE_ENABLED 1
#define DEBUG_IO_FORCE_IMAGE_SAVE_ENABLED 0

namespace debug_io {

	std::string	getParentDir(const std::string &path);

	void		ensureDirExists(const std::string &path);

	void		dumpImage(const std::string &name, const AnyImage &image, bool verbose = false, bool force = false);

	void		dumpPointCloud(const std::string &name, const std::vector<point3d> &points, const std::vector<point3uc> &colors = {});

	typedef std::tuple<std::vector<point3f>, std::vector<point3u>> vertices_and_faces_t;
	void		dumpModel(const std::string &name, const std::vector<point3f> &vertices, const std::vector<point3u> &faces);
	void		dumpModel(const std::string &name, const vertices_and_faces_t &vertices_and_faces);

	image8u		colorMapping(const image32f &image, bool force = false);

	image8u		randomMapping(const image32i &ids_map, int nodata_value = std::numeric_limits<int>::max(), const std::string &bg_img_path = "", bool force = false);
	image8u		randomMapping(const image32u &ids_map, unsigned int nodata_value = std::numeric_limits<unsigned int>::max(), const std::string &bg_img_path = "", bool force = false);

	image8u		randomMapping(const image32i &ids_map, const std::string &bg_img_path, bool force = false);
	image8u		randomMapping(const image32u &ids_map, const std::string &bg_img_path, bool force = false);

	image8u		angleMapping(const image32f &angles, bool force = false);

	image8u		depthMapping(const image32f &depth, float nodata_value = std::numeric_limits<float>::max(), bool force = false);

	image8u		upscaleNearestNeighbor(const image8u &image, int k);

	// note that input faces are point4u: {v0, v1, v2, pageId}
	vertices_and_faces_t	representTexmappingAs3DModel(const std::vector<point2f> &texmapping_vertices, const std::vector<point4u> &texmapping_faces, size_t atlas_size); // returns vertices+faces (can be saved via dumpModel(...))

	std::string	padint(int i);

}
