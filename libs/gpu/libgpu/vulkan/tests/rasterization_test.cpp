#include <gtest/gtest.h>

#include <libbase/timer.h>
#include <libbase/stats.h>
#include <libbase/point.h>
#include <libimages/debug_io.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/vulkan/vulkan_api_headers.h>

#include "test_utils.h"

#include "kernels/defines.h"
#include "kernels/kernels.h"

#define DEBUG_DIR std::string("")

TEST(vulkan, renderRectangle)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		std::vector<point3f> vertices_xyz(4);
		float fromX = 0.15, toX = 0.75;
		float fromY = 0.25, toY = 0.65;
		float nearZ = 0.20, farZ = 0.60;
		float emptyZ = 0.9;
		vertices_xyz[0] = point3f(fromX,	fromY,	nearZ);	// top-left (near)
		vertices_xyz[1] = point3f(toX,		fromY,	farZ);	// top-right (far)
		vertices_xyz[2] = point3f(fromX,	toY,	nearZ);	// bottom-left (near)
		vertices_xyz[3] = point3f(toX,		toY,	farZ);	// bottom-right (near)

		std::vector<point3u> faces(2);
		faces[0] = point3u(0, 1, 2);
		faces[1] = point3u(2, 1, 3);

		gpu::gpu_mem_vertices_xyz gpu_vertices;
		gpu_vertices.resizeN(vertices_xyz.size());
		gpu_vertices.writeN((gpu::Vertex3D*) vertices_xyz.data(), vertices_xyz.size());

		gpu::gpu_mem_32u gpu_faces;
		gpu_faces.resizeN(3 * faces.size());
		gpu_faces.writeN((unsigned int*) faces.data(), 3 * faces.size());

		std::vector<std::pair<size_t, size_t>> rendering_test_sizes_wh = {
				{1000,	800}, // the most simple
				{10,	10},  // it is used to check corner case when rectangle borders passes through pixel centers
				{239,	239},
				{1023,	767},
				{1024,	768},
				{1025,	769},

				// 1x1 and 3x3 leads to bigger depth epsilon (don't know why)
				{3,		3},
				{1,		1},
		};
		for (auto rendering_test_size : rendering_test_sizes_wh) {
			size_t width = rendering_test_size.first;
			size_t height = rendering_test_size.second;
			size_t cn = 3;
			std::cout << "evaluating viewport resolution " << width << "x" << height << "..." << std::endl;

			gpu::shared_device_depth_image depth_buffer(width, height);
			gpu_image8u color_buffer(width, height, cn);
			gpu_image32i face_id_buffer(width, height);

			avk2::KernelSource kernel_rasterization(avk2::getRasterizeKernel());
			// about int8<->float conversion rules - see https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap3.html#fundamentals-fixedfpconv
			point4uc white_clear_color;
			for (int c = 0; c < 3; ++c) {
				white_clear_color[c] = (SPECIAL_EMPTY_VALUE + c);
			}

			std::vector<double> frame_fps;
			int NO_FACE_INDEX = std::numeric_limits<int>::max();
			for (size_t frame = 0; frame < 10; ++frame) {
				avk2::InstanceContext::renderDocStartCapture("rasterization test");
				timer timer; timer.start();
				kernel_rasterization.initRender(width, height).geometry(gpu_vertices, gpu_faces)
					.setDepthAttachment(depth_buffer, emptyZ)
					.addAttachment(color_buffer, white_clear_color).addAttachment(face_id_buffer, NO_FACE_INDEX)
					.params(narrow_cast<int>(faces.size()))
					.exec();
				frame_fps.push_back(1.0/timer.elapsed());
				avk2::InstanceContext::renderDocEndCapture();
			}
			std::cout << "frame render (estimated FPS) " << stats::valuesStatsLine(frame_fps) << std::endl;

			image8u out_image = color_buffer.read();
			image32i out_face_id = face_id_buffer.read();
			image32f out_depth = depth_buffer.read();
			rassert(out_depth.width() == width && out_depth.height() == height, 520136621);
			rassert(out_face_id.width() == width && out_face_id.height() == height, 520136626);
			rassert(out_image.width() == width && out_image.height() == height, 45123412312);

			// TODO generalize methods (drop .exr float32 image, drop normalized grayscale from float32 image) + create dir from test case name
			if (!DEBUG_DIR.empty()) {
				std::string viewport_wxh = to_string(width) + "x" + to_string(height);
				debug_io::dumpImage(DEBUG_DIR + "raster_" + viewport_wxh + "_out_image.jpg", out_image);
				debug_io::dumpImage(DEBUG_DIR + "raster_" + viewport_wxh + "_out_depth.jpg", debug_io::depthMapping(out_depth));
			}

			float epsD = (width > 3) ? 0.00001f : 0.001f; // 1x1 and 3x3 leads to bigger depth epsilon (don't know why)
			std::vector<size_t> faces_npixels(faces.size(), 0);
			for (ptrdiff_t j = 0; j < height; ++j) {
				for (ptrdiff_t i = 0; i < width; ++i) {
					// coordinates of the center of the pixel
					float x = (i + 0.5f);
					float y = (j + 0.5f);

					// see rasterization rules - https://learn.microsoft.com/en-us/windows/uwp/graphics-concepts/rasterization-rules#triangle-rasterization-rules-without-multisampling
					// left and top edges of the rectangle are inclusive
					// right and bottom edges of the rectangle are exclusive
					// if width=10 and height=10 - then edges of the rectangle passes through pixel centers
					// so this predicate is tested well on such corner cases
					bool is_inside_rectangle = (x >= fromX*width && x < toX*width) && (y >= fromY*height && y < toY*height);

					if (is_inside_rectangle) {
						float expectedD = nearZ + (farZ - nearZ) * (x - fromX*width) / (toX*width - fromX*width);
						float foundD = out_depth.ptr(j)[i];
						EXPECT_IN_RANGE(foundD, expectedD-epsD, expectedD+epsD);

						int foundFaceId = out_face_id.ptr(j)[i];
						EXPECT_IN_RANGE(foundFaceId, 0, faces.size() - 1);
						faces_npixels[foundFaceId] += 1;

						for (ptrdiff_t c = 0; c < cn; ++c) {
							unsigned char foundColor = out_image.ptr(j)[cn * i + c];
							EXPECT_EQ(foundColor, SOME_COLOR_VALUE+c);
						}
					} else {
						float foundD = out_depth.ptr(j)[i];
						EXPECT_EQ(foundD, emptyZ);

						int foundFaceId = out_face_id.ptr(j)[i];
						EXPECT_EQ(foundFaceId, NO_FACE_INDEX);

						for (ptrdiff_t c = 0; c < cn; ++c) {
							unsigned char foundColor = out_image.ptr(j)[cn * i + c];
							if (avk2::isMoltenVK()) {
								// see https://github.com/KhronosGroup/MoltenVK/issues/2272 for more details
								EXPECT_IN_RANGE(foundColor, SPECIAL_EMPTY_VALUE+c, SPECIAL_EMPTY_VALUE+c+1);
							} else {
								EXPECT_EQ(foundColor, SPECIAL_EMPTY_VALUE+c);
							}
						}
					}
				}
			}
			std::cout << "per-face npixels: " << stats::vectorToString(faces_npixels) << std::endl;
			if (width > 1 && height > 1) {
				for (size_t face_id = 0; face_id < faces_npixels.size(); ++face_id) {
					EXPECT_GE(faces_npixels[face_id], 1);
				}
			}
		}
	}

	checkPostInvariants();
}
