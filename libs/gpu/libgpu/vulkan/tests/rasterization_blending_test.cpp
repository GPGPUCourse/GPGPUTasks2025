#include <gtest/gtest.h>

#include <libbase/timer.h>
#include <libbase/point.h>
#include <libbase/stats.h>
#include <libgpu/shared_device_buffer.h>

#include "test_utils.h"

#include "kernels/defines.h"
#include "kernels/kernels.h"

#define DEBUG_DIR std::string("")

TEST(vulkan, renderRectangleWithBlending)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		size_t test_size = 1;
		rassert(test_size >= 1, 719729832);

		size_t width  = test_size*2*1024;
		size_t height = test_size*1024;
		size_t cn = 4;
		std::cout << "evaluating viewport resolution " << width << "x" << height << "..." << std::endl;

		gpu_image32f colors_accumulator_buffer(width, height, cn);
		colors_accumulator_buffer.fill(0.0f);

		int nrectangles = 5;
		int ntimes_per_rectangle = 10;
		for (int rectangle_index = 0; rectangle_index < nrectangles; ++rectangle_index) {
			double rectangle_size = 0.5 / (1 << rectangle_index);
			double rectangle_from_border = (1.0 - rectangle_size) / 2.0;

			std::vector<point3f> vertices_xyz(4);
			float fromX = rectangle_from_border, toX = 1.0 - rectangle_from_border;
			float fromY = rectangle_from_border, toY = 1.0 - rectangle_from_border;
			float rectZ = 0.50;
			vertices_xyz[0] = point3f(fromX,	fromY,	rectZ);	// top-left
			vertices_xyz[1] = point3f(toX,		fromY,	rectZ);	// top-right
			vertices_xyz[2] = point3f(fromX,	toY,	rectZ);	// bottom-left
			vertices_xyz[3] = point3f(toX,		toY,	rectZ);	// bottom-right

			std::vector<point3u> faces(2);
			faces[0] = point3u(0, 1, 2);
			faces[1] = point3u(2, 1, 3);

			gpu::gpu_mem_vertices_xyz gpu_vertices;
			gpu_vertices.resizeN(vertices_xyz.size());
			gpu_vertices.writeN((gpu::Vertex3D*) vertices_xyz.data(), vertices_xyz.size());

			gpu::gpu_mem_32u gpu_faces;
			gpu_faces.resizeN(3 * faces.size());
			gpu_faces.writeN((unsigned int*) faces.data(), 3 * faces.size());

			avk2::KernelSource kernel_rasterization_with_blending(avk2::getRasterizeWithBlendingKernel());

			std::vector<double> frame_fps;
			for (int frame = 0; frame < ntimes_per_rectangle; ++frame) {
				timer timer; timer.start();
				kernel_rasterization_with_blending.initRender(width, height).geometry(gpu_vertices, gpu_faces)
					.addAttachment(colors_accumulator_buffer).setColorAttachmentsBlending(true)
					.exec();
				frame_fps.push_back(1.0/timer.elapsed());
			}
			std::cout << "rectangle size=" + to_string(rectangle_size) + " - frame render (estimated FPS) " << stats::valuesStatsLine(frame_fps) << std::endl;
		}

		image32f out_colors_accumulator = colors_accumulator_buffer.read();
		rassert(out_colors_accumulator.width() == width && out_colors_accumulator.height() == height && out_colors_accumulator.channels() == cn, 451412341);

		#pragma omp parallel for
		for (ptrdiff_t j = 0; j < height; ++j) {
			for (ptrdiff_t i = 0; i < width; ++i) {
				std::vector<float> channel_values = {BLENDING_RED_VALUE, BLENDING_GREEN_VALUE, BLENDING_BLUE_VALUE, BLENDING_ALPHA_VALUE};
				for (ptrdiff_t c = 0; c < cn; ++c) {
					int pixel_nrectangles_overlap = 0;
					for (int rectangle_index = 0; rectangle_index < nrectangles; ++rectangle_index) {
						double rectangle_size = 0.5 / (1 << rectangle_index);
						double rectangle_from_border = (1.0 - rectangle_size) / 2.0;

						float fromX = rectangle_from_border, toX = 1.0 - rectangle_from_border;
						float fromY = rectangle_from_border, toY = 1.0 - rectangle_from_border;
						float x = (i + 0.5) / width; float y = (j + 0.5) / height;
						if (x >= fromX && x <= toX && y >= fromY && y <= toY) {
							++pixel_nrectangles_overlap;
						}
					}
					float found_color = out_colors_accumulator.ptr(j)[cn * i + c];
					float pixel_channel_expected_value = channel_values[c] * pixel_nrectangles_overlap * ntimes_per_rectangle;
					EXPECT_EQ(found_color, pixel_channel_expected_value);
				}
			}
		}
	}

	checkPostInvariants();
}
