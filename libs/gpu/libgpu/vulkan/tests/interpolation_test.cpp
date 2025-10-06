#include <gtest/gtest.h>

#include <libimages/images.h>
#include <libbase/data_type.h>
#include <libbase/timer.h>
#include <libbase/point.h>
#include <libgpu/shared_device_image.h>
//#include <image/filter/filter.h>
#include <libgpu/vulkan/vk/common_host.h>

#include <random>

#include "test_utils.h"
#include "kernels/kernels.h"
#include "kernels/defines.h"

template <typename T>
void test_image_read_write(size_t cn)
{
	Random r(239);
	DataType type = DataTypeTraits<T>::type();
	ptrdiff_t min_value, max_value;
	if (std::numeric_limits<T>::is_integer) {
		min_value = std::numeric_limits<T>::min();
		max_value = std::numeric_limits<T>::max();
	} else {
		min_value = -MAX_F32_USED_VALUE;
		max_value =  MAX_F32_USED_VALUE;
	}

	std::cout << "evaluating type=" << cn << "x" << typeName(type) << " (with values in range [" << to_string(min_value) << "; " << max_value << "])" << std::endl;

	const size_t width = 1025;
	const size_t height = 1023;

	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		TypedImage<T> image_a(width, height, cn);
		for (ptrdiff_t j = 0; j < height; ++j) {
			for (ptrdiff_t i = 0; i < width; ++i) {
				for (ptrdiff_t c = 0; c < cn; ++c) {
					image_a.ptr(j)[cn * i + c] = generate_random_color<T>(r, min_value, max_value);
				}
			}
		}

		gpu::shared_device_image gpu_image_a;
		gpu_image_a.resize(width, height, cn, type);

		double data_size = width * height * cn * sizeof(T);
		timer pcie_timer;

		pcie_timer.restart();
		gpu_image_a.write(image_a);
		std::cout << "RAM --PCI-E-> VRAM " << width << "x" << height << "x" << cn << "x" << typeName(type) << " image write bandwidth: " << (data_size / pcie_timer.elapsed()) / (1024 * 1024) << " MB/s" << std::endl;

		pcie_timer.restart();
		TypedImage<T> image_a_readed = gpu_image_a.read();
		std::cout << "RAM <-PCI-E-- VRAM " << width << "x" << height << "x" << cn << "x" << typeName(type) << " image read bandwidth: " << (data_size / pcie_timer.elapsed()) / (1024 * 1024) << " MB/s" << std::endl;

		for (ptrdiff_t j = 0; j < height; ++j) {
			for (ptrdiff_t i = 0; i < width; ++i) {
				for (ptrdiff_t c = 0; c < cn; ++c) {
					T a = image_a.ptr(j)[cn * i + c];
					T a_readed = image_a_readed.ptr(j)[cn * i + c];
					rassert(a_readed == a, 804440611);
				}
			}
		}
	}

	checkPostInvariants();
}

template <typename T>
void test_image_interpolation(size_t cn, float dx_shift, float dy_shift)
{
	Random r;
	r.seed(239);
	DataType type = DataTypeTraits<T>::type();
	ptrdiff_t min_value, max_value;
	if (std::numeric_limits<T>::is_integer) {
		min_value = std::numeric_limits<T>::min();
		max_value = std::numeric_limits<T>::max();
	} else {
		min_value = -MAX_F32_USED_VALUE;
		max_value = MAX_F32_USED_VALUE;
	}

	std::cout << "evaluating type=" << cn << "x" << typeName(type) << " (with values in range [" << to_string(min_value) << "; " << max_value << "])" << std::endl;

	const size_t width = 1025;
	const size_t height = 1023;
	const size_t res_width = 2 * width;
	const size_t res_height = 2 * height;

	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		TypedImage<T> image_a(width, height, cn);
		for (ptrdiff_t j = 0; j < height; ++j) {
			for (ptrdiff_t i = 0; i < width; ++i) {
				for (ptrdiff_t c = 0; c < cn; ++c) {
					image_a.ptr(j)[cn * i + c] = generate_random_color<T>(r, min_value, max_value);
				}
			}
		}

		gpu::shared_device_image gpu_image_a, gpu_image_b;
		gpu_image_a.resize(width, height, cn, type);
		gpu_image_b.resize(res_width, res_height, cn, type);

		gpu_image_a.write(image_a);

		avk2::KernelSource kernel_interpolation(avk2::getInterpolationKernel(type, cn));

		bool is_zero_shift = (dx_shift == 0.0f && dy_shift == 0.0f);
		struct {
			unsigned int	width, height;
			float			scale;
			float			dx, dy;
			unsigned int	is_zero_shift;
		} params = {res_width, res_height, 1.0f, 11.0f+dx_shift, -5.0f+dy_shift, is_zero_shift};
		static_assert(sizeof(params) == 3*sizeof(unsigned int) + 3*sizeof(float), "3514123123");

		gpu::WorkSize worksize(VK_GROUP_SIZE_X, VK_GROUP_SIZE_Y, gpu_image_b.width(), gpu_image_b.height());
		kernel_interpolation.exec(params, worksize, gpu_image_a, gpu_image_b);

		TypedImage<T> image_b = gpu_image_b.read();
		double nvalues = 0.0;
		double sum_diff = 0.0;
		double max_diff = 0.0;
		for (ptrdiff_t j = 0; j < image_b.height(); ++j) {
			for (ptrdiff_t i = 0; i < image_b.width(); ++i) {
				for (ptrdiff_t c = 0; c < cn; ++c) {
					point2d pt1 = point2d(i + 0.5, j + 0.5);
					point2d pt0 = (pt1 / (double) params.scale) - point2d(params.dx, params.dy);
					T b = image_b.ptr(j)[cn * i + c];
					if (pt0.x < 0 || pt0.x >= width || pt0.y < 0 || pt0.y >= height) {
						rassert(b == SPECIAL_EMPTY_VALUE, 721367219);
					} else {
						if (is_zero_shift) {
							T a = image_a.ptr((ptrdiff_t)pt0.y)[cn * (ptrdiff_t)pt0.x + c];
							rassert(b == a, 4561455125);
						}

						T a = image_a.sample_casted(pt0.x, pt0.y, c);
						double diff = std::abs(1.0*a - 1.0*b);
						nvalues += 1;
						sum_diff += diff;
						max_diff = std::max(max_diff, diff);
					}
				}
			}
		}
		std::cout << "avg diff=" << (sum_diff / nvalues) << " max diff=" << max_diff << std::endl;
		double error_threshold = 0;
		if (!is_zero_shift) {
			if (std::is_same_v<T, unsigned char>) {
				error_threshold = 1;   // = 0.0039 * 255
			} else if (std::is_same_v<T, unsigned short>) {
				error_threshold = 183; // = 0.0028 * 65535
			} else if (std::is_same_v<T, float>) {
				error_threshold = MAX_F32_USED_VALUE / 150.0f;
			} else {
				rassert(false, 316006651);
			}
		}
		rassert(max_diff <= error_threshold, 3451254125412);
	}

	checkPostInvariants();
}

TEST(vulkan, imageReadWrite)
{
	for (size_t cn = 1; cn <= VK_MAX_NCHANNELS; ++cn) {
		test_image_read_write<unsigned char>	(cn);
		test_image_read_write<unsigned short>	(cn);
		test_image_read_write<float>			(cn);
	}
}

TEST(vulkan, interpolateImageWithZeroShift)
{
	float dx_shift = 0.0f;
	float dy_shift = 0.0f;
	for (size_t cn = 1; cn <= VK_MAX_NCHANNELS; ++cn) {
		test_image_interpolation<unsigned char>		(cn, dx_shift, dy_shift);
		test_image_interpolation<unsigned short>	(cn, dx_shift, dy_shift);
		test_image_interpolation<float>				(cn, dx_shift, dy_shift);
	}
}

TEST(vulkan, interpolateImageWithShift)
{
	float dx_shift = 0.3f;
	float dy_shift = 0.8f;
	for (size_t cn = 1; cn <= VK_MAX_NCHANNELS; ++cn) {
		test_image_interpolation<unsigned char>		(cn, dx_shift, dy_shift);
		test_image_interpolation<unsigned short>	(cn, dx_shift, dy_shift);
		test_image_interpolation<float>				(cn, dx_shift, dy_shift);
	}
}

