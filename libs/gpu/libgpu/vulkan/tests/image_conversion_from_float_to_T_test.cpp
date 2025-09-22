#include <gtest/gtest.h>

#include <libimages/images.h>
#include <libgpu/shared_device_image.h>
#include <libbase/math.h>

#include <random>
#include <algorithm>

#include "test_utils.h"
#include "kernels/kernels.h"
#include "kernels/defines.h"

template <typename T>
void checkValue(unsigned int global_i, unsigned int max_global_i, T found_value, double &sum_error);

template<>
void checkValue<unsigned char>(unsigned int global_i, unsigned int max_global_i, unsigned char found_value, double &sum_error) {
	double res32f = (1 + (global_i * 255.0 / max_global_i));
	res32f = std::clamp(res32f, 1.0, 255.0);
	unsigned char expected_value = narrow_cast<unsigned char>(std::round(res32f));
	if (found_value != expected_value) {
		EXPECT_IN_RANGE(found_value, std::floor(res32f), std::ceil(res32f));
		sum_error += 1.0;
	}
}

template<>
void checkValue<unsigned short>(unsigned int global_i, unsigned int max_global_i, unsigned short found_value, double &sum_error) {
	double res32f = (1 + (global_i * 65535.0 / max_global_i));
	res32f = std::clamp(res32f, 1.0, 65535.0);
	unsigned short expected_value = narrow_cast<unsigned short>(std::round(res32f));
	if (found_value != expected_value) {
		EXPECT_IN_RANGE(found_value, std::floor(res32f), std::ceil(res32f));
		sum_error += 1.0;
	}
}

template<>
void checkValue<float>(unsigned int global_i, unsigned int max_global_i, float found_value, double &sum_error) {
	double res32f = (FLT_EPSILON + (global_i * 1.0 / max_global_i));
	res32f = std::clamp(res32f, (double) FLT_EPSILON, 1.0);
	float expected_value = res32f;
	rassert(expected_value > 0.0f, 650859056);
	rassert(expected_value >= FLT_EPSILON, 621341);
	if (found_value != expected_value) {
		float max_error = 1e-6;
		float abs_error = std::abs(expected_value - found_value);
		ASSERT_LE(abs_error, max_error);
		sum_error += abs_error;
	}
}

template <typename T>
void test_image_conversion_from_float_to_T()
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		std::vector<std::pair<size_t, size_t>> rendering_test_sizes_wh = {
				{2,		2},
				{10,	10},
				{255,	255},
				{256,	256},
				{257,	257},
				{1023,	1023},
				{1024,	1024},
				{1025,	1025},
				{9973,	9967}, // large prime numbers
		};
		for (auto rendering_test_size : rendering_test_sizes_wh) {
			const unsigned int width = rendering_test_size.first;
			const unsigned int height = rendering_test_size.second + 1; // last row will be filled with zero
			std::cout << "testing " << width << "x" << height << "..." << std::endl;
			gpu::shared_device_image_typed<T> gpu_image(width, height);

			TypedImage<T> image(width, height, 1);
			T default_color[] = {(T) 0};
			image.fill(default_color);
			gpu_image.write(image);

			avk2::KernelSource kernel(avk2::getImageConversionFromFloatToT(DataTypeTraits<T>::type()));

			struct {
				unsigned int	width, height;
			} params = {width, height};
			gpu::WorkSize worksize(VK_GROUP_SIZE_X, VK_GROUP_SIZE_Y, width, height);
			kernel.exec(params, worksize, gpu_image);

			image = gpu_image.read();
			{
				ptrdiff_t j = height - 1; // last row should be filled with zero
				for (ptrdiff_t i = 0; i < width; ++i) {
					T found_value = image.ptr(j)[i];
					rassert(found_value == (T) 0, 816139087);
				}
			}
			double sum_error = 0.0;
			size_t ntotal = 0;
			for (ptrdiff_t j = 0; j < height - 1; ++j) {
				for (ptrdiff_t i = 0; i < width; ++i) {
					T found_value = image.ptr(j)[i];
					rassert(found_value != (T) 0, 54613241324);

					unsigned int global_i = j * params.width + i;
					unsigned int max_global_i = (params.height - 1) * params.width - 1;
					rassert(global_i <= max_global_i, 56253241);

					checkValue<T>(global_i, max_global_i, found_value, sum_error);
					++ntotal;
				}
			}
			if (DataTypeTraits<T>::type() == DataType32f) {
				double avg_error = sum_error / ntotal;
				std::cout << "avg error: " << avg_error << std::endl;
				ASSERT_LE(avg_error, 1e-7);
			} else {
				size_t different_rounding = sum_error;
				std::cout << "different rounding: " << sum_error << "/" << ntotal << " ~ " << to_percent(different_rounding, ntotal) << "%" << std::endl;
				ASSERT_LE(different_rounding, ntotal * 0.04); // in practice it is ~3%
			}
		}
	}
	checkPostInvariants();
}

TEST(vulkan, checkConversionFloatToUChar)
{
	test_image_conversion_from_float_to_T<unsigned char>();
}

TEST(vulkan, checkConversionFloatToUShort)
{
	test_image_conversion_from_float_to_T<unsigned short>();
}

TEST(vulkan, checkConversionFloatToFloat)
{
	test_image_conversion_from_float_to_T<float>();
}
