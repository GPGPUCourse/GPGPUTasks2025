#include <gtest/gtest.h>

#include <libgpu/shared_device_buffer.h>

#include "test_utils.h"

#include "kernels/kernels.h"
#include "kernels/defines.h"

void writeAtIndex(unsigned int n, unsigned int buffer, int index)
{
	gpu::gpu_mem_32u gpu_buffer0, gpu_buffer1;
	gpu_buffer0.resizeN(n);
	gpu_buffer1.resizeN(n);

	avk2::KernelSource kernel(avk2::getWriteValueAtIndexKernel());

	struct {
		unsigned int chosenBuffer;
		int index;
		unsigned int value;
	} params = {buffer, index, 239017};

	kernel.exec(params, gpu::WorkSize(VK_GROUP_SIZE, 1), gpu_buffer0, gpu_buffer1);
}

TEST(vulkan, bufferMagicGuards)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		if (!context.isMemoryGuardsEnabled() || !context.isMemoryGuardsChecksAfterKernelsEnabled()) {
			std::cout << "memory guards were forced to be enabled (with checks after kernels) - because this unit-test evaluates their implementation" << std::endl;
			context.setMemoryGuards(true);
			context.setMemoryGuardsChecksAfterKernels(true);
		}

		std::vector<unsigned int> ns = {1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513};
		for (unsigned int buffer = 0; buffer < 2; ++buffer) {
			for (unsigned int n: ns) {
				gpu::gpu_mem_32u gpu_buffer;
				gpu_buffer.resizeN(n);
				rassert(gpu_buffer.checkMagicGuards("freshly allocated buffer"), 200081508);

				std::vector<unsigned int> data(n + 1, 239019);
				gpu_buffer.writeN(data.data(), n);
				rassert(gpu_buffer.checkMagicGuards("after correct write"), 417780940);
				gpu_buffer.writeN(data.data(), n + 1);
				rassert(!gpu_buffer.checkMagicGuards("check should fail after out-of-bounds write"), 464479709);

				writeAtIndex(n, buffer, 0);
				writeAtIndex(n, buffer, n / 2);
				writeAtIndex(n, buffer, n - 1);

				EXPECT_THROW(writeAtIndex(n, buffer, n + 0), assertion_error);
				EXPECT_THROW(writeAtIndex(n, buffer, n + 1), assertion_error);
				EXPECT_THROW(writeAtIndex(n, buffer, n + 2), assertion_error);
				EXPECT_THROW(writeAtIndex(n, buffer, n + 3), assertion_error);
				// let's check out-of-bounds write to the last uint in the small magic guard:
				EXPECT_THROW(writeAtIndex(n, buffer, n + GPU_BUFFER_SMALL_MAGIC_GUARD_NBYTES/sizeof(unsigned int) - 1), assertion_error);

				if (n * sizeof(unsigned int) > GPU_BUFFER_BIG_MAGIC_GUARD_NBYTES) {
					// big buffers have bigger magic guards
					EXPECT_THROW(writeAtIndex(n, buffer, n + GPU_BUFFER_BIG_MAGIC_GUARD_NVALUES - 1), assertion_error);
				}
			}
		}
	}

	checkPostInvariants();
}
