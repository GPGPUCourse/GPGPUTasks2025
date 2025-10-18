#include <gtest/gtest.h>

#include "test_utils.h"


int main(int argc, char **argv) {
	if (isDebuggerAttached()) {
		std::cout << "attached debugger detected" << std::endl;
		gtest::forceBreakOnFailure();
	}

	::testing::InitGoogleTest(&argc, argv);
	int code = RUN_ALL_TESTS();

	// we need to gracefully clear Vulkan context before it is too late (otherwise we encounter segfault on some systems)
	std::cout << "clear Vulkan global instance context..." << std::endl;
	avk2::InstanceContext::clearGlobalInstanceContext();
	std::cout << "instance context cleared" << std::endl;

	return code;
}
