#include "gtest_utils.h"

#include <gtest/gtest.h>

#if defined(__linux__)
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>
#elif defined(WIN32)
#include <windows.h>
#endif

bool isDebuggerAttached()
{
#if defined(__linux__)
	// see https://stackoverflow.com/a/24969863
	char buf[4096];

	const int status_fd = open("/proc/self/status", O_RDONLY);
	if (status_fd == -1)
		return false;

	const ssize_t num_read = read(status_fd, buf, sizeof(buf) - 1);
	close(status_fd);

	if (num_read <= 0)
		return false;

	buf[num_read] = '\0';
	constexpr char tracerPidString[] = "TracerPid:";
	const auto tracer_pid_ptr = strstr(buf, tracerPidString);
	if (!tracer_pid_ptr)
		return false;

	for (const char* characterPtr = tracer_pid_ptr + sizeof(tracerPidString) - 1; characterPtr <= buf + num_read; ++characterPtr)
	{
		if (isspace(*characterPtr))
			continue;
		else
			return isdigit(*characterPtr) != 0 && *characterPtr != '0';
	}
	return false;
#elif defined(WIN32)
	return IsDebuggerPresent(); // see https://stackoverflow.com/a/78123395
#elif defined(__APPLE__)
	return false; // unimplemented
#endif
}

void gtest::forceBreakOnFailure()
{
	// this is useful to debug gtest EXPECTs' failures - because application will be stopped on the first failure
	testing::GTEST_FLAG(break_on_failure) = true;
}

std::string gtest::getCaseName()
{
	return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}
