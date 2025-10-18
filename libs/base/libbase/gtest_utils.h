#pragma once

#include <string>


bool isDebuggerAttached();

namespace gtest {
	void forceBreakOnFailure(); // this is useful to debug gtest EXPECTs' failures - because application will be stopped on the first failure

	std::string getCaseName();
}
