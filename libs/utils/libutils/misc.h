#pragma once

#include "libbase/string_utils.h"
#include <CL/cl.h>
#include <libgpu/device.h>
#include <libgpu/context.h>
#include <libgpu/opencl/device_info.h>
#include <libgpu/opencl/engine.h>

#include <string>
#include <limits>
#include <iostream>
#include <stdexcept>

#define DEVICE_NOT_SUPPORT_API  std::string("Device doesn't support requested API")
#define CODE_IS_NOT_IMPLEMENTED std::string("Code is not implemented yet")

namespace gpu {
	void printDeviceInfo(const gpu::Device &device);

	gpu::Device chooseGPUDevice(const std::vector<gpu::Device> &devices, int argc, char **argv);

        gpu::Context activateContext(const gpu::Device& device, gpu::Context::Type api);
}

namespace ocl {

	class Kernel {
	public:
		Kernel() {}

		Kernel(const char *source_code, size_t source_code_length, std::string kernel_name,
			   std::string defines = std::string())
		{
			init(source_code, source_code_length, kernel_name, defines);
		}

		void init(const char *source_code, size_t source_code_length, std::string kernel_name,
				  std::string defines = std::string())
		{
			program_ = std::make_shared<ocl::ProgramBinaries>(source_code, source_code_length, kernel_name, defines);
			kernel_ = std::make_shared<ocl::KernelSource>(program_, kernel_name);
		}

		void compile(bool printLog=false)
		{
			if (!kernel_)
				throw std::runtime_error("Null kernel!");
			kernel_->precompile(printLog);
		}

		typedef ocl::OpenCLKernel::Arg Arg;

		void exec(const gpu::WorkSize &ws, const Arg &arg0 = Arg(), const Arg &arg1 = Arg(), const Arg &arg2 = Arg(), const Arg &arg3 = Arg(), const Arg &arg4 = Arg(), const Arg &arg5 = Arg(), const Arg &arg6 = Arg(), const Arg &arg7 = Arg(), const Arg &arg8 = Arg(), const Arg &arg9 = Arg(), const Arg &arg10 = Arg(), const Arg &arg11 = Arg(), const Arg &arg12 = Arg(), const Arg &arg13 = Arg(), const Arg &arg14 = Arg(), const Arg &arg15 = Arg(), const Arg &arg16 = Arg(), const Arg &arg17 = Arg(), const Arg &arg18 = Arg(), const Arg &arg19 = Arg(), const Arg &arg20 = Arg(), const Arg &arg21 = Arg(), const Arg &arg22 = Arg(), const Arg &arg23 = Arg(), const Arg &arg24 = Arg(), const Arg &arg25 = Arg(), const Arg &arg26 = Arg(), const Arg &arg27 = Arg(), const Arg &arg28 = Arg(), const Arg &arg29 = Arg(), const Arg &arg30 = Arg(), const Arg &arg31 = Arg(), const Arg &arg32 = Arg(), const Arg &arg33 = Arg(), const Arg &arg34 = Arg(), const Arg &arg35 = Arg(), const Arg &arg36 = Arg(), const Arg &arg37 = Arg(), const Arg &arg38 = Arg(), const Arg &arg39 = Arg(), const Arg &arg40 = Arg())
		{
			if (!kernel_)
				throw std::runtime_error("Null kernel!");
			kernel_->exec(ws, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40);
		}

	private:
		std::shared_ptr<ocl::ProgramBinaries> program_;
		std::shared_ptr<ocl::KernelSource> kernel_;
	};
}
