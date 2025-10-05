#pragma once

#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <set>
#include <mutex>
#include <vector>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <stdint.h>

#include <CL/cl.h>
#include "libbase/string_utils.h"
#include "../../../base/libbase/data_type.h"
#include <libgpu/work_size.h>
#include <libgpu/opencl/device_info.h>
#include <libgpu/opencl/utils.h>
#include <libgpu/utils.h>
#include "../../../base/libbase/thread_mutex.h"
#include <memory>
#include <map>

namespace gpu {
	class WorkSize;
	class shared_device_buffer;

	template <typename T>
	class shared_device_buffer_typed;
}

namespace ocl {

	template<class T>
	struct OpenCLType;

	template<> struct OpenCLType<int8_t>		{ typedef cl_char	type; static std::string name() { return "char";	}  static int8_t		max() { return CL_CHAR_MAX;		} static int8_t			min() { return CL_CHAR_MIN;	} };
	template<> struct OpenCLType<int16_t>		{ typedef cl_short	type; static std::string name() { return "short";	}  static int16_t		max() { return CL_SHRT_MAX;		} static int16_t		min() { return CL_SHRT_MIN;	} };
	template<> struct OpenCLType<int32_t>		{ typedef cl_int	type; static std::string name() { return "int";		}  static int32_t		max() { return CL_INT_MAX;		} static int32_t		min() { return CL_INT_MIN;	} };
	template<> struct OpenCLType<uint8_t>		{ typedef cl_uchar	type; static std::string name() { return "uchar";	}  static uint8_t		max() { return CL_UCHAR_MAX;	} static uint8_t		min() { return 0;			} };
	template<> struct OpenCLType<uint16_t>		{ typedef cl_ushort	type; static std::string name() { return "ushort";	}  static uint16_t		max() { return CL_CHAR_MAX;		} static uint16_t		min() { return 0; 			} };
	template<> struct OpenCLType<uint32_t>		{ typedef cl_uint	type; static std::string name() { return "uint";	}  static uint32_t		max() { return CL_UINT_MAX;		} static uint32_t		min() { return 0;			} };
	template<> struct OpenCLType<float>			{ typedef cl_float	type; static std::string name() { return "float";	}  static float			max() { return CL_FLT_MAX;		} static float			min() { return CL_FLT_MIN;	} };
	template<> struct OpenCLType<double>		{ typedef cl_double	type; static std::string name() { return "double";	}  static double		max() { return std::numeric_limits<double>::max();		} static double			min() { return CL_DBL_MIN;	} };

	class OpenCLEngine;

	typedef std::shared_ptr<OpenCLEngine>	sh_ptr_ocl_engine;

	class LocalMem {
	public:
		LocalMem(size_t size) : size(size) { }

		::size_t size;
	};

	class OpenCLKernelArg {
	public:
		OpenCLKernelArg() : is_null(true), size(0), value(0), cl_mem_storage(NULL) { }

		template <typename T>
		OpenCLKernelArg(const T &arg) : is_null(false), size(sizeof(arg)), value(&arg), cl_mem_storage(NULL) { }

		OpenCLKernelArg(const LocalMem &arg) : is_null(false), size(arg.size), value(0), cl_mem_storage(NULL) { }

		OpenCLKernelArg(const gpu::shared_device_buffer &arg);

		template <typename T>
		OpenCLKernelArg(const gpu::shared_device_buffer_typed<T> &arg);

		bool			is_null;
		size_t			size;
		const void *	value;
	protected:
		cl_mem 			cl_mem_storage;
	};

	class OpenCLKernel {
	public:
		OpenCLKernel();
		~OpenCLKernel();

		void		create(cl_program program, const char *kernel_name, cl_device_id device_id_=NULL);

		cl_kernel	kernel(void)			{ return kernel_;			}
		std::string kernelName(void)		{ return kernel_name_;		}
		size_t		workGroupSize(void)		{ return work_group_size_;	}

		typedef OpenCLKernelArg Arg;

		void setArgs(const Arg &arg0 = Arg(), const Arg &arg1 = Arg(), const Arg &arg2 = Arg(), const Arg &arg3 = Arg(), const Arg &arg4 = Arg(), const Arg &arg5 = Arg(), const Arg &arg6 = Arg(), const Arg &arg7 = Arg(), const Arg &arg8 = Arg(), const Arg &arg9 = Arg(), const Arg &arg10 = Arg(), const Arg &arg11 = Arg(), const Arg &arg12 = Arg(), const Arg &arg13 = Arg(), const Arg &arg14 = Arg(), const Arg &arg15 = Arg(), const Arg &arg16 = Arg(), const Arg &arg17 = Arg(), const Arg &arg18 = Arg(), const Arg &arg19 = Arg(), const Arg &arg20 = Arg(), const Arg &arg21 = Arg(), const Arg &arg22 = Arg(), const Arg &arg23 = Arg(), const Arg &arg24 = Arg(), const Arg &arg25 = Arg(), const Arg &arg26 = Arg(), const Arg &arg27 = Arg(), const Arg &arg28 = Arg(), const Arg &arg29 = Arg(), const Arg &arg30 = Arg(), const Arg &arg31 = Arg(), const Arg &arg32 = Arg(), const Arg &arg33 = Arg(), const Arg &arg34 = Arg(), const Arg &arg35 = Arg(), const Arg &arg36 = Arg(), const Arg &arg37 = Arg(), const Arg &arg38 = Arg(), const Arg &arg39 = Arg(), const Arg &arg40 = Arg())
		{
			setArg( 0, arg0);
			setArg( 1, arg1);
			setArg( 2, arg2);
			setArg( 3, arg3);
			setArg( 4, arg4);
			setArg( 5, arg5);
			setArg( 6, arg6);
			setArg( 7, arg7);
			setArg( 8, arg8);
			setArg( 9, arg9);
			setArg(10, arg10);
			setArg(11, arg11);
			setArg(12, arg12);
			setArg(13, arg13);
			setArg(14, arg14);
			setArg(15, arg15);
			setArg(16, arg16);
			setArg(17, arg17);
			setArg(18, arg18);
			setArg(19, arg19);
			setArg(20, arg20);
			setArg(21, arg21);
			setArg(22, arg22);
			setArg(23, arg23);
			setArg(24, arg24);
			setArg(25, arg25);
			setArg(26, arg26);
			setArg(27, arg27);
			setArg(28, arg28);
			setArg(29, arg29);
			setArg(30, arg30);
			setArg(31, arg31);
			setArg(32, arg32);
			setArg(33, arg33);
			setArg(34, arg34);
			setArg(35, arg35);
			setArg(36, arg36);
			setArg(37, arg37);
			setArg(38, arg38);
			setArg(39, arg39);
			setArg(40, arg40);
		}

	protected:
		void		setArg(cl_uint arg_index, size_t arg_size, const void *arg_value);

		void		setArg(cl_uint arg_index, const Arg &arg)
		{
			if (!arg.is_null)
				setArg(arg_index, arg.size, arg.value);
		}

		cl_kernel	kernel_;
		size_t		work_group_size_;
		std::string	kernel_name_;
	};

	class OpenCLEngine {
	public:
		OpenCLEngine();
		~OpenCLEngine();

		void				init(cl_device_id device_id = 0, const char *cl_params = 0, bool verbose = false);
		void				init(cl_platform_id platform_id = 0, cl_device_id device_id = 0, const char *cl_params = 0, bool verbose = false);
		cl_mem				createImage2D(size_t width, size_t height, size_t cn, DataType data_type);
		cl_sampler			createImageSampler();
		cl_mem				createBuffer(cl_mem_flags flags, size_t size);
		void				writeBuffer(cl_mem buffer, cl_bool blocking_write, size_t offset, size_t cb, const void *ptr);
		void				writeBufferRect(cl_mem buffer, cl_bool blocking_write, const size_t buffer_origin[3], const size_t host_origin[3], const size_t region[3],
								size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, const void *ptr);
		void				readBuffer(cl_mem buffer, cl_bool blocking_read, size_t offset, size_t cb, void *ptr);
		void				readBufferRect(cl_mem buffer, cl_bool blocking_write, const size_t buffer_origin[3], const size_t host_origin[3], const size_t region[3],
											size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, void *ptr);
		void				copyBuffer(cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t cb);
		void				copyBufferToImage(cl_mem src_buffer, cl_mem dst_image, size_t width, size_t height, size_t src_offset, size_t dst_x_offset, size_t dst_y_offset, bool async);
		void				readImage(cl_mem src_image, size_t width, size_t height, size_t row_pitch, void *ptr);
		void				ndRangeKernel(OpenCLKernel &kernel, cl_uint work_dim, const size_t *global_work_offset,
											const size_t *global_work_size, const size_t *local_work_size, bool synchronized = true);
		void				releaseMemObject(cl_mem memobj);

		std::string					additionalCompileOptions() const;

		void						initSupportedImageFormatsList();
		std::string					getProgramBinaryShortId(const std::string &program_name) const;
		std::string					getProgramBinaryLongId(const std::string &program_name, const std::string &additional_options) const;

		const DeviceInfo &	deviceInfo() const			{ return device_info_;				}

		cl_platform_id		platform()					{ return platform_id_;				}
		cl_device_id		device()					{ return device_id_;				}
		cl_context			context()					{ return context_;					}
		cl_command_queue	queue()						{ return command_queue_;			}

		const std::string &	deviceName() const 				{ return device_info_.device_name;				}
		size_t				maxComputeUnits() const			{ return device_info_.max_compute_units;		}
		size_t				maxWorkgroupSize() const		{ return device_info_.max_workgroup_size;		}
		size_t				maxWorkItemSizes(int dim) const	{ return device_info_.max_work_item_sizes[dim];	}
		uint64_t			maxMemAllocSize() const			{ return device_info_.max_mem_alloc_size;		}
		uint64_t			globalMemSize() const			{ return device_info_.global_mem_size;			}
		size_t				deviceAddressBits() const		{ return device_info_.device_address_bits;		}
		size_t 				wavefrontSize() const			{ return wavefront_size_;						}
		uint64_t			totalMemSize() const			{ return total_mem_size_;						}
		uint64_t			freeMem();

		std::map<int, cl_program> &		programs()	{ return programs_;	}
		std::map<int, OpenCLKernel *> &	kernels()	{ return kernels_;	}

		cl_program						findProgram(int id) const;
		OpenCLKernel *					findKernel(int id) const;

	protected:
		void				trackEvent(cl_event ev, std::string message="");

		void				createContext(cl_platform_id platform_id, cl_device_id device_id);
		void				releaseContext();

		cl_platform_id		platform_id_;
		cl_device_id		device_id_;
		cl_context			context_;
		cl_command_queue	command_queue_;

		size_t 				wavefront_size_;

		std::vector<cl_image_format>	supported_image_formats_;
		bool							supported_image_formats_inited_;
		bool							print_supported_image_formats_;

		DeviceInfo			device_info_;
		uint64_t			total_mem_size_;

		std::map<int, cl_program>		programs_;
		std::map<int, OpenCLKernel *>	kernels_;

		// Although OpenCL specification doesn't prohibit use of multiple contexts for the same
		// device, in practice drivers are not very well tested for such scenario. For example,
		// AMD drivers (at least 23.10.2) crash when calling clReleaseContext in parallel from
		// multiple threads for contexts created on the same device. The problem doesn't seem
		// to occur when clReleaseContext is called in parallel for contexts created on different devices.
		// 
		// To avoid such problem we track a list of created contexts for each device with a reference
		// count for each context. When a new context needs to be created, we first check if
		// a context for the same device already exists, and use existing context instead of
		// creating a new one.

		struct OpenCLContext {
			cl_platform_id	platform_id;
			cl_device_id	device_id;
			cl_context		context;
			size_t			count;
		};

		typedef std::map<std::pair<cl_platform_id, cl_device_id>, OpenCLContext> ContextMap;

		static ContextMap	contexts_;
		static std::mutex	contexts_mutex_;
	};

	void		oclPrintBuildLog(cl_program program);

class VersionedBinary {

public:
	VersionedBinary(const char *data, const size_t size,
					int bits, const int opencl_major_version, const int opencl_minor_version);

	const char *			data() const				{ return data_;					}
	size_t					size() const				{ return size_;					}
	size_t					deviceAddressBits() const	{ return device_address_bits_;	}
	int						openclMajorVersion() const	{ return opencl_major_version_;	}
	int						openclMinorVersion() const	{ return opencl_minor_version_;	}
	std::pair<int, int>		openclVersion() const		{ return std::make_pair(opencl_major_version_, opencl_minor_version_); }

protected:
	const char *			data_;
	const size_t			size_;
	size_t					device_address_bits_;
	const int				opencl_major_version_;
	const int				opencl_minor_version_;
};

class ProgramBinaries {
public:
	ProgramBinaries(std::vector<const VersionedBinary *> binaries, std::string program_name = std::string(), std::string defines = std::string());
	ProgramBinaries(const char *source_code, size_t source_code_length, std::string program_name = std::string(), std::string defines = std::string());

	int										id() const { return id_; }
	std::string								defines() const { return defines_; }
	const VersionedBinary*					getBinary(const std::shared_ptr<OpenCLEngine> &cl) const;
	const VersionedBinary*					getBinary(OpenCLEngine *cl) const;
	const std::string &						programName() const { return program_name_; };

protected:
	int										id_;
	std::vector<const VersionedBinary *>	binaries_;
	std::shared_ptr<VersionedBinary>		binary_with_lifetime_;
	std::string								program_name_;
	std::string								defines_;
};

class KernelSource {
public:
	KernelSource(const ocl::ProgramBinaries& program);
	KernelSource(const ocl::ProgramBinaries& program, const char *name);
	KernelSource(const ocl::ProgramBinaries& program, const std::string &name);
	KernelSource(std::shared_ptr<ocl::ProgramBinaries> program, const char *name);
	KernelSource(std::shared_ptr<ocl::ProgramBinaries> program, const std::string &name);

	typedef OpenCLKernel::Arg Arg;

	void exec(const gpu::WorkSize &ws, const Arg &arg0 = Arg(), const Arg &arg1 = Arg(), const Arg &arg2 = Arg(), const Arg &arg3 = Arg(), const Arg &arg4 = Arg(), const Arg &arg5 = Arg(), const Arg &arg6 = Arg(), const Arg &arg7 = Arg(), const Arg &arg8 = Arg(), const Arg &arg9 = Arg(), const Arg &arg10 = Arg(), const Arg &arg11 = Arg(), const Arg &arg12 = Arg(), const Arg &arg13 = Arg(), const Arg &arg14 = Arg(), const Arg &arg15 = Arg(), const Arg &arg16 = Arg(), const Arg &arg17 = Arg(), const Arg &arg18 = Arg(), const Arg &arg19 = Arg(), const Arg &arg20 = Arg(), const Arg &arg21 = Arg(), const Arg &arg22 = Arg(), const Arg &arg23 = Arg(), const Arg &arg24 = Arg(), const Arg &arg25 = Arg(), const Arg &arg26 = Arg(), const Arg &arg27 = Arg(), const Arg &arg28 = Arg(), const Arg &arg29 = Arg(), const Arg &arg30 = Arg(), const Arg &arg31 = Arg(), const Arg &arg32 = Arg(), const Arg &arg33 = Arg(), const Arg &arg34 = Arg(), const Arg &arg35 = Arg(), const Arg &arg36 = Arg(), const Arg &arg37 = Arg(), const Arg &arg38 = Arg(), const Arg &arg39 = Arg(), const Arg &arg40 = Arg());
	void execAsynchronized(const gpu::WorkSize &ws, const Arg &arg0 = Arg(), const Arg &arg1 = Arg(), const Arg &arg2 = Arg(), const Arg &arg3 = Arg(), const Arg &arg4 = Arg(), const Arg &arg5 = Arg(), const Arg &arg6 = Arg(), const Arg &arg7 = Arg(), const Arg &arg8 = Arg(), const Arg &arg9 = Arg(), const Arg &arg10 = Arg(), const Arg &arg11 = Arg(), const Arg &arg12 = Arg(), const Arg &arg13 = Arg(), const Arg &arg14 = Arg(), const Arg &arg15 = Arg(), const Arg &arg16 = Arg(), const Arg &arg17 = Arg(), const Arg &arg18 = Arg(), const Arg &arg19 = Arg(), const Arg &arg20 = Arg(), const Arg &arg21 = Arg(), const Arg &arg22 = Arg(), const Arg &arg23 = Arg(), const Arg &arg24 = Arg(), const Arg &arg25 = Arg(), const Arg &arg26 = Arg(), const Arg &arg27 = Arg(), const Arg &arg28 = Arg(), const Arg &arg29 = Arg(), const Arg &arg30 = Arg(), const Arg &arg31 = Arg(), const Arg &arg32 = Arg(), const Arg &arg33 = Arg(), const Arg &arg34 = Arg(), const Arg &arg35 = Arg(), const Arg &arg36 = Arg(), const Arg &arg37 = Arg(), const Arg &arg38 = Arg(), const Arg &arg39 = Arg(), const Arg &arg40 = Arg());
	void execSubdivided(const gpu::WorkSize &ws, const Arg &arg0 = Arg(), const Arg &arg1 = Arg(), const Arg &arg2 = Arg(), const Arg &arg3 = Arg(), const Arg &arg4 = Arg(), const Arg &arg5 = Arg(), const Arg &arg6 = Arg(), const Arg &arg7 = Arg(), const Arg &arg8 = Arg(), const Arg &arg9 = Arg(), const Arg &arg10 = Arg(), const Arg &arg11 = Arg(), const Arg &arg12 = Arg(), const Arg &arg13 = Arg(), const Arg &arg14 = Arg(), const Arg &arg15 = Arg(), const Arg &arg16 = Arg(), const Arg &arg17 = Arg(), const Arg &arg18 = Arg(), const Arg &arg19 = Arg(), const Arg &arg20 = Arg(), const Arg &arg21 = Arg(), const Arg &arg22 = Arg(), const Arg &arg23 = Arg(), const Arg &arg24 = Arg(), const Arg &arg25 = Arg(), const Arg &arg26 = Arg(), const Arg &arg27 = Arg(), const Arg &arg28 = Arg(), const Arg &arg29 = Arg(), const Arg &arg30 = Arg(), const Arg &arg31 = Arg(), const Arg &arg32 = Arg(), const Arg &arg33 = Arg(), const Arg &arg34 = Arg(), const Arg &arg35 = Arg(), const Arg &arg36 = Arg(), const Arg &arg37 = Arg(), const Arg &arg38 = Arg(), const Arg &arg39 = Arg(), const Arg &arg40 = Arg());

	void precompile(bool printLog=false);
	void precompile(const std::shared_ptr<OpenCLEngine> &cl, bool printLog=false);

	size_t getMaxWorkgroupSize();

protected:
	int getNextKernelId();

	OpenCLKernel *getKernel(const std::shared_ptr<OpenCLEngine> &cl, bool printLog=false);

	const  ocl::ProgramBinaries				&program_;
	std::shared_ptr<ocl::ProgramBinaries>	program_lifetime_;

	int				id_;
	std::string		name_;
};

}
