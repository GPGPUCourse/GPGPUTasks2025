#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#define CPU_ARCH_X86_64
#define CPU_ARCH_X86_FAMILY
#elif defined(__i386__) || defined(_M_IX86)
#define CPU_ARCH_X86
#define CPU_ARCH_X86_FAMILY
#elif defined(__aarch64__) || defined(_M_ARM64)
#define CPU_ARCH_ARM64
#define CPU_ARCH_ARM_FAMILY
#else
#error Unsupported architecture
#endif
