#pragma once

#include <cstdint>
#include <iostream>
#include <string_view>
#include <format>

#include <cuda_runtime.h>
#include <optix.h>

#define CUDA_CHECK(call)			checkCUDA(call, #call, __FILE__, __LINE__)
#define CUDA_SYNC_CHECK()			cudaSyncCheck(__FILE__, __LINE__)
#define OPTIX_CHECK(call)			checkOptix(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK_LOG(call)		{ char OPTIX_LOG[2048]{}; size_t OPTIX_LOG_SIZE{}; checkOptixLog(call, OPTIX_LOG, OPTIX_LOG_SIZE, #call, __FILE__, __LINE__); }

#define R_ASSERT(cond)				if (!static_cast<bool>(cond)) \
									{ \
										std::cerr << std::format("Assertion failed.\n\tFile: {}\n\tLine: {}\n", __FILE__, __LINE__); \
										std::abort(); \
									}
#define R_ASSERT_LOG(cond, msg)		if (!static_cast<bool>(cond)) \
									{ \
										std::cerr << std::format("Assertion failed.\n\tFile: {}\n\tLine: {}\n\tMessage: {}\n", __FILE__, __LINE__, msg); \
										std::abort(); \
									}
#define R_LOG(msg)					std::cerr << std::format("Log message.\n\tFile: {}\n\tLine: {}\n\tMessage: {}\n", __FILE__, __LINE__, msg);
#define R_LOG_COND(cond, msg)		if (!static_cast<bool>(cond)) \
									{ \
										std::cerr << std::format("Log message.\n\tFile: {}\n\tLine: {}\n\tMessage: {}\n", __FILE__, __LINE__, msg); \
									}

inline void checkCUDA(cudaError_t error, const char* call, const char* file, uint32_t line)
{
	if (error != cudaSuccess)
		std::cerr << std::format("CUDA call ({}) failed.\n\tError: {}\n\tFile: {}\n\tLine: {}\n", call, cudaGetErrorString(error), file, line);
}
inline void checkCUDA(CUresult error, const char* call, const char* file, uint32_t line)
{
	if (error != CUDA_SUCCESS)
	{
		const char* errStr{};
		cuGetErrorString(error, &errStr);
		std::cerr << std::format("CUDA call ({}) failed.\n\tError: {}\n\tFile: {}\n\tLine: {}\n", call, errStr, file, line);
	}
}
inline void cudaSyncCheck(const char* file, uint32_t line)
{
	cudaDeviceSynchronize();
	cudaError_t error{ cudaGetLastError() };
	if (error != cudaSuccess)
		std::cerr << std::format("CUDA error.\n\tFile: {}\n\tLine: {}\n", cudaGetErrorString(error), file, line);
}
inline void checkOptix(OptixResult res, const char* call, const char* file, uint32_t line)
{
	if (res != OPTIX_SUCCESS)
		std::cerr << std::format("OptiX call ({}) failed.\n\tError: {}\n\tFile: {}\n\tLine: {}\n", call, static_cast<int>(res), file, line);
}
inline void checkOptixLog(OptixResult res, const char* log, size_t logSize, const char* call, const char* file, uint32_t line)
{
	if (res != OPTIX_SUCCESS)
		std::cerr << std::format("OptiX call ({}) failed.\n\tError: {}\n\tLog: {}\n\tFile: {}\n\tLine: {}\n", call, static_cast<int>(res), std::string_view{ log, logSize }, file, line);
}
