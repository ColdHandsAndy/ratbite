#include <cuda_runtime.h>
#include <cuda/std/cstdint>

#include <glm/glm.hpp>

#include "../core/util_macros.h"

CU_DEVICE CU_INLINE void tonemap(glm::vec3& col)
{
	float a{ 2.51f };
	float b{ 0.03f };
	float c{ 2.43f };
	float d{ 0.59f };
	float e{ 0.14f }; 
	col = glm::clamp((col * (a * col + b)) / (col * (c * col + d) + e), 0.0f, 1.0f);
}

extern "C" __global__ void resolveImage(const uint32_t winWidth, const uint32_t winHeight, const glm::mat3 colorspaceTransform, const glm::dvec4* renderData, cudaSurfaceObject_t presentData)
{
	uint32_t x{ blockIdx.x * blockDim.x + threadIdx.x };
	uint32_t y{ blockIdx.y * blockDim.y + threadIdx.y };
	if (x >= winWidth || y >= winHeight)
		return;

	glm::dvec4 data{ renderData[y * winWidth + x] };
	double normval{ 1.0 / data.w };
	glm::vec3 normalized{ static_cast<float>(data.x * normval), 
						  static_cast<float>(data.y * normval), 
						  static_cast<float>(data.z * normval) };

	glm::vec3 color{ colorspaceTransform * normalized };

	tonemap(color);

	uchar4 res{ 
		static_cast<uint8_t>(color.x * 255.99999f),
		static_cast<uint8_t>(color.y * 255.99999f),
		static_cast<uint8_t>(color.z * 255.99999f),
		255,
	};

	constexpr uint32_t byteSize{ sizeof(res.x) * 4 };
	surf2Dwrite(res, presentData, x * byteSize, y);
	
	return;
}
