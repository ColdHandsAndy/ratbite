#pragma once

#include <cuda/std/cmath>
#include <glm/vec2.hpp>
#include <glm/common.hpp>
#include <glm/exponential.hpp>

#include "../core/util.h"

namespace filter
{
	namespace gaussian
	{
		CU_DEVICE CU_INLINE glm::vec2 sampleDistribution(const glm::vec2& uv)
		{
			constexpr float rad{ 1.5f };
			constexpr float sigma{ 0.5f };
			const float uTH{ cuda::std::expf(-(rad * rad) / (2.0f * sigma * sigma)) };
			const float u{ cuda::std::fmaxf(0.00001f, uv.x * (1.0f - uTH) + uTH) };
			const float v{ uv.y };

			float m{ sigma * cuda::std::sqrtf(-2.0f * cuda::std::logf(u)) };
			float nu{ m * cuda::std::cosf(2.0f * glm::pi<float>() * v) };
			float nv{ m * cuda::std::sinf(2.0f * glm::pi<float>() * v) };

			return glm::vec2{nu, nv};
		}
	}
	CU_DEVICE CU_INLINE double computeFilterWeight(const glm::vec2& xy)
	{
		return 1.0;
	}
}
