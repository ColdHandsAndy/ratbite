#pragma once

#include <cuda/std/cmath>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/util_macros.h"

namespace sampling
{
	namespace disk
	{
		CU_DEVICE CU_INLINE glm::vec2 sampleUniform2DConcentric(glm::vec2 uv)
		{
			uv = uv * 2.0f - 1.0f;
			float theta{};
			float r{};
			if (cuda::std::fabs(uv.x) > cuda::std::fabs(uv.y))
			{
				r = uv.x;
				theta = (glm::pi<float>() / 4.0f) * (uv.y / uv.x);
			}
			else
			{
				r = uv.y;
				theta = (glm::pi<float>() / 2.0f) - (glm::pi<float>() / 4.0f) * (uv.x / uv.y);
			}
			uv.x = r * cuda::std::cosf(theta);
			uv.y = r * cuda::std::sinf(theta);
			return uv;
		}
		CU_DEVICE CU_INLINE glm::vec2 sampleUniform2DPolar(const glm::vec2& uv)
		{
			float r{ cuda::std::sqrtf(uv.x) };
			float theta{ 2.0f * glm::pi<float>() * uv.y };
			return glm::vec2{r * cuda::std::cosf(theta), r * cuda::std::sinf(theta) };
		}
		CU_DEVICE CU_INLINE glm::vec3 sampleUniform3D(const glm::vec2& uv, const glm::quat& frame)
		{
			glm::mat3 matFrame{ glm::mat3_cast(frame) };
			glm::vec2 xy{ sampleUniform2DConcentric(uv) };
			return matFrame[0] * xy.x + matFrame[1] * xy.y;
		}
	}
}
