#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/util_macros.h"

namespace sampling
{
	namespace disk
	{
		CU_DEVICE CU_INLINE glm::vec2 sampleUniform2D(glm::vec2 uv)
		{
			uv = uv * 2.0f - 1.0f;
			float theta{};
			float r{};
			if (glm::abs(uv.x) > glm::abs(uv.y))
			{
				r = uv.x;
				theta = (glm::pi<float>() / 4.0f) * (uv.y / uv.x);
			}
			else
			{
				r = uv.y;
				theta = (glm::pi<float>() / 2.0f) - (glm::pi<float>() / 4.0f) * (uv.x / uv.y);
			}
			uv.x = r * glm::cos(theta);
			uv.y = r * glm::sin(theta);
			return uv;
		}
		CU_DEVICE CU_INLINE glm::vec3 sampleUniform3D(const glm::vec2& uv, const glm::quat& frame)
		{
			glm::mat3 matFrame{ glm::mat3_cast(frame) };
			glm::vec2 xy{ sampleUniform2D(uv) };
			return matFrame[0] * xy.x + matFrame[1] * xy.y;
		}
	}
}
