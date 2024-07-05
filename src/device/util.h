#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cmath>

#include <glm/vec3.hpp>
#include "glm/geometric.hpp"

#include "../core/util_macros.h"

namespace utility
{
	//Modified version of https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_6.pdf
	CU_DEVICE CU_INLINE glm::vec3 offsetRay(const glm::vec3& p, const glm::vec3& n)
	{
		constexpr float originInterval{ 1.0f / 32.0f };
		constexpr float floatScale{ 1.0f / 65536.0f };
		constexpr float intScale{ 256.0f };

		glm::ivec3 intOff{ intScale * n.x, intScale * n.y, intScale * n.z };

		glm::vec3 pIntOff{ __int_as_float(__float_as_int(p.x) + ((p.x < 0.0f) ? -intOff.x : intOff.x)),
						   __int_as_float(__float_as_int(p.y) + ((p.y < 0.0f) ? -intOff.y : intOff.y)),
						   __int_as_float(__float_as_int(p.z) + ((p.z < 0.0f) ? -intOff.z : intOff.z)) };

		return glm::vec3{ fabsf(p.x) < originInterval ? p.x + floatScale * n.x : pIntOff.x,
						  fabsf(p.y) < originInterval ? p.y + floatScale * n.y : pIntOff.y,
						  fabsf(p.z) < originInterval ? p.z + floatScale * n.z : pIntOff.z };
	}

	CU_DEVICE CU_INLINE glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n)
	{
		return -v + 2.0f * glm::dot(v, n) * n;
	}
}
