#pragma once

#include <glm/glm.hpp>

#include "../core/util_macros.h"

namespace filter
{
	CU_DEVICE CU_INLINE double computeFilterWeight(const glm::vec2& xy)
	{
		return 1.0;
	}
}