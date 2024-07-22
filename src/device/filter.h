#pragma once

#include <glm/vec2.hpp>

#include "../core/util.h"

namespace filter
{
	CU_DEVICE CU_INLINE double computeFilterWeight(const glm::vec2& xy)
	{
		return 1.0;
	}
}
