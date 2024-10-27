#pragma once

#include <cuda/std/cstdint>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "../core/util.h"
#include "../device/sampling.h"

struct Ray
{
	glm::vec3 o{};
	glm::vec3 d{};
};

CU_DEVICE CU_INLINE Ray generatePinholeCameraDirection(const glm::vec2& sampleCoordinate,
		const glm::vec2& invRes, const glm::vec2& perspectiveScale,
		const glm::vec3& camU, const glm::vec3& camV, const glm::vec3& camW)
{
		const float xScale{ 2.0f * sampleCoordinate.x * invRes.x - 1.0f };
		const float zScale{ 2.0f * sampleCoordinate.y * invRes.y - 1.0f };
		glm::vec3 rD{ glm::normalize(camV
		+ camU * xScale * perspectiveScale.x
		+ camW * zScale * perspectiveScale.y) };

		return Ray{.o = glm::vec3{0.0f}, .d = glm::vec3{rD}};
}
CU_DEVICE CU_INLINE Ray generateThinLensCamera(const glm::vec2& sampleCoordinate,
		const glm::vec2& lensSample, float focusDistance, float appertureRadius,
		const glm::vec2& invRes, const glm::vec2& perspectiveScale,
		const glm::vec3& camU, const glm::vec3& camV, const glm::vec3& camW)
{
		const float xSample{ (2.0f * sampleCoordinate.x * invRes.x - 1.0f) * perspectiveScale.x };
		const float zSample{ (2.0f * sampleCoordinate.y * invRes.y - 1.0f) * perspectiveScale.y };

		const glm::vec3 focusPoint{ glm::vec3{xSample, zSample, 1.0f} * focusDistance };
		const glm::vec3 lensPoint{ glm::vec3{sampling::disk::sampleUniform2DPolar(lensSample) * appertureRadius, 0.0f} };

		glm::vec3 rO{ camU * lensPoint.x + camW * lensPoint.y };
		glm::vec3 rD{ glm::normalize(focusPoint - lensPoint) };
		rD = camU * rD.x + camW * rD.y + camV * rD.z;

		return Ray{.o = rO, .d = rD};
}
