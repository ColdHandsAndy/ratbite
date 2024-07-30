#pragma once

#include <cuda/std/cstdint>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/util.h"

enum class LightType : uint16_t
{
	NONE,
	DISK,
	SPHERE,
	SKY,

	DESC,
};
CU_CONSTANT CU_DEVICE inline constexpr uint32_t KSampleableLightCount{ 2 };
CU_CONSTANT CU_DEVICE inline constexpr LightType KOrderedTypes[]{ LightType::SPHERE, LightType::DISK };
CU_CONSTANT CU_DEVICE inline constexpr uint32_t KSphereLightIndex{ 0 };
CU_CONSTANT CU_DEVICE inline constexpr uint32_t KDiskLightIndex{ 1 };
struct DiskLightData
{
	glm::vec3 position{};
	float powerScale{};
	glm::quat frame{};
	float radius{};
	uint16_t materialIndex{};
};
struct SphereLightData
{
	glm::vec3 position{};
	float powerScale{};
	glm::quat frame{};
	float radius{};
	uint16_t materialIndex{};
};
