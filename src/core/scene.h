#pragma once

#include <cstdint>
#include <array>

#include <optix_types.h>

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/material.h"
#include "../core/spectral.h"

struct SceneData
{
	constexpr static uint32_t triangleCount{ 34 };
	constexpr static uint32_t geometryMaterialCount{ 5 };
	constexpr static uint32_t trianglePrimSBTCount{ geometryMaterialCount };

	const std::array<glm::vec4, triangleCount * 3> vertices
	{ {
		// Floor
		{    -0.0f,    0.0f,    0.0f, 0.0f  },
		{    -0.0f,    0.0f,  559.2f, 0.0f  },
		{  -556.0f,    0.0f,  559.2f, 0.0f  },

		{    -0.0f,    0.0f,    0.0f, 0.0f  },
		{  -556.0f,    0.0f,  559.2f, 0.0f  },
		{  -556.0f,    0.0f,    0.0f, 0.0f  },

		// Ceiling
		{    -0.0f,  548.8f,    0.0f, 0.0f  },
		{  -556.0f,  548.8f,    0.0f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },

		{    -0.0f,  548.8f,    0.0f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },
		{    -0.0f,  548.8f,  559.2f, 0.0f  },

		// Back wall
		{    -0.0f,    0.0f,  559.2f, 0.0f  },
		{    -0.0f,  548.8f,  559.2f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },

		{    -0.0f,    0.0f,  559.2f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },
		{  -556.0f,    0.0f,  559.2f, 0.0f  },

		// Right wall
		{    -0.0f,    0.0f,    0.0f, 0.0f  },
		{    -0.0f,  548.8f,    0.0f, 0.0f  },
		{    -0.0f,  548.8f,  559.2f, 0.0f  },

		{    -0.0f,    0.0f,    0.0f, 0.0f  },
		{    -0.0f,  548.8f,  559.2f, 0.0f  },
		{    -0.0f,    0.0f,  559.2f, 0.0f  },

		// Left wall
		{  -556.0f,    0.0f,    0.0f, 0.0f  },
		{  -556.0f,    0.0f,  559.2f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },

		{  -556.0f,    0.0f,    0.0f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },
		{  -556.0f,  548.8f,    0.0f, 0.0f  },

		// Short block
		{  -130.0f,  165.0f + 9.0f,   65.0f, 0.0f  },
		{  - 82.0f,  165.0f + 9.0f,  225.0f, 0.0f  },
		{  -242.0f,  165.0f + 9.0f,  274.0f, 0.0f  },

		{  -130.0f,  165.0f + 9.0f,   65.0f, 0.0f  },
		{  -242.0f,  165.0f + 9.0f,  274.0f, 0.0f  },
		{  -290.0f,  165.0f + 9.0f,  114.0f, 0.0f  },

		{  -290.0f,    0.0f + 9.0f,  114.0f, 0.0f  },
		{  -290.0f,  165.0f + 9.0f,  114.0f, 0.0f  },
		{  -240.0f,  165.0f + 9.0f,  272.0f, 0.0f  },

		{  -290.0f,    0.0f + 9.0f,  114.0f, 0.0f  },
		{  -240.0f,  165.0f + 9.0f,  272.0f, 0.0f  },
		{  -240.0f,    0.0f + 9.0f,  272.0f, 0.0f  },

		{  -130.0f,    0.0f + 9.0f,   65.0f, 0.0f  },
		{  -130.0f,  165.0f + 9.0f,   65.0f, 0.0f  },
		{  -290.0f,  165.0f + 9.0f,  114.0f, 0.0f  },

		{  -130.0f,    0.0f + 9.0f,   65.0f, 0.0f  },
		{  -290.0f,  165.0f + 9.0f,  114.0f, 0.0f  },
		{  -290.0f,    0.0f + 9.0f,  114.0f, 0.0f  },

		{   -82.0f,    0.0f + 9.0f,  225.0f, 0.0f  },
		{   -82.0f,  165.0f + 9.0f,  225.0f, 0.0f  },
		{  -130.0f,  165.0f + 9.0f,   65.0f, 0.0f  },

		{  - 82.0f,    0.0f + 9.0f,  225.0f, 0.0f  },
		{  -130.0f,  165.0f + 9.0f,   65.0f, 0.0f  },
		{  -130.0f,    0.0f + 9.0f,   65.0f, 0.0f  },

		{  -240.0f,    0.0f + 9.0f,  272.0f, 0.0f  },
		{  -240.0f,  165.0f + 9.0f,  272.0f, 0.0f  },
		{   -82.0f,  165.0f + 9.0f,  225.0f, 0.0f  },

		{  -240.0f,    0.0f + 9.0f,  272.0f, 0.0f  },
		{   -82.0f,  165.0f + 9.0f,  225.0f, 0.0f  },
		{   -82.0f,    0.0f + 9.0f,  225.0f, 0.0f  },

		{  -130.0f,    0.0f + 9.0f,   65.0f, 0.0f  },
		{  -242.0f,    0.0f + 9.0f,  274.0f, 0.0f  },
		{  - 82.0f,    0.0f + 9.0f,  225.0f, 0.0f  },

		{  -130.0f,    0.0f + 9.0f,   65.0f, 0.0f  },
		{  -290.0f,    0.0f + 9.0f,  114.0f, 0.0f  },
		{  -242.0f,    0.0f + 9.0f,  274.0f, 0.0f  },

		// Tall block
		{  -423.0f,  330.0f,  247.0f, 0.0f  },
		{  -265.0f,  330.0f,  296.0f, 0.0f  },
		{  -314.0f,  330.0f,  455.0f, 0.0f  },

		{  -423.0f,  330.0f,  247.0f, 0.0f  },
		{  -314.0f,  330.0f,  455.0f, 0.0f  },
		{  -472.0f,  330.0f,  406.0f, 0.0f  },

		{  -423.0f,    0.0f,  247.0f, 0.0f  },
		{  -423.0f,  330.0f,  247.0f, 0.0f  },
		{  -472.0f,  330.0f,  406.0f, 0.0f  },

		{  -423.0f,    0.0f,  247.0f, 0.0f  },
		{  -472.0f,  330.0f,  406.0f, 0.0f  },
		{  -472.0f,    0.0f,  406.0f, 0.0f  },

		{  -472.0f,    0.0f,  406.0f, 0.0f  },
		{  -472.0f,  330.0f,  406.0f, 0.0f  },
		{  -314.0f,  330.0f,  456.0f, 0.0f  },

		{  -472.0f,    0.0f,  406.0f, 0.0f  },
		{  -314.0f,  330.0f,  456.0f, 0.0f  },
		{  -314.0f,    0.0f,  456.0f, 0.0f  },

		{  -314.0f,    0.0f,  456.0f, 0.0f  },
		{  -314.0f,  330.0f,  456.0f, 0.0f  },
		{  -265.0f,  330.0f,  296.0f, 0.0f  },

		{  -314.0f,    0.0f,  456.0f, 0.0f  },
		{  -265.0f,  330.0f,  296.0f, 0.0f  },
		{  -265.0f,    0.0f,  296.0f, 0.0f  },

		{  -265.0f,    0.0f,  296.0f, 0.0f  },
		{  -265.0f,  330.0f,  296.0f, 0.0f  },
		{  -423.0f,  330.0f,  247.0f, 0.0f  },

		{  -265.0f,    0.0f,  296.0f, 0.0f  },
		{  -423.0f,  330.0f,  247.0f, 0.0f  },
		{  -423.0f,    0.0f,  247.0f, 0.0f  },

		{  -423.0f,    0.0f,  247.0f, 0.0f  },
		{  -314.0f,    0.0f,  455.0f, 0.0f  },
		{  -265.0f,    0.0f,  296.0f, 0.0f  },

		{  -423.0f,    0.0f,  247.0f, 0.0f  },
		{  -472.0f,    0.0f,  406.0f, 0.0f  },
		{  -314.0f,    0.0f,  455.0f, 0.0f  },
	} };

	std::array<uint32_t, triangleCount> SBTIndices
	{ {
		0, 0,                                // Floor         -- white lambert
		0, 0,                                // Ceiling       -- white lambert
		0, 0,                                // Back wall     -- white lambert
		1, 1,                                // Right wall    -- green lambert
		2, 2,                                // Left wall     -- red lambert
		4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  // Short block   -- white lambert
		3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  // Tall block    -- white lambert
	} };

	static constexpr uint32_t lightCount{ 1 };
	struct Disk
	{
		const glm::vec3 pos{ -278.0f, 514.0f, 279.5f };
		// const glm::vec3 pos{ -98.0f, 124.0f, 379.5f };
		const float radius{ 80.0f };
		const float area{ 2.0f * glm::pi<float>() * radius };
		const glm::vec3 normal{ glm::normalize(glm::vec3{0.0f, -1.0f, 0.0f}) };
		// const glm::vec3 normal{ glm::normalize(glm::vec3{-0.3f, -0.3f, -1.0f}) };
		const glm::quat frame{ genDiskFrame() };
		const float scale{ 5.8f };
		//uint32_t pdfStructureIndex{};

		glm::quat genDiskFrame() const
		{
			glm::vec3 tang{ glm::normalize(glm::cross(glm::abs(glm::dot(glm::vec3{0.0f, 1.0f, 0.0f}, normal)) < 0.9999f ? glm::vec3{0.0f, 1.0f, 0.0f} : glm::vec3{1.0f, 0.0f, 0.0f}, normal)) };
			glm::vec3 bitang{ glm::cross(normal, tang) };
			return glm::quat_cast(glm::mat3(tang, bitang, normal));
		}

		OptixAabb getOptixAABB() const
		{
			OptixAabb t{};
			float xSpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.x * normal.x)) };
			t.maxX = xSpan == 0.0f ? std::nextafterf(pos.x, FLT_MAX) : pos.x + xSpan * radius;
			t.minX = xSpan == 0.0f ? std::nextafterf(pos.x, -FLT_MAX) : pos.x - xSpan * radius;
			float ySpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.y * normal.y)) };
			t.maxY = ySpan == 0.0f ? std::nextafterf(pos.y, FLT_MAX) : pos.y + ySpan * radius;
			t.minY = ySpan == 0.0f ? std::nextafterf(pos.y, -FLT_MAX) : pos.y - ySpan * radius;
			float zSpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.z * normal.z)) };
			t.maxZ = zSpan == 0.0f ? std::nextafterf(pos.z, FLT_MAX) : pos.z + zSpan * radius;
			t.minZ = zSpan == 0.0f ? std::nextafterf(pos.z, -FLT_MAX) : pos.z - zSpan * radius;
			return t;
		}
		OptixAabb getOptixAABB(const glm::vec3& translation) const
		{
			OptixAabb t{};
			float xSpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.x * normal.x)) };
			t.maxX = xSpan == 0.0f ? std::nextafterf(pos.x, FLT_MAX) : pos.x + xSpan * radius;
			t.minX = xSpan == 0.0f ? std::nextafterf(pos.x, -FLT_MAX) : pos.x - xSpan * radius;
			t.maxX += translation.x;
			t.minX += translation.x;
			float ySpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.y * normal.y)) };
			t.maxY = ySpan == 0.0f ? std::nextafterf(pos.y, FLT_MAX) : pos.y + ySpan * radius;
			t.minY = ySpan == 0.0f ? std::nextafterf(pos.y, -FLT_MAX) : pos.y - ySpan * radius;
			t.maxY += translation.y;
			t.minY += translation.y;
			float zSpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.z * normal.z)) };
			t.maxZ = zSpan == 0.0f ? std::nextafterf(pos.z, FLT_MAX) : pos.z + zSpan * radius;
			t.minZ = zSpan == 0.0f ? std::nextafterf(pos.z, -FLT_MAX) : pos.z - zSpan * radius;
			t.maxZ += translation.z;
			t.minZ += translation.z;
			return t;
		}
	} diskLight{};
	static constexpr uint32_t lightMaterialIndex{ geometryMaterialCount };

	struct MaterialDescriptor //We need 'MaterialDescrioptor' to find material data before rendering
	{
		uint32_t bxdfIndex{};
		SpectralData::SpectralDataType baseIOR{};
		SpectralData::SpectralDataType baseAC{};
		SpectralData::SpectralDataType baseEmission{};
		float roughness{};
	};
	const std::array<MaterialDescriptor, geometryMaterialCount + lightCount> materialDescriptors
	{ {
		MaterialDescriptor{.bxdfIndex = 0,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_AG_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_AG_AC,
			.baseEmission = SpectralData::SpectralDataType::DESC,
			.roughness = 1.0f},
		MaterialDescriptor{.bxdfIndex = 0,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_AU_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_AU_AC,
			.baseEmission = SpectralData::SpectralDataType::DESC,
			.roughness = 1.0f},
		MaterialDescriptor{.bxdfIndex = 0,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_CU_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_CU_AC,
			.baseEmission = SpectralData::SpectralDataType::DESC,
			.roughness = 1.0f},
		MaterialDescriptor{.bxdfIndex = 0,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC,
			.baseEmission = SpectralData::SpectralDataType::DESC,
			.roughness = 0.2f},
		MaterialDescriptor{.bxdfIndex = 1,
			.baseIOR = SpectralData::SpectralDataType::D_GLASS_BK7_IOR,
			.baseEmission = SpectralData::SpectralDataType::DESC,
			.roughness = 0.001f},
		MaterialDescriptor{.bxdfIndex = 0,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_TIO2_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_TIO2_AC,
			.baseEmission = SpectralData::SpectralDataType::ILLUM_D65,
			.roughness = 1.0f},
	} };
};
