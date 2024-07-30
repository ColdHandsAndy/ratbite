#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <stack>
#include <string>

#include <optix_types.h>

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/material.h"
#include "../core/spectral.h"
#include "../core/light.h"

struct SceneData
{
	constexpr static uint32_t triangleCount{ 34 };
	constexpr static uint32_t geometryMaterialCount{ 5 };

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

	class DiskLight
	{
	private:
		glm::vec3 m_pos{ -278.0f, 514.0f, 279.5f };
		float m_radius{ 80.0f };
		// float m_area{ glm::pi<float>() * m_radius * m_radius };
		glm::vec3 m_normal{ glm::normalize(glm::vec3{0.0f, -1.0f, 0.0f}) };
		glm::quat m_frame{ genDiskFrame() };
		float m_scale{ 1.0f };
		int m_materialDescIndex{ 0 };

	public:
		DiskLight(const glm::vec3& position, float radius, const glm::vec3& normal, float powerScale, int matDescIndex)
			: m_pos{ position }, m_radius{ radius }, m_normal{ normal }, m_scale{ powerScale }, m_materialDescIndex{ matDescIndex }
		{}
		const glm::vec3& getPosition() const { return m_pos; }
		const glm::vec3& getNormal() const { return m_normal; }
		const glm::quat& getFrame() const { return m_frame; }
		float getRadius() const { return m_radius; }
		// float getArea() const { return m_area; }
		float getPowerScale() const { return m_scale; }
		int getMaterialIndex() const { return m_materialDescIndex; }

		void setPosition(const glm::vec3& position) { m_pos = position; }
		void setNormal(const glm::vec3& normal) { m_normal = normal; m_frame = genDiskFrame(); }
		void setRadius(float radius) { m_radius = radius; }
		void setPowerScale(float scale) { m_scale = scale; }
		void setMaterialDescIndex(int index) { m_materialDescIndex = index; }

		OptixAabb getOptixAABB() const
		{
			OptixAabb t{};
			float xSpan{ glm::max(0.0f, glm::sqrt(1.0f - m_normal.x * m_normal.x)) };
			t.maxX = xSpan == 0.0f ? std::nextafterf(m_pos.x, FLT_MAX) : m_pos.x + xSpan * m_radius;
			t.minX = xSpan == 0.0f ? std::nextafterf(m_pos.x, -FLT_MAX) : m_pos.x - xSpan * m_radius;
			float ySpan{ glm::max(0.0f, glm::sqrt(1.0f - m_normal.y * m_normal.y)) };
			t.maxY = ySpan == 0.0f ? std::nextafterf(m_pos.y, FLT_MAX) : m_pos.y + ySpan * m_radius;
			t.minY = ySpan == 0.0f ? std::nextafterf(m_pos.y, -FLT_MAX) : m_pos.y - ySpan * m_radius;
			float zSpan{ glm::max(0.0f, glm::sqrt(1.0f - m_normal.z * m_normal.z)) };
			t.maxZ = zSpan == 0.0f ? std::nextafterf(m_pos.z, FLT_MAX) : m_pos.z + zSpan * m_radius;
			t.minZ = zSpan == 0.0f ? std::nextafterf(m_pos.z, -FLT_MAX) : m_pos.z - zSpan * m_radius;
			return t;
		}
		OptixAabb getOptixAABB(const glm::vec3& translation) const
		{
			OptixAabb t{};
			float xSpan{ glm::max(0.0f, glm::sqrt(1.0f - m_normal.x * m_normal.x)) };
			t.maxX = xSpan == 0.0f ? std::nextafterf(m_pos.x, FLT_MAX) : m_pos.x + xSpan * m_radius;
			t.minX = xSpan == 0.0f ? std::nextafterf(m_pos.x, -FLT_MAX) : m_pos.x - xSpan * m_radius;
			t.maxX += translation.x;
			t.minX += translation.x;
			float ySpan{ glm::max(0.0f, glm::sqrt(1.0f - m_normal.y * m_normal.y)) };
			t.maxY = ySpan == 0.0f ? std::nextafterf(m_pos.y, FLT_MAX) : m_pos.y + ySpan * m_radius;
			t.minY = ySpan == 0.0f ? std::nextafterf(m_pos.y, -FLT_MAX) : m_pos.y - ySpan * m_radius;
			t.maxY += translation.y;
			t.minY += translation.y;
			float zSpan{ glm::max(0.0f, glm::sqrt(1.0f - m_normal.z * m_normal.z)) };
			t.maxZ = zSpan == 0.0f ? std::nextafterf(m_pos.z, FLT_MAX) : m_pos.z + zSpan * m_radius;
			t.minZ = zSpan == 0.0f ? std::nextafterf(m_pos.z, -FLT_MAX) : m_pos.z - zSpan * m_radius;
			t.maxZ += translation.z;
			t.minZ += translation.z;
			return t;
		}

	private:
		glm::quat genDiskFrame() const
		{
			glm::vec3 tang{ glm::normalize(glm::cross(glm::abs(glm::dot(glm::vec3{0.0f, 1.0f, 0.0f}, m_normal)) < 0.9999f ? glm::vec3{0.0f, 1.0f, 0.0f} : glm::vec3{1.0f, 0.0f, 0.0f}, m_normal)) };
			glm::vec3 bitang{ glm::cross(m_normal, tang) };
			return glm::quat_cast(glm::mat3(tang, bitang, m_normal));
		}

	};
	class SphereLight
	{
	private:
		glm::vec3 m_pos{ -278.0f, 514.0f, 279.5f };
		float m_radius{ 50.0f };
		// float m_area{ 4.0f * glm::pi<float>() * m_radius * m_radius };
		glm::vec3 m_normal{ glm::normalize(glm::vec3{0.0f, 1.0f, 0.0f}) };
		glm::quat m_frame{ genDiskFrame() };
		float m_scale{ 1.0f };
		int m_materialDescIndex{ 0 };
	public:
		SphereLight(const glm::vec3& position, float radius, float powerScale, int matDescIndex)
			: m_pos{ position }, m_radius{ radius }, m_normal{ 0.0f, 1.0f, 0.0f }, m_scale{ powerScale }, m_materialDescIndex{ matDescIndex }
		{}

		const glm::vec3& getPosition() const { return m_pos; }
		const glm::vec3& getNormal() const { return m_normal; }
		const glm::quat& getFrame() const { return m_frame; }
		float getRadius() const { return m_radius; }
		// float getArea() const { return m_area; }
		float getPowerScale() const { return m_scale; }
		int getMaterialIndex() const { return m_materialDescIndex; }

		void setPosition(const glm::vec3& position) { m_pos = position; }
		void setRadius(float radius) { m_radius = radius; }
		void setPowerScale(float scale) { m_scale = scale; }
		void setMaterialDescIndex(int index) { m_materialDescIndex = index; }

		OptixAabb getOptixAABB() const
		{
			OptixAabb t{};
			float xSpan{ 1.0f };
			t.maxX = xSpan == 0.0f ? std::nextafterf(m_pos.x, FLT_MAX) : m_pos.x + xSpan * m_radius;
			t.minX = xSpan == 0.0f ? std::nextafterf(m_pos.x, -FLT_MAX) : m_pos.x - xSpan * m_radius;
			float ySpan{ 1.0f };
			t.maxY = ySpan == 0.0f ? std::nextafterf(m_pos.y, FLT_MAX) : m_pos.y + ySpan * m_radius;
			t.minY = ySpan == 0.0f ? std::nextafterf(m_pos.y, -FLT_MAX) : m_pos.y - ySpan * m_radius;
			float zSpan{ 1.0f };
			t.maxZ = zSpan == 0.0f ? std::nextafterf(m_pos.z, FLT_MAX) : m_pos.z + zSpan * m_radius;
			t.minZ = zSpan == 0.0f ? std::nextafterf(m_pos.z, -FLT_MAX) : m_pos.z - zSpan * m_radius;
			return t;
		}
		OptixAabb getOptixAABB(const glm::vec3& translation) const
		{
			OptixAabb t{};
			float xSpan{ 1.0f };
			t.maxX = xSpan == 0.0f ? std::nextafterf(m_pos.x, FLT_MAX) : m_pos.x + xSpan * m_radius;
			t.minX = xSpan == 0.0f ? std::nextafterf(m_pos.x, -FLT_MAX) : m_pos.x - xSpan * m_radius;
			t.maxX += translation.x;
			t.minX += translation.x;
			float ySpan{ 1.0f };
			t.maxY = ySpan == 0.0f ? std::nextafterf(m_pos.y, FLT_MAX) : m_pos.y + ySpan * m_radius;
			t.minY = ySpan == 0.0f ? std::nextafterf(m_pos.y, -FLT_MAX) : m_pos.y - ySpan * m_radius;
			t.maxY += translation.y;
			t.minY += translation.y;
			float zSpan{ 1.0f };
			t.maxZ = zSpan == 0.0f ? std::nextafterf(m_pos.z, FLT_MAX) : m_pos.z + zSpan * m_radius;
			t.minZ = zSpan == 0.0f ? std::nextafterf(m_pos.z, -FLT_MAX) : m_pos.z - zSpan * m_radius;
			t.maxZ += translation.z;
			t.minZ += translation.z;
			return t;
		}
	private:
		glm::quat genDiskFrame() const
		{
			glm::vec3 tang{ glm::normalize(glm::cross(glm::abs(glm::dot(glm::vec3{0.0f, 1.0f, 0.0f}, m_normal)) < 0.9999f ? glm::vec3{0.0f, 1.0f, 0.0f} : glm::vec3{1.0f, 0.0f, 0.0f}, m_normal)) };
			glm::vec3 bitang{ glm::cross(m_normal, tang) };
			return glm::quat_cast(glm::mat3(tang, bitang, m_normal));
		}
	};
	std::vector<SceneData::DiskLight> diskLights{};
	std::vector<SceneData::SphereLight> sphereLights{};
	uint32_t lightCount{};

	enum class BxDF
	{
		CONDUCTOR,
		DIELECTIRIC,
		DESC
	};
	struct MaterialDescriptor //We need 'MaterialDescrioptor' to find material data before rendering
	{
		std::string name{};
		BxDF bxdf{};
		SpectralData::SpectralDataType baseIOR{};
		SpectralData::SpectralDataType baseAC{};
		SpectralData::SpectralDataType baseEmission{};
		float roughness{};
	};
	std::vector<MaterialDescriptor> materialDescriptors
	{ {
		MaterialDescriptor{.name = "Floor, Back wall and Ceiling",
			.bxdf = BxDF::CONDUCTOR,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_AG_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_AG_AC,
			.roughness = 1.0f},
		MaterialDescriptor{.name = "Right wall",
			.bxdf = BxDF::CONDUCTOR,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_AU_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_AU_AC,
			.roughness = 1.0f},
		MaterialDescriptor{.name = "Left wall",
			.bxdf = BxDF::CONDUCTOR,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_CU_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_CU_AC,
			.roughness = 1.0f},
		MaterialDescriptor{.name = "Tall block",
			.bxdf = BxDF::CONDUCTOR,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC,
			.roughness = 0.1f},
		MaterialDescriptor{.name = "Short block",
			.bxdf = BxDF::DIELECTIRIC,
			.baseIOR = SpectralData::SpectralDataType::D_GLASS_BK7_IOR,
			.roughness = 0.01f},
	} };


	bool materialDescriptorChangesMade{ false };
	bool newMaterialDescriptorAdded{ false };
	int changedMaterialDescriptorIndex{};
	MaterialDescriptor tempDescriptor{};

	bool lightDataChangesMade{ false };
	LightType changedLightType{};
	// bool lightDataAABBChanged{ false };
	// int changedLightIndex{};

	bool changesMade() const { return materialDescriptorChangesMade || lightDataChangesMade; }

	SceneData()
	{
		int matIndex{ static_cast<int>(materialDescriptors.size()) };
		materialDescriptors.push_back(
				MaterialDescriptor{.name = "Light",
				.bxdf = BxDF::CONDUCTOR,
				.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR,
				.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC,
				.baseEmission = SpectralData::SpectralDataType::ILLUM_F5,
				.roughness = 1.0f});
		DiskLight disk{ {-278.0f, 514.0f, 279.5f},
			80.0f,
			glm::normalize(glm::vec3{0.0f, -1.0f, 0.0f}),
			0.4f,
			matIndex };
		diskLights.push_back(disk);
		// DiskLight disk2{ {-278.0f, 314.0f, 279.5f},
		// 	80.0f,
		// 	glm::normalize(glm::vec3{0.0f, -1.0f, 0.0f}),
		// 	0.1f,
		// 	matIndex };
		// diskLights.push_back(disk2);
		// DiskLight disk3{ {-278.0f, 114.0f, 279.5f},
		// 	80.0f,
		// 	glm::normalize(glm::vec3{0.0f, -1.0f, 0.0f}),
		// 	0.2f,
		// 	matIndex };
		// diskLights.push_back(disk3);
		SphereLight sphere{ {-78.0f, 214.0f, 339.5f}, 90.0f, 0.1f, matIndex };
		sphereLights.push_back(sphere);
		// SphereLight sphere1{ {-378.0f, 454.0f, 279.5f}, 120.0f, 0.1f, matIndex };
		// sphereLights.push_back(sphere1);

		lightCount = diskLights.size() + sphereLights.size();
	}
};
