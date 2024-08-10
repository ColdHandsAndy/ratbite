#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <stack>
#include <string>
#include <filesystem>

#include <optix_types.h>

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/debug_macros.h"
#include "../core/material.h"
#include "../core/spectral.h"
#include "../core/light.h"
#include "../core/util.h"

struct SceneData
{
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
	std::vector<MaterialDescriptor> materialDescriptors{};

	bool materialDescriptorChangesMade{ false };
	bool newMaterialDescriptorAdded{ false };
	int changedMaterialDescriptorIndex{};
	MaterialDescriptor tempDescriptor{};


	// TODO: Need to free index buffer before deleting a submesh
	struct Submesh
	{
		IndexType indexType{};
		uint32_t primitiveCount{};
		void* indices{};
		std::vector<glm::vec4> vertices{};
		std::vector<glm::vec4> normals{};
		// std::vector<glm::quat> frame{};
		int materialIndex{};
		static Submesh createSubmesh(size_t indexCount, IndexType indexType, size_t vertexCount, int materialIndex)
		{
			Submesh smesh{};
			smesh.primitiveCount = indexCount / 3;
			if (smesh.primitiveCount == 0)
				return smesh;

			smesh.materialIndex = materialIndex;
			smesh.indexType = indexType;

			size_t tSize{};
			switch (indexType)
			{
				case IndexType::UINT_16:
					tSize = sizeof(uint16_t);
					break;
				case IndexType::UINT_32:
					tSize = sizeof(uint32_t);
					break;
				default:
					R_ERR_LOG("Invalid type passed");
					break;
			}
			smesh.indices = malloc(indexCount * tSize);

			smesh.vertices.resize(vertexCount);
			smesh.normals.resize(vertexCount);

			return smesh;
		}
	};
	struct Mesh
	{
		std::vector<Submesh> submeshes{};
	};
	struct Instance
	{
		int meshIndex{};
		glm::mat3x4 transform{};
	};
	struct Model
	{
		std::filesystem::path path{};
		std::string name{};

		std::vector<Mesh> meshes{};
		std::vector<Instance> instances{};
	};
	std::vector<Model> models{};
	int getGeometryMaterialCount() const
	{
		int count{ 0 };
		for(auto& model : models)
			for(auto& mesh : model.meshes)
				count += mesh.submeshes.size();
		return count;
	}
	void loadModel(const std::filesystem::path& path, const MaterialDescriptor* assignedMaterial = nullptr);


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

	bool lightDataChangesMade{ false };
	LightType changedLightType{};
	// bool lightDataAABBChanged{ false };
	// int changedLightIndex{};


	bool changesMade() const { return materialDescriptorChangesMade || lightDataChangesMade; }

	SceneData()
	{
		MaterialDescriptor mat{ MaterialDescriptor{
			.name = "Model",
			.bxdf = BxDF::CONDUCTOR,
			.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR,
			.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC,
			.baseEmission = SpectralData::SpectralDataType::NONE,
			.roughness = 1.0f} };
		loadModel("A:/Models/gltf/deccer cubes/deccer_cubes.gltf", &mat);
		// loadModel("A:/Models/gltf/cornell_scene/cornell_scene.gltf", &mat);
		// loadModel("A:/Models/gltf/prism/prism.gltf", &mat);


		int matIndex{};
		// matIndex = static_cast<int>(materialDescriptors.size());
		// materialDescriptors.push_back(
		// 		MaterialDescriptor{.name = "Light 0",
		// 		.bxdf = BxDF::CONDUCTOR,
		// 		.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR,
		// 		.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC,
		// 		.baseEmission = SpectralData::SpectralDataType::ILLUM_D65,
		// 		.roughness = 1.0f});
		// DiskLight disk{ {-278.0f, 514.0f, 279.5f},
		// 	80.0f,
		// 	glm::normalize(glm::vec3{0.0f, -1.0f, 0.0f}),
		// 	0.0f,
		// 	matIndex };
		// diskLights.push_back(disk);
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
		matIndex = static_cast<int>(materialDescriptors.size());
		materialDescriptors.push_back(
				MaterialDescriptor{.name = "Light 1",
				.bxdf = BxDF::CONDUCTOR,
				.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR,
				.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC,
				.baseEmission = SpectralData::SpectralDataType::ILLUM_F5,
				.roughness = 1.0f});
		SphereLight sphere{ {-78.0f, 214.0f, 339.5f}, 90.0f, 0.1f, matIndex };
		sphereLights.push_back(sphere);
		// matIndex = static_cast<int>(materialDescriptors.size());
		// materialDescriptors.push_back(
		// 		MaterialDescriptor{.name = "Light 2",
		// 		.bxdf = BxDF::CONDUCTOR,
		// 		.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR,
		// 		.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC,
		// 		.baseEmission = SpectralData::SpectralDataType::ILLUM_F9,
		// 		.roughness = 1.0f});
		// SphereLight sphere1{ {-378.0f, 454.0f, 279.5f}, 120.0f, 0.1f, matIndex };
		// sphereLights.push_back(sphere1);

		lightCount = diskLights.size() + sphereLights.size();
	}
};
