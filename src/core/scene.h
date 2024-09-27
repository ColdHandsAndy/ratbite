#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>

#include <optix_types.h>

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/debug_macros.h"
#include "../core/material.h"
#include "../core/spectral.h"
#include "../core/texture.h"
#include "../core/light.h"
#include "../core/util.h"

struct SceneData
{
	inline static uint32_t m_idGen{ 0 };

	enum class BxDF
	{
		PURE_CONDUCTOR,
		PURE_DIELECTRIC,
		COMPLEX_SURFACE,
		DESC
	};
	struct MaterialDescriptor
	{
		std::string name{};
		BxDF bxdf{};

		bool doubleSided{ false };

		// Spectral Material
		SpectralData::SpectralDataType baseIOR{};
		SpectralData::SpectralDataType baseAC{};
		SpectralData::SpectralDataType baseEmission{};
		float roughness{};
		// Triplet Material
		float ior{ 1.5f };
		int baseColorTextureIndex{ -1 };
		int bcTexCoordIndex{};
		int metalRoughnessTextureIndex{ -1 };
		int mrTexCoordIndex{};
		int normalTextureIndex{ -1 };
		int nmTexCoordIndex{};
		int transmissionTextureIndex{ -1 };
		int trTexCoordIndex{};

		bool bcFactorPresent{};
		glm::vec4 baseColorFactor{};
		bool alphaCutoffPresent{};
		float alphaCutoff{};
		bool metFactorPresent{};
		float metalnessFactor{};
		bool roughFactorPresent{};
		float roughnessFactor{};
		bool transmitFactorPresent{};
		float transmitFactor{};
	};

	// TODO: Need to free index buffer before deleting a submesh
	struct Submesh
	{
		IndexType indexType{};
		uint32_t primitiveCount{};
		void* indices{};
		uint32_t vertexCount{};
		std::vector<glm::vec4> vertices{};
		std::vector<glm::vec4> normals{};
		std::vector<glm::vec4> tangents{};
		std::vector<std::vector<glm::vec2>> texCoordsSets{};
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

			smesh.vertexCount = vertexCount;
			smesh.vertices.resize(vertexCount);

			return smesh;
		}
		void addNormals()
		{
			normals.resize(vertexCount);
		}
		void addTangents()
		{
			tangents.resize(vertexCount);
		}
		void addTexCoordsSet(int index)
		{
			if (index >= texCoordsSets.size())
			{
				texCoordsSets.resize(index + 1);
			}
			texCoordsSets[index] = std::vector<glm::vec2>(vertexCount);
		}
	};
	struct Mesh
	{
		std::vector<Submesh> submeshes{};
	};
	struct Instance
	{
		int meshIndex{};
		glm::mat4x3 transform{};
	};
	struct Model
	{
		uint32_t id{};
		std::filesystem::path path{};
		std::string name{};

		std::vector<Mesh> meshes{};
		std::vector<Instance> instances{};

		glm::mat4x3 transform{};

		size_t triangleCount{};

		struct ImageData
		{
			void* data{};
			uint32_t width{};
			uint32_t height{};
			uint32_t channelCount{};
			size_t byteSize{};
		};
		struct TextureData
		{
			int imageIndex{};
			bool sRGB{};
			TextureFilter filter{};
			TextureAddress addressX{};
			TextureAddress addressY{};
		};
		std::vector<ImageData> imageData{};
		std::vector<TextureData> textureData{};

		std::vector<MaterialDescriptor> materialDescriptors{};

		Model() = default;
		Model(Model&& model)
			: id{ model.id }, path{ std::move(model.path) }, name{ std::move(model.name) },
			meshes{ std::move(model.meshes) }, instances{ std::move(model.instances) },
			transform{ model.transform },
			triangleCount{ model.triangleCount },
			imageData{ std::move(model.imageData) }, textureData{ std::move(model.textureData) },
			materialDescriptors{ std::move(model.materialDescriptors) }
		{
		}
		Model& operator=(Model&& model) 
		{
			id = model.id;
			path = std::move(model.path);
			name = std::move(model.name);

			meshes = std::move(model.meshes);
			instances = std::move(model.instances);

			transform = model.transform;

			triangleCount = model.triangleCount;

			imageData = std::move(model.imageData);
			textureData = std::move(model.textureData);

			materialDescriptors = std::move(model.materialDescriptors);

			return *this;
		}
		Model& operator=(const Model& model)
		{
			id = model.id;
			path = model.path;
			name = model.name;

			meshes = model.meshes;
			instances = model.instances;

			transform = model.transform;

			triangleCount = model.triangleCount;

			imageData = model.imageData;
			textureData = model.textureData;

			materialDescriptors = model.materialDescriptors;

			return *this;
		}

		int addTextureData(void* data, uint32_t width, uint32_t height, size_t byteSize)
		{
			imageData.emplace_back(data, width, height, byteSize);
			return imageData.size() - 1;
		}
		void clearImageData()
		{
			for(auto& tD : imageData)
			{
				free(tD.data);
			}
			imageData.clear();
			textureData.clear();
		}
	};
	std::vector<Model> models{};
	int loadModel(const std::filesystem::path& path, const glm::mat4& transform = glm::identity<glm::mat4>());


	class DiskLight
	{
	private:
		uint32_t m_id{};
		glm::vec3 m_pos{ -278.0f, 514.0f, 279.5f };
		float m_radius{ 80.0f };
		// float m_area{ glm::pi<float>() * m_radius * m_radius };
		glm::vec3 m_normal{ glm::normalize(glm::vec3{0.0f, -1.0f, 0.0f}) };
		glm::quat m_frame{ genDiskFrame() };
		float m_scale{ 1.0f };

		MaterialDescriptor m_matDesc{};

	public:
		DiskLight(const glm::vec3& position, float radius, const glm::vec3& normal, float powerScale, const MaterialDescriptor& matDesc)
			: m_pos{ position }, m_radius{ radius }, m_normal{ normal }, m_scale{ powerScale }, m_matDesc{ matDesc }, m_id{ ++m_idGen }
		{}
		DiskLight& operator=(const DiskLight& light)
		{
			m_id = light.m_id;
			m_pos = light.m_pos;
			m_radius = light.m_radius;
			m_normal = light.m_normal;
			m_frame = light.m_frame;
			m_scale = light.m_scale;
			m_matDesc = light.m_matDesc;

			return *this;
		}

		const glm::vec3& getPosition() const { return m_pos; }
		const glm::vec3& getNormal() const { return m_normal; }
		const glm::quat& getFrame() const { return m_frame; }
		float getRadius() const { return m_radius; }
		// float getArea() const { return m_area; }
		float getPowerScale() const { return m_scale; }
		uint32_t getID() const { return m_id; }
		const MaterialDescriptor& getMaterialDescriptor() { return m_matDesc; }

		void setPosition(const glm::vec3& position) { m_pos = position; }
		void setNormal(const glm::vec3& normal) { m_normal = normal; m_frame = genDiskFrame(); }
		void setRadius(float radius) { m_radius = radius; }
		void setPowerScale(float scale) { m_scale = scale; }
		void setEmissionSpectrum(SpectralData::SpectralDataType type) { m_matDesc.baseEmission = type; }

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
		uint32_t m_id{};
		glm::vec3 m_pos{ -278.0f, 514.0f, 279.5f };
		float m_radius{ 50.0f };
		// float m_area{ 4.0f * glm::pi<float>() * m_radius * m_radius };
		glm::vec3 m_normal{ glm::normalize(glm::vec3{0.0f, 1.0f, 0.0f}) };
		glm::quat m_frame{ genDiskFrame() };
		float m_scale{ 1.0f };

		MaterialDescriptor m_matDesc{};
	public:
		SphereLight(const glm::vec3& position, float radius, float powerScale, const MaterialDescriptor& matDesc)
			: m_pos{ position }, m_radius{ radius }, m_normal{ 0.0f, 1.0f, 0.0f }, m_scale{ powerScale }, m_matDesc{ matDesc }, m_id{ ++m_idGen }
		{}
		SphereLight& operator=(const SphereLight& light)
		{
			m_id = light.m_id;
			m_pos = light.m_pos;
			m_radius = light.m_radius;
			m_normal = light.m_normal;
			m_frame = light.m_frame;
			m_scale = light.m_scale;
			m_matDesc = light.m_matDesc;

			return *this;
		}

		const glm::vec3& getPosition() const { return m_pos; }
		const glm::vec3& getNormal() const { return m_normal; }
		const glm::quat& getFrame() const { return m_frame; }
		float getRadius() const { return m_radius; }
		// float getArea() const { return m_area; }
		float getPowerScale() const { return m_scale; }
		uint32_t getID() const { return m_id; }
		const MaterialDescriptor& getMaterialDescriptor() { return m_matDesc; }

		void setPosition(const glm::vec3& position) { m_pos = position; }
		void setRadius(float radius) { m_radius = radius; }
		void setPowerScale(float scale) { m_scale = scale; }
		void setEmissionSpectrum(SpectralData::SpectralDataType type) { m_matDesc.baseEmission = type; }

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

	int getLightCount() const { return diskLights.size() + sphereLights.size(); }

	SceneData()
	{
		// loadModel("A:/Models/gltf/mc_village.glb");
		// SphereLight sphere{ {-78.0f, 214.0f, 339.5f}, 90.0f, 0.1f,
		// 	MaterialDescriptor{.name = "Light 0",
		// 				.bxdf = BxDF::PURE_CONDUCTOR,
		// 				.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR,
		// 				.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC,
		// 				.baseEmission = SpectralData::SpectralDataType::ILLUM_D65,
		// 				.roughness = 1.0f} };
		// sphereLights.push_back(sphere);
		//
		// DiskLight disk{ {-78.0f, 214.0f, 339.5f}, 90.0f, {0.0f, -1.0f, 0.0f}, 0.1f,
		// 	MaterialDescriptor{.name = "Light 1",
		// 				.bxdf = BxDF::PURE_CONDUCTOR,
		// 				.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR,
		// 				.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC,
		// 				.baseEmission = SpectralData::SpectralDataType::ILLUM_D65,
		// 				.roughness = 1.0f} };
		// diskLights.push_back(disk);
	}
};
