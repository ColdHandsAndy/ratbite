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
		bool sheenPresent{ false };

		// Spectral Material
		SpectralData::SpectralDataType baseIOR{};
		SpectralData::SpectralDataType baseAC{};
		SpectralData::SpectralDataType baseEmission{};
		float roughness{};
		// Complex Material
		float ior{ 1.5f };
		int baseColorTextureIndex{ -1 };
		int bcTexCoordIndex{};
		int metalRoughnessTextureIndex{ -1 };
		int mrTexCoordIndex{};
		int normalTextureIndex{ -1 };
		int nmTexCoordIndex{};
		int transmissionTextureIndex{ -1 };
		int trTexCoordIndex{};
		int sheenColorTextureIndex{ -1 };
		int shcTexCoordIndex{};
		int sheenRoughTextureIndex{ -1 };
		int shrTexCoordIndex{};
		int emissiveTextureIndex{ -1 };
		int emTexCoordIndex{};

		bool bcFactorPresent{};
		glm::vec4 baseColorFactor{};
		enum class AlphaInterpretation
		{
			NONE,
			CUTOFF,
			BLEND
		};
		AlphaInterpretation alphaInterpretation{};
		float alphaCutoff{};
		bool metFactorPresent{};
		float metalnessFactor{};
		bool roughFactorPresent{};
		float roughnessFactor{};
		bool transmitFactorPresent{};
		float transmitFactor{};
		bool sheenColorFactorPresent{};
		float sheenColorFactor[3]{};
		bool sheenRoughnessFactorPresent{};
		float sheenRoughnessFactor{};
		bool emissiveFactorPresent{};
		float emissiveFactor[3]{};
	};

	struct Submesh
	{
		IndexType indexType{};
		uint32_t primitiveCount{};
		void* indices{};
		std::vector<uint32_t> discardedPrimitives{};
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
	struct EmissiveMeshSubset
	{
		int instanceIndex{};
		int submeshIndex{};
		float transformFluxCorrection{};
		struct TriangleData
		{
			glm::vec3 v0WS{}; // Vertices transformed to world space
			glm::vec3 v1WS{};
			glm::vec3 v2WS{};
			glm::vec3 v0{};
			glm::vec3 v1{};
			glm::vec3 v2{};
			glm::vec2 uv0{};
			glm::vec2 uv1{};
			glm::vec2 uv2{};
			uint32_t primIndex{};
			float flux{};
		};
		std::vector<TriangleData> triangles{};
	};
	struct Model
	{
		uint32_t id{};
		std::filesystem::path path{};
		std::string name{};

		std::vector<Mesh> meshes{};
		std::vector<Instance> instances{};
		std::vector<EmissiveMeshSubset> instancedEmissiveMeshSubsets{};

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
			meshes{ std::move(model.meshes) }, instances{ std::move(model.instances) }, instancedEmissiveMeshSubsets{ std::move(model.instancedEmissiveMeshSubsets) },
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
			instancedEmissiveMeshSubsets = std::move(model.instancedEmissiveMeshSubsets);

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
			instancedEmissiveMeshSubsets = model.instancedEmissiveMeshSubsets;

			transform = model.transform;

			triangleCount = model.triangleCount;

			imageData = model.imageData;
			textureData = model.textureData;

			materialDescriptors = model.materialDescriptors;

			return *this;
		}

		void setNewTransform(const glm::mat4x3& newTransform)
		{
			transform = newTransform;
			for (auto& subset : instancedEmissiveMeshSubsets)
			{
				glm::mat4 worldFromModel{ newTransform };
				worldFromModel[3][3] = 1.0f;
				glm::mat4 modelFromLocal{ instances[subset.instanceIndex].transform };
				modelFromLocal[3][3] = 1.0f;
				for (auto& tri : subset.triangles)
				{
					tri.v0WS = worldFromModel * modelFromLocal * glm::vec4{tri.v0, 1.0f};
					tri.v1WS = worldFromModel * modelFromLocal * glm::vec4{tri.v1, 1.0f};
					tri.v2WS = worldFromModel * modelFromLocal * glm::vec4{tri.v2, 1.0f};
				}
				subset.transformFluxCorrection = glm::abs(glm::determinant(glm::mat3{worldFromModel * modelFromLocal}));
			}
		}
		bool hasEmissiveData() const
		{
			return instancedEmissiveMeshSubsets.size() != 0;
		}

		int addTextureData(void* data, uint32_t width, uint32_t height, size_t byteSize)
		{
			imageData.emplace_back(data, width, height, byteSize);
			return imageData.size() - 1;
		}

		void clearVertexData()
		{
			for (auto& mesh : meshes)
				for (auto& submesh : mesh.submeshes)
				{
					submesh.vertices.clear();
					submesh.normals.clear();
					submesh.tangents.clear();
					for (auto& texCoordSet : submesh.texCoordsSets)
						texCoordSet.clear();
					free(submesh.indices);
					submesh.discardedPrimitives.clear();
				}
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
		glm::vec3 m_pos{ 0.0f };
		float m_radius{ 1.0f };
		// float m_area{ glm::pi<float>() * m_radius * m_radius };
		glm::vec3 m_normal{ glm::normalize(glm::vec3{0.0f, 0.0f, -1.0f}) };
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
			glm::vec3 tang{ glm::normalize(glm::cross(glm::abs(glm::dot(glm::vec3{0.0f, 0.0f, 1.0f}, m_normal)) < 0.9999f ? glm::vec3{0.0f, 0.0f, 1.0f} : glm::vec3{1.0f, 0.0f, 0.0f}, m_normal)) };
			glm::vec3 bitang{ glm::cross(m_normal, tang) };
			return glm::quat_cast(glm::mat3(tang, bitang, m_normal));
		}

	};
	class SphereLight
	{
	private:
		uint32_t m_id{};
		glm::vec3 m_pos{ 0.0f };
		float m_radius{ 1.0f };
		// float m_area{ 4.0f * glm::pi<float>() * m_radius * m_radius };
		glm::vec3 m_normal{ glm::normalize(glm::vec3{0.0f, 0.0f, 1.0f}) };
		glm::quat m_frame{ genDiskFrame() };
		float m_scale{ 1.0f };

		MaterialDescriptor m_matDesc{};
	public:
		SphereLight(const glm::vec3& position, float radius, float powerScale, const MaterialDescriptor& matDesc)
			: m_pos{ position }, m_radius{ radius }, m_normal{ 0.0f, 0.0f, 1.0f }, m_scale{ powerScale }, m_matDesc{ matDesc }, m_id{ ++m_idGen }
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
			glm::vec3 tang{ glm::normalize(glm::cross(glm::abs(glm::dot(glm::vec3{0.0f, 0.0f, 1.0f}, m_normal)) < 0.9999f ? glm::vec3{0.0f, 0.0f, 1.0f} : glm::vec3{1.0f, 0.0f, 0.0f}, m_normal)) };
			glm::vec3 bitang{ glm::cross(m_normal, tang) };
			return glm::quat_cast(glm::mat3(tang, bitang, m_normal));
		}
	};
	std::vector<SceneData::DiskLight> diskLights{};
	std::vector<SceneData::SphereLight> sphereLights{};

	std::string environmentMapPath{};

	int getLightCount() const { return diskLights.size() + sphereLights.size(); }

	SceneData()
	{
		
	}
};
