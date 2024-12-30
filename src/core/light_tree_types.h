#pragma once

#include <cuda/std/cstdint>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/util.h"
#include "../core/math_util.h"

enum class LightType
{
	TRIANGLE,
	DISK,
	SPHERE,

	SKY,
	NONE,

	ALL,
};
// LightType enums we can have multiple of are also used as indices into arrays therefore static asserts
constexpr uint32_t KTriangleLightsArrayIndex{ static_cast<uint32_t>(LightType::TRIANGLE) };
constexpr uint32_t KDiskLightsArrayIndex{ static_cast<uint32_t>(LightType::DISK) };
constexpr uint32_t KSphereLightsArrayIndex{ static_cast<uint32_t>(LightType::SPHERE) };
constexpr uint32_t KLightTypeCount{ 3 };
static_assert(static_cast<int>(LightType::TRIANGLE) == 0);
static_assert(static_cast<int>(LightType::DISK) == 1);
static_assert(static_cast<int>(LightType::SPHERE) == 2);

struct DiskLightData
{
	glm::vec3 position{};
	float powerScale{};
	glm::quat frame{};
	float radius{};
	uint32_t materialIndex{};
};
struct SphereLightData
{
	glm::vec3 position{};
	float powerScale{};
	glm::quat frame{};
	float radius{};
	uint32_t materialIndex{};
};
struct EmissiveTriangleLightData
{
	float vertices[9]{};
	float uvs[6]{};
	uint32_t materialIndex{};
	uint32_t primitiveDataIndex{};
};

namespace LightTree
{
	constexpr inline uint32_t KLightCountBits{ 4 };
	constexpr inline uint32_t KLightOffsetBits{ 31 - KLightCountBits };
	constexpr inline uint32_t KMaxLeafLightCount{ (1 << KLightCountBits) - 1 };
	constexpr inline uint32_t KMaxLeafLightOffset{ (1 << KLightOffsetBits) - 1 };
	constexpr inline uint32_t KMaxDepth{ 64 };
	constexpr inline float KEnvironmentMapImportance{ 0.3f };

	struct NodeAttributes
	{
		float spatialMean[3]{};
		float spatialVariance{};
		float flux{};
		float averageDirection[3]{};
		float sharpness{};

		static NodeAttributes add(const NodeAttributes& nodeA, const NodeAttributes& nodeB)
		{
			namespace SG = SphericalGaussian;

			NodeAttributes res{};

			const float fluxWeightA{ nodeA.flux / (nodeA.flux + nodeB.flux) };
			const float fluxWeightB{ nodeB.flux / (nodeA.flux + nodeB.flux) };

			const float axisLengthA{ SG::VMFSharpnessToAxisLength(nodeA.sharpness) };
			const float axisLengthB{ SG::VMFSharpnessToAxisLength(nodeB.sharpness) };
			float averageDirection[3]{
				fluxWeightA * axisLengthA * nodeA.averageDirection[0] + fluxWeightB * axisLengthB * nodeB.averageDirection[0],
				fluxWeightA * axisLengthA * nodeA.averageDirection[1] + fluxWeightB * axisLengthB * nodeB.averageDirection[1],
				fluxWeightA * axisLengthA * nodeA.averageDirection[2] + fluxWeightB * axisLengthB * nodeB.averageDirection[2], };
			float averageDirectionLength{ sqrt(averageDirection[0] * averageDirection[0] + averageDirection[1] * averageDirection[1] + averageDirection[2] * averageDirection[2]) };
			averageDirection[0] = averageDirectionLength > 0.0001f ? averageDirection[0] / averageDirectionLength : 0.0f;
			averageDirection[1] = averageDirectionLength > 0.0001f ? averageDirection[1] / averageDirectionLength : 0.0f;
			averageDirection[2] = averageDirectionLength > 0.0001f ? averageDirection[2] / averageDirectionLength : 0.0f;
			float spatialMeanDif[]{
				nodeB.spatialMean[0] - nodeA.spatialMean[0],
				nodeB.spatialMean[1] - nodeA.spatialMean[1],
				nodeB.spatialMean[2] - nodeA.spatialMean[2], };
			float spatialMeanDifLengthSq{ spatialMeanDif[0] * spatialMeanDif[0] + spatialMeanDif[1] * spatialMeanDif[1] + spatialMeanDif[2] * spatialMeanDif[2] };

			res.spatialMean[0] = fluxWeightA * nodeA.spatialMean[0] + fluxWeightB * nodeB.spatialMean[0];
			res.spatialMean[1] = fluxWeightA * nodeA.spatialMean[1] + fluxWeightB * nodeB.spatialMean[1];
			res.spatialMean[2] = fluxWeightA * nodeA.spatialMean[2] + fluxWeightB * nodeB.spatialMean[2];
			res.spatialVariance = fluxWeightA * nodeA.spatialVariance + fluxWeightB * nodeB.spatialVariance + fluxWeightA * fluxWeightB * spatialMeanDifLengthSq;
			res.averageDirection[0] = averageDirection[0];
			res.averageDirection[1] = averageDirection[1];
			res.averageDirection[2] = averageDirection[2];
			res.flux = nodeA.flux + nodeB.flux;
			res.sharpness = SG::VMFAxisLengthToSharpness(averageDirectionLength);
			return res;
		}
	};
	struct UnpackedNode
	{
		NodeAttributes attributes;
		union Core
		{
			struct
			{
				uint32_t rightChildIndex;
			} branch;
			struct
			{
				uint32_t lightCount;
				uint32_t lightOffset;
			} leaf;
		} core;
	};

	struct PackedNode
	{
		float spatialMean[3]{};
		float spatialVariance{};
		uint32_t packedAverageDirection{};
		float sharpness{};
		float flux{};
		uint32_t coreData{};

		PackedNode() {}
		PackedNode(const NodeAttributes& attributes,
				uint32_t rightChildIndex)
		{
			packAttributes(attributes);
			packCoreData(rightChildIndex);
		}
		PackedNode(const NodeAttributes& attributes,
				uint32_t lightOffset, uint32_t lightCount)
		{
			packAttributes(attributes);
			packCoreData(lightOffset, lightCount);
		}
		// PackedNode(const float* nodeSpatialMean, float nodeSpatialVariance,
		// 		const float* averageDirection,
		// 		float nodeSharpness, float nodeFlux,
		// 		uint32_t rightChildIndex)
		// 	: spatialMean{ nodeSpatialMean[0], nodeSpatialMean[1], nodeSpatialMean[2] }, spatialVariance{ nodeSpatialVariance },
		// 	packedAverageDirection{ Octohedral::encodeU32(averageDirection[0], averageDirection[1], averageDirection[2]) },
		// 	sharpness{ nodeSharpness }, flux{ nodeFlux },
		// 	coreData{ ((1u << 31u) - 1u) & rightChildIndex }
		// {}
		// PackedNode(const float* nodeSpatialMean, float nodeSpatialVariance,
		// 		const float* averageDirection,
		// 		float nodeSharpness, float nodeFlux,
		// 		uint32_t lightOffset, uint32_t lightCount)
		// 	: spatialMean{ nodeSpatialMean[0], nodeSpatialMean[1], nodeSpatialMean[2] }, spatialVariance{ nodeSpatialVariance },
		// 	packedAverageDirection{ Octohedral::encodeU32(averageDirection[0], averageDirection[1], averageDirection[2]) },
		// 	sharpness{ nodeSharpness }, flux{ nodeFlux },
		// 	coreData{ (1u << 31u) | ((lightCount & ((1u << KLightCountBits) - 1u)) << KLightOffsetBits) | (lightOffset & ((1u << KLightOffsetBits) - 1u)) }
		// {}

		bool isLeaf() const { return (coreData >> 31u) != 0u; }

		void packAttributes(const NodeAttributes& attributes)
		{
			spatialMean[0] = attributes.spatialMean[0];
			spatialMean[1] = attributes.spatialMean[1];
			spatialMean[2] = attributes.spatialMean[2];
			spatialVariance = attributes.spatialVariance;
			packedAverageDirection = Octohedral::encodeU32(attributes.averageDirection[0], attributes.averageDirection[1], attributes.averageDirection[2]);
			sharpness = attributes.sharpness;
			flux = attributes.flux;
		}
		void packCoreData(uint32_t rightChildIndex)
		{
			coreData = ((1u << 31u) - 1u) & rightChildIndex;
		}
		void packCoreData(uint32_t lightOffset, uint32_t lightCount)
		{
			coreData = (1u << 31u) | ((lightCount & ((1u << KLightCountBits) - 1u)) << KLightOffsetBits) | (lightOffset & ((1u << KLightOffsetBits) - 1u));
		}
	};
	static_assert(sizeof(PackedNode) == 32);

	CU_HOSTDEVICE CU_INLINE UnpackedNode unpackNode(const PackedNode& packedNode, bool& isLeaf)
	{
		UnpackedNode res{};
		res.attributes.spatialMean[0] = packedNode.spatialMean[0];
		res.attributes.spatialMean[1] = packedNode.spatialMean[1];
		res.attributes.spatialMean[2] = packedNode.spatialMean[2];
		res.attributes.spatialVariance = packedNode.spatialVariance;
		res.attributes.flux = packedNode.flux;
		res.attributes.sharpness = packedNode.sharpness;
		Octohedral::decode((packedNode.packedAverageDirection & 0xFFFF) / 65535.0f, (packedNode.packedAverageDirection >> 16) / 65535.0f,
				res.attributes.averageDirection[0], res.attributes.averageDirection[1], res.attributes.averageDirection[2]);

		isLeaf = (packedNode.coreData >> 31) != 0;
		if (isLeaf)
		{
			res.core.leaf.lightCount = (packedNode.coreData >> KLightOffsetBits) & ((1u << KLightCountBits) - 1u);
			res.core.leaf.lightOffset = packedNode.coreData & ((1u << KLightOffsetBits) - 1u);
		}
		else
		{
			res.core.branch.rightChildIndex = packedNode.coreData;
		}

		return res;
	}

	struct LightPointer
	{
		uint32_t lptr{};

		constexpr static inline uint32_t KLightIndexBits{ 29u };
		constexpr static inline uint32_t KLightTypeBits{ 32u - KLightIndexBits };
		static_assert(static_cast<uint32_t>(LightType::NONE) < (1u << KLightTypeBits));
		CU_HOSTDEVICE CU_INLINE void pack(LightType type, uint32_t index)
		{
			lptr = (static_cast<uint32_t>(type) << KLightIndexBits) | (index & ((1u << KLightIndexBits) - 1u));
		}
		CU_HOSTDEVICE CU_INLINE void unpack(LightType& type, uint32_t& index) const
		{
			type = static_cast<LightType>(lptr >> KLightIndexBits);
			index = lptr & ((1u << KLightIndexBits) - 1u);
		}
	};
}
