#pragma once

#include <cuda/std/cstdint>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/util.h"
#include "../core/math_util.h"

enum class LightType
{
	NONE,
	TRIANGLE,
	DISK,
	SPHERE,
	SKY,

	DESC,
};
CU_CONSTANT uint32_t KSampleableLightCount{ 2 };
CU_CONSTANT LightType KOrderedTypes[]{ LightType::SPHERE, LightType::DISK };
CU_CONSTANT uint32_t KSphereLightIndex{ 0 };
CU_CONSTANT uint32_t KDiskLightIndex{ 1 };
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
	uint32_t materialIndex{};
	uint32_t primitiveDataIndex{};
};

namespace LightTree
{
	constexpr inline uint32_t KLightCountBits{ 4 };
	constexpr inline uint32_t KLightOffsetBits{ 31 - KLightCountBits };
	constexpr inline uint32_t KMaxLeafLightCount{ 1 << KLightCountBits };
	constexpr inline uint32_t KMaxLeafLightOffset{ 1 << KLightOffsetBits };
	constexpr inline uint32_t KMaxDepth{ 64 };

	struct NodeAttributes
	{
		float spatialMean[3]{};
		float spatialVariance{};
		float flux{};
		float averageDirection[3]{};
		float sharpness{};

		static NodeAttributes add(const NodeAttributes& nodeA, const NodeAttributes& nodeB)
		{
			NodeAttributes res{};

			const float fluxWeightA{ nodeA.flux / (nodeA.flux + nodeB.flux) };
			const float fluxWeightB{ nodeB.flux / (nodeA.flux + nodeB.flux) };

			float averageDirection[3]{
				fluxWeightA * nodeA.averageDirection[0] + fluxWeightB * nodeB.averageDirection[0],
				fluxWeightA * nodeA.averageDirection[1] + fluxWeightB * nodeB.averageDirection[1],
				fluxWeightA * nodeA.averageDirection[2] + fluxWeightB * nodeB.averageDirection[2], };
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
			const float& adl{ averageDirectionLength };
			res.sharpness = (3.0f * adl - (adl * adl * adl)) / (1.0f - (adl * adl));
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
			: spatialMean{ attributes.spatialMean[0], attributes.spatialMean[1], attributes.spatialMean[2] }, spatialVariance{ attributes.spatialVariance },
			packedAverageDirection{ Octohedral::encodeU32(attributes.averageDirection[0], attributes.averageDirection[1], attributes.averageDirection[2]) },
			sharpness{ attributes.sharpness }, flux{ attributes.flux },
			coreData{ ((1u << 31u) - 1u) & rightChildIndex }
		{}
		PackedNode(const NodeAttributes& attributes,
				uint32_t lightOffset, uint32_t lightCount)
			: spatialMean{ attributes.spatialMean[0], attributes.spatialMean[1], attributes.spatialMean[2] }, spatialVariance{ attributes.spatialVariance },
			packedAverageDirection{ Octohedral::encodeU32(attributes.averageDirection[0], attributes.averageDirection[1], attributes.averageDirection[2]) },
			sharpness{ attributes.sharpness }, flux{ attributes.flux },
			coreData{ (1u << 31u) | ((lightCount & ((1u << KLightCountBits) - 1u)) << KLightOffsetBits) | (lightOffset & ((1u << KLightOffsetBits) - 1u)) }
		{}
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

	CU_HOSTDEVICE CU_INLINE UnpackedNode unpackNode(const PackedNode& packedNode)
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

		if (packedNode.coreData >> 31 == 0)
		{
			res.core.branch.rightChildIndex = packedNode.coreData & ((1u << 31u) - 1u);
		}
		else
		{
			res.core.leaf.lightCount = (packedNode.coreData >> KLightOffsetBits) & ((1u << KLightCountBits) - 1u);
			res.core.leaf.lightOffset = packedNode.coreData & ((1u << KLightOffsetBits) - 1u);
		}

		return res;
	}
}
