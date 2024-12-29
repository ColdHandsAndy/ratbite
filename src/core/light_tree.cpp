#include "../core/light_tree.h"

#include <cmath>
#include <numbers>
#include <algorithm>
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include "../core/scene.h"
#include "../core/debug_macros.h"
#include "../core/math_util.h"

static LightTree::NodeAttributes generateLightNodeAttributes(const SceneData& scene, const LightTree::Builder::SortData& sortData)
{
	namespace SG = SphericalGaussian;

	LightTree::NodeAttributes resAttr{};

	uint32_t modelIdx{ sortData.lightDataRef.triangleRef.modelIndex };
	uint32_t subsetIdx{ sortData.lightDataRef.triangleRef.subsetIndex };
	uint32_t triangleIdx{ sortData.lightDataRef.triangleRef.triangleIndex };

	SceneData::EmissiveMeshSubset::TriangleData triangle{ scene.models[modelIdx].instancedEmissiveMeshSubsets[subsetIdx].triangles[triangleIdx] };

	glm::vec3 e1{ triangle.v1WS - triangle.v0WS };
	glm::vec3 e2{ triangle.v2WS - triangle.v0WS };
	glm::vec3 normal{ glm::normalize(glm::cross(e1, e2)) };
	glm::vec3 spatialMean{ (triangle.v0WS + triangle.v1WS + triangle.v2WS) / 3.0f };
	// Spatial means are in the world space
	resAttr.spatialMean[0] = spatialMean.x;
	resAttr.spatialMean[1] = spatialMean.y;
	resAttr.spatialMean[2] = spatialMean.z;
	resAttr.spatialVariance =
		((e1.x * e1.x + e1.y * e1.y + e1.z * e1.z) + (e2.x * e2.x + e2.y * e2.y + e2.z * e2.z) - glm::dot(e1, e2)) / 18.0f;
	resAttr.averageDirection[0] = normal.x;
	resAttr.averageDirection[1] = normal.y;
	resAttr.averageDirection[2] = normal.z;
	resAttr.sharpness = SG::VMFAxisLengthToSharpness(0.5f);
	resAttr.flux = sortData.flux; // Using this value since it is already transform corrected

	return resAttr;
}

namespace LightTree
{
	Tree Builder::build(const SceneData& scene, SortData* lightsSortData, const int lightCount, const int triangleLightCount)
	{
		Tree tree{};

		// Check if there are lights in the scene
			// If not - return
		if (lightCount == 0)
			return tree;
		else if (lightCount > KMaxLeafLightOffset + maxLightCountPerLeaf)
			R_ERR_LOG("Light count exceeds the maximum supported amount of lights.");

		// Initialize Tree nodes and bitmasks
		tree.reservedNodeCount = lightCount * 2;
		tree.nodes = new PackedNode[tree.reservedNodeCount];
		tree.nodeCount = 0;
		tree.lightPointers = new LightPointer[lightCount];
		tree.lightCount = lightCount;
		R_ASSERT_LOG(lightCount == triangleLightCount, "Only triangle lights work now");
		tree.lightCounts[KTriangleLightsArrayIndex] = triangleLightCount;
		tree.bitmaskSets[KTriangleLightsArrayIndex] = new uint64_t[triangleLightCount];
		memset(tree.bitmaskSets[KTriangleLightsArrayIndex], 0xFF, triangleLightCount * sizeof(uint64_t));

		// Build the Tree
		buildNodeHierarchy(scene, 0, 0, lightsSortData, SortRange{0, static_cast<uint32_t>(lightCount)}, tree);
		R_ASSERT_LOG(tree.nodeCount != 0, "No nodes produced by light tree construction");
		for (int i{ 0 }; i < triangleLightCount; ++i)
			R_ASSERT_LOG(tree.bitmaskSets[KTriangleLightsArrayIndex][i] != UINT64_MAX, "Invalid light tree bitmask generated.");

		// Fill the branch node spatial and directional data
		fillBranchNodeAttributes(0, tree);

		return tree;
	}
	uint32_t Builder::buildNodeHierarchy(const SceneData& scene, uint64_t bitmask, uint32_t depth, SortData* sortData, const SortRange& range, Tree& tree)
	{
		auto reallocateTreeNodes{ [&tree]()
			{
				printf("Tree node data was reallocated.\n");
				PackedNode* newAlloc{ new PackedNode[tree.reservedNodeCount * 2] };
				memcpy(newAlloc, tree.nodes, sizeof(PackedNode) * tree.reservedNodeCount);
				delete[] tree.nodes;
				tree.reservedNodeCount *= 2;
				tree.nodes = newAlloc;
			} };

		float flux{ 0.0 };
		AABB::BBox bounds{ AABB::BBox::getDefault() };

		// Create BBox union for the entire range
		for (uint32_t i{ range.begin }; i < range.end; ++i)
		{
			bounds = AABB::createUnion(bounds, sortData[i].bounds);
			flux += sortData[i].flux;
		}

		SplitResult split{ range.length() > maxLightCountPerLeaf ? splitFunction(sortData, range, bounds, flux) : SplitResult{} };

		if (split.isValid())
		{
			// Split sort with SplitResult
			auto compFunc{ [dim = split.axis](const SortData& a, const SortData& b) -> bool
				{
					float aC[3]{};
					a.bounds.getCenter(aC[0], aC[1], aC[2]);
					float bC[3]{};
					b.bounds.getCenter(bC[0], bC[1], bC[2]);
					return aC[dim] < bC[dim];
				} };
			std::nth_element(sortData + range.begin, sortData + split.index, sortData + range.end, compFunc);

			// Push space for this branch node and increase node count
			int nodeIndex{ tree.nodeCount };
			if (tree.nodeCount == tree.reservedNodeCount)
				reallocateTreeNodes();
			tree.nodeCount += 1;

			// Build child branches and leafs
			if (depth == KMaxDepth)
				R_ERR_LOG("Max light tree depth exceeded.");
			uint32_t leftIndex{ buildNodeHierarchy(scene, bitmask | (0ull << depth), depth + 1, sortData, SortRange{range.begin, split.index}, tree) };
			uint32_t rightIndex{ buildNodeHierarchy(scene, bitmask | (1ull << depth), depth + 1, sortData, SortRange{split.index, range.end}, tree) };

			// Fill the node info
			PackedNode branch{};
			branch.packCoreData(rightIndex);
			tree.nodes[nodeIndex] = branch;

			return nodeIndex;
		}
		else
		{
			// Allocate and fill the new leaf node
			int nodeIndex{ tree.nodeCount };
			if (tree.nodeCount == tree.reservedNodeCount)
				reallocateTreeNodes();
			tree.nodeCount += 1;

			NodeAttributes nodeAttributes{ generateLightNodeAttributes(scene, sortData[range.begin]) };
			// Get every light and compute their combined node attributes
			for (uint32_t i{ range.begin + 1 }; i < range.end; ++i)
				nodeAttributes = NodeAttributes::add(nodeAttributes, generateLightNodeAttributes(scene, sortData[i]));

			PackedNode leaf{ nodeAttributes, range.begin, range.length() };
			tree.nodes[nodeIndex] = leaf;

			// Fill bitmasks for the leaf node's lights
			for (uint32_t i{ range.begin }; i < range.end; ++i)
			{
				tree.lightPointers[i].pack(sortData[i].lightType, sortData[i].lightIndex);
				tree.bitmaskSets[KTriangleLightsArrayIndex][sortData[i].lightIndex] = bitmask;
			}

			return nodeIndex;
		}
	}
	Builder::SplitResult Builder::splitFunction(SortData* sortData, const SortRange& range, const AABB::BBox& bounds, float flux)
	{
		constexpr float KCosAngleEntireCone{ -1.0f };

		// Compute the minimum cos cone angle that includes both given cones
		const auto computeCosConeAngle{ [](const float* coneDirA, const float cosThetaA, const float* coneDirB, const float cosThetaB) {
			float cosResult{ KCosAngleEntireCone };
			if (cosThetaA != KCosAngleEntireCone && cosThetaB != KCosAngleEntireCone)
			{
				const float cosDiffTheta{ coneDirA[0] * coneDirB[0] + coneDirA[1] * coneDirB[1] + coneDirA[2] * coneDirB[2] };
				const float sinDiffTheta{ std::sqrt(1.0f - cosDiffTheta * cosDiffTheta) };
				const float sinThetaB{ std::sqrt(1.0f - cosThetaB * cosThetaB) };

				// Rotate (cosDiffTheta, sinDiffTheta) counterclockwise by the other cone's spread angle.
				float cosTotalTheta{ cosThetaB * cosDiffTheta - sinThetaB * sinDiffTheta };
				float sinTotalTheta{ sinThetaB * cosDiffTheta + cosThetaB * sinDiffTheta };

				// If the total angle is less than pi, store the new cone angle.
				// Otherwise, the bounding cone will be deactivated because it would represent the whole sphere.
				if (sinTotalTheta > 0.0f)
					cosResult = std::min(cosThetaB, cosTotalTheta);
			}
			return cosResult;
		} };
		// Function evaluating SAOH metric for a node
		const auto evalSAOH{ [](const AABB::BBox& bounds, const float flux, const float cosTheta) {
			constexpr float pi{ std::numbers::pi_v<float> };
			float aabbCost{ bounds.isValid() ? bounds.getSurfaceArea() : 0.0f };
			float thetaO{ cosTheta != KCosAngleEntireCone ? std::acos(std::max(std::min(cosTheta, 1.0f), -1.0f)) : pi };
			float thetaW{ std::min(thetaO + pi / 2.0f, pi) };
			float sinThetaO{ std::sin(thetaO) };
			float cosThetaO{ std::cos(thetaO) };
			float orientationCost{ 2.0f * pi * (1.0f - cosThetaO) +
				pi / 2.0f * (2.0f * thetaW * sinThetaO - std::cos(thetaO - 2.0f * thetaW) - 2.0f * thetaO * sinThetaO + cosThetaO) };
			float cost{ flux * aabbCost * orientationCost };
			R_ASSERT_LOG(cost >= 0.0f && !std::isnan(cost) && !std::isinf(cost), "Calculated SAOH cost is invalid.");
			return cost;
		} };

		// Initialize split data and dimensions
		float splitCost{ std::numeric_limits<float>::max() };
		Builder::SplitResult split{};
		float dimensions[3]{};
		bounds.getDimensions(dimensions[0], dimensions[1], dimensions[2]);
		uint32_t largestDimension{ dimensions[0] > dimensions[1] && dimensions[0] > dimensions[2] ? 0u : (dimensions[1] > dimensions[2] ? 1u : 2u) };

		// Bins approximate multiple lights which are close to each other and they are used for SAOH computation for a speedup
		struct Bin
		{
			AABB::BBox bounds{};
			uint32_t lCount{ 0 };
			float flux{ 0.0f };
			float coneDirection[3]{ 0.0f, 0.0f, 0.0f };
			float cosConeAngle{ 1.0f };

			Bin() = default;
			Bin(const SortData& sd)
				: bounds{ sd.bounds }, lCount{ 1 }, flux{ sd.flux },
				coneDirection{ sd.coneDirection[0], sd.coneDirection[1], sd.coneDirection[2] }, cosConeAngle{ sd.cosConeAngle } {}
			Bin& operator|= (const Bin& rhs)
			{
				bounds = AABB::createUnion(bounds, rhs.bounds);
				lCount += rhs.lCount;
				flux += rhs.flux;
				coneDirection[0] += rhs.coneDirection[0];
				coneDirection[1] += rhs.coneDirection[1];
				coneDirection[2] += rhs.coneDirection[2];
				return *this;
			}
		};

		// Allocate bins and costs memory
		const uint32_t costsArraySize{ binCount - 1 };
		Bin* bins{ new Bin[binCount] };
		float* costs{ new float[costsArraySize] };

		// Function that computes the best split along a given dimension
		const auto binAlongDimension = [this, &bins, &costs, &costsArraySize,
			  &range, &sortData, &bounds,
			  &split, &splitCost, &dimensions, &largestDimension,
			  &computeCosConeAngle, &evalSAOH](const uint32_t dimension)
		{
			// Bin ID is an index into equally partitioned space
			auto getBinId = [&](const SortData& sd)
			{
				float bmin{ bounds.min[dimension] };
				float bmax{ bounds.max[dimension] };
				float w{ bmax - bmin };
				R_ASSERT(w >= 0.0f);
				float scale{ w > std::numeric_limits<float>::min() ? static_cast<float>(binCount) / w : 0.0f };
				float c[3]{};
				sd.bounds.getCenter(c[0], c[1], c[2]);
				float p{ c[dimension] };
				R_ASSERT(bmin <= p && p <= bmax);
				return std::min(static_cast<uint32_t>((p - bmin) * scale), binCount - 1);
			};

			// Initialize bins
			for (int i{ 0 }; i < binCount; ++i)
				bins[i] = Bin{};
			for (uint32_t i{ range.begin }; i < range.end; ++i)
				bins[getBinId(sortData[i])] |= sortData[i];

			// Compute bins cosConeAngles and coneDirections
			for (int i{ 0 }; i < binCount; ++i)
			{
				Bin& bin{ bins[i] };
				float coneDirL{ std::sqrt(
						bin.coneDirection[0] * bin.coneDirection[0] +
						bin.coneDirection[1] * bin.coneDirection[1] +
						bin.coneDirection[2] * bin.coneDirection[2]) };
				if (coneDirL < std::numeric_limits<float>::min())
				{
					bin.cosConeAngle = KCosAngleEntireCone;
				}
				else
				{
					bin.cosConeAngle = 1.0f;
					bin.coneDirection[0] /= coneDirL;
					bin.coneDirection[1] /= coneDirL;
					bin.coneDirection[2] /= coneDirL;
				}
			}
			for (uint32_t i{ range.begin }; i < range.end; ++i)
			{
				const SortData& sd{ sortData[i] };
				Bin& bin{ bins[getBinId(sd)] };
				bin.cosConeAngle = computeCosConeAngle(bin.coneDirection, bin.cosConeAngle, sd.coneDirection, sd.cosConeAngle);
			}

			// Sweeping over the bins to calculate costs
			Bin total{};
			for (int i{ 0 }; i < costsArraySize; ++i)
			{
				total |= bins[i];

				float cosTheta{ KCosAngleEntireCone };
				float coneDirL{ std::sqrt(
						total.coneDirection[0] * total.coneDirection[0] +
						total.coneDirection[1] * total.coneDirection[1] +
						total.coneDirection[2] * total.coneDirection[2]) };
				if (coneDirL > std::numeric_limits<float>::min())
				{
					cosTheta = 1.0f;
					float coneDir[3]{
						total.coneDirection[0] / coneDirL,
						total.coneDirection[1] / coneDirL,
						total.coneDirection[2] / coneDirL, };
					for (int j{ 0 }; j <= i; ++j)
						cosTheta = computeCosConeAngle(coneDir, cosTheta, bins[j].coneDirection, bins[j].cosConeAngle);
				}

				costs[i] = evalSAOH(total.bounds, total.flux, cosTheta);
			}
			total = Bin{};
			for (int i{ static_cast<int>(costsArraySize) }; i > 0; --i)
			{
				total |= bins[i];

				float cosTheta{ KCosAngleEntireCone };
				float coneDirL{ std::sqrt(
						total.coneDirection[0] * total.coneDirection[0] +
						total.coneDirection[1] * total.coneDirection[1] +
						total.coneDirection[2] * total.coneDirection[2]) };
				if (coneDirL > std::numeric_limits<float>::min())
				{
					cosTheta = 1.0f;
					float coneDir[3]{
						total.coneDirection[0] / coneDirL,
						total.coneDirection[1] / coneDirL,
						total.coneDirection[2] / coneDirL, };
					for (int j = i; j <= costsArraySize; ++j)
						cosTheta = computeCosConeAngle(coneDir, cosTheta, bins[j].coneDirection, bins[j].cosConeAngle);
				}

				costs[i - 1] += evalSAOH(total.bounds, total.flux, cosTheta);
			}

			// Get the best split
			float axisSplitCost{ std::numeric_limits<float>::max() };
			SplitResult axisSplit{ .axis = dimension, .index = 0 };
			for (uint32_t i{ 0 }, lIdx{ range.begin }; i < costsArraySize; ++i)
			{
				lIdx += bins[i].lCount;
				if (costs[i] < axisSplitCost)
				{
					axisSplitCost = costs[i];
					axisSplit = { .axis = dimension, .index = lIdx };
				}
			}
			R_ASSERT_LOG(range.begin <= axisSplit.index && axisSplit.index <= range.end, "Split index is outside the range");

			// Correct split cost to avoid long and thin cuts
			axisSplitCost *= dimensions[largestDimension] / dimensions[dimension];

			// Return if all lights are on one side of the split
			if (axisSplit.index == range.begin || axisSplit.index == range.end)
				return;

			// Choose the better split between current one and the one we just found
			if (axisSplitCost < splitCost)
			{
				splitCost = axisSplitCost;
				split = axisSplit;
			}
		};

		if (splitAlongLargestDimensionOnly)
		{
			binAlongDimension(largestDimension);
		}
		else
		{
			for (int dim{ 0 }; dim < 3; ++dim)
			{
				binAlongDimension(dim);
			}
		}

		if (!split.isValid())
		{
			if (range.length() <= maxLightCountPerLeaf)
				return SplitResult{};
			else
				return SplitResult{.axis = largestDimension, .index = range.middle()};
		}

		// TODO: Compare leaf cost and split cost then create a leaf if necessary

		delete[] bins;
		delete[] costs;
		return split;
	}

	NodeAttributes Builder::fillBranchNodeAttributes(uint32_t nodeIndex, Tree& tree)
	{
		const PackedNode& packedNode{ tree.nodes[nodeIndex] };
		bool isLeaf{};
		const UnpackedNode unpackedNode{ unpackNode(packedNode, isLeaf) };
		if (isLeaf)
		{
			// TODO: Cache average direction for leaf nodes so no precision is lost on decoding
			NodeAttributes attribs{ unpackedNode.attributes };
			return attribs;
		}
		else
		{
			uint32_t leftIndex{ nodeIndex + 1 };
			uint32_t rightIndex{ unpackedNode.core.branch.rightChildIndex };
			NodeAttributes leftNodeAttr{ fillBranchNodeAttributes(leftIndex, tree) };
			NodeAttributes rightNodeAttr{ fillBranchNodeAttributes(rightIndex, tree) };

			NodeAttributes attribs{ NodeAttributes::add(leftNodeAttr, rightNodeAttr) };
			tree.nodes[nodeIndex].packAttributes(attribs);
			return attribs;
		}
	}
}
