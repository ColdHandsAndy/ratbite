#include <algorithm>
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include "../core/light_tree_types.h"
#include "../core/light_tree.h"
#include "../core/scene.h"
#include "../core/debug_macros.h"

namespace
{
	LightTree::NodeAttributes generateLightNodeAttributes(const SceneData& scene, const LightTree::Builder::SortData& lightsSortData)
	{
		LightTree::NodeAttributes resAttr{};

		uint32_t modelIdx{ lightsSortData.lightDataRef.triangleRef.modelIndex };
		uint32_t subsetIdx{ lightsSortData.lightDataRef.triangleRef.subsetIndex };
		uint32_t triangleIdx{ lightsSortData.lightDataRef.triangleRef.triangleIndex };

		SceneData::EmissiveMeshSubset::TriangleData triangle{ scene.models[modelIdx].instancedEmissiveMeshSubsets[subsetIdx].triangles[triangleIdx] };
		const SceneData::Instance& instance{ scene.models[modelIdx].instances[scene.models[modelIdx].instancedEmissiveMeshSubsets[subsetIdx].instanceIndex] };
		triangle.v0 = instance.transform * glm::vec4{triangle.v0, 1.0f};
		triangle.v1 = instance.transform * glm::vec4{triangle.v1, 1.0f};
		triangle.v2 = instance.transform * glm::vec4{triangle.v2, 1.0f};

		glm::vec3 e1{ triangle.v1 - triangle.v0 };
		glm::vec3 e2{ triangle.v2 - triangle.v0 };
		glm::vec3 normal{ glm::normalize(glm::cross(e1, e2)) };
		glm::vec3 spatialMean{ (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f };
		resAttr.spatialMean[0] = spatialMean.x;
		resAttr.spatialMean[1] = spatialMean.y;
		resAttr.spatialMean[2] = spatialMean.z;
		resAttr.spatialVariance =
			((e1.x * e1.x + e1.y * e1.y + e1.z * e1.z) + (e2.x * e2.x + e2.y * e2.y + e2.z * e2.z) - glm::dot(e1, e2))
			/ 18.0f;
		resAttr.averageDirection[0] = normal.x;
		resAttr.averageDirection[1] = normal.y;
		resAttr.averageDirection[2] = normal.z;
		resAttr.sharpness = (3.0f * 0.5f - 0.5f * 0.5f * 0.5f) / (1.0f - 0.5f * 0.5f);
		resAttr.flux = lightsSortData.flux; // Using this value since it is already transform corrected

		return resAttr;
	}
}

namespace LightTree
{
	Tree Builder::build(const SceneData& scene, SortData* lightsSortData, const int lightCount)
	{
		// TODO: (IMPORTANT) Ensure all light flux values are above the given threshold and were transform corrected
		Tree tree{};

		// Check if there are lights in the scene
			// If not - return
		if (lightCount == 0)
			return tree;
		else if (lightCount > KMaxLeafLightOffset + KMaxLeafLightCount)
			R_ERR_LOG("Light count exceeds the maximum supported amount of lights.");

		// Initialize Tree nodes and bitmasks
		tree.reservedNodeCount = lightCount * 2;
		tree.nodes = new PackedNode[tree.reservedNodeCount];
		tree.nodeCount = 0;
		tree.bitmaskSets[Tree::TRIANGLE_SET] = new uint64_t[lightCount];
		memset(tree.bitmaskSets[Tree::TRIANGLE_SET], 0xFF, lightCount * sizeof(uint64_t));

		// Build the Tree
		buildNodeHierarchy(scene, 0, 0, lightsSortData, SortRange{0, static_cast<uint32_t>(lightCount)}, tree);
		R_ASSERT_LOG(tree.nodeCount != 0, "No nodes produced by light tree construction");
		for (int i{ 0 }; i < lightCount; ++i)
			R_ASSERT_LOG(tree.bitmaskSets[Tree::TRIANGLE_SET][i] != UINT64_MAX, "Invalid light tree bitmask generated.");

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
				tree.reservedNodeCount *= 2;
				tree.nodes = newAlloc;
			} };

		float flux{ 0.0 };
		AABB::BBox bound{ AABB::BBox::getDefault() };

		// Create BBox union for the entire range
		for (uint32_t i{ range.begin }; i < range.end; ++i)
		{
			bound = AABB::createUnion(bound, sortData[i].bound);
			flux += sortData[i].flux;
		}

		SplitResult split{ range.length() < KMaxLeafLightCount ? splitFunction(sortData, range, bound, flux) : SplitResult{} };

		if (split.isValid())
		{
			// Split sort with SplitResult
			auto compFunc{ [dim = split.axis](const SortData& a, const SortData& b) -> bool
				{
					float aC[3]{};
					a.bound.getCenter(aC[0], aC[1], aC[2]);
					float bC[3]{};
					b.bound.getCenter(bC[0], bC[1], bC[2]);
					return aC[dim] < bC[dim];
				} };
			std::nth_element(sortData + range.begin, sortData + split.index, sortData + range.end, compFunc);

			// Push space for this branch node and increase node count
			int nodeIndex{ tree.nodeCount };
			if (tree.nodeCount == tree.reservedNodeCount)
				reallocateTreeNodes();
			tree.nodeCount += 1;

			// Build child branches and leafs
			if (KMaxDepth)
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
				tree.bitmaskSets[Tree::TRIANGLE_SET][i] = bitmask;
			}

			return nodeIndex;
		}
	}
	Builder::SplitResult Builder::splitFunction(SortData* sortData, const SortRange& range, const AABB::BBox& bound, float flux)
	{
		// Placeholder middle split ( TODO: - BAAAAAD. REPLACE IT IMEDIATELY)
		Builder::SplitResult split{};
		float dims[3]{};
		bound.getDimensions(dims[0], dims[1], dims[2]);
		split.axis = (dims[2] > dims[0] && dims[2] > dims[1]) ? 2 : (dims[1] > dims[0] ? 1 : 0);
		split.index = range.middle();
		return split;
	}

	NodeAttributes Builder::fillBranchNodeAttributes(uint32_t nodeIndex, Tree& tree)
	{
		const PackedNode& packedNode{ tree.nodes[nodeIndex] };
		const UnpackedNode unpackedNode{ unpackNode(packedNode) };
		if (packedNode.isLeaf())
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
