#pragma once

#include <cstdint>
#include <cmath>

#include "../core/light_tree_types.h"

// Light tree implementatino is based on
// "Importance Sampling of Many Lights on the GPU"
// https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_18.pdf
// and
// "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting"
// https://gpuopen.com/download/publications/Hierarchical_Light_Sampling_with_Accurate_Spherical_Gaussian_Lighting.pdf

class SceneData;
// TODO: Add other kinds of lights to the LightTree
namespace LightTree
{
	struct Tree
	{
		PackedNode* nodes{};
		int nodeCount{};
		int reservedNodeCount{};
		enum BitmaskSet
		{
			TRIANGLE_SET,
			// DISK_SET,
			// SPHERE_SET,

			ALL_SETS
		};
		uint64_t* bitmaskSets[BitmaskSet::ALL_SETS]{};

		void clear()
		{
			for (int i{ 0 }; i < ARRAYSIZE(bitmaskSets); ++i)
			{
				if (bitmaskSets[i] != nullptr)
				{
					delete[] bitmaskSets[i];
					bitmaskSets[i] = nullptr;
				}
			}
			if (nodes != nullptr)
			{
				delete[] nodes;
				nodes = nullptr;
				reservedNodeCount = 0;
				nodeCount = 0;
			}
		}
	};
	struct Builder
	{
		struct SortData
		{
			AABB::BBox bound{};
			float center[3]{};
			float coneDirection[3]{};
			float cosConeAngle{};
			float flux{};
			// LightType lightType{};
			union
			{
				struct
				{
					uint32_t modelIndex{};
					uint32_t subsetIndex{};
					uint32_t triangleIndex{};
				} triangleRef;
			} lightDataRef;
		};
		Tree build(const SceneData& scene, SortData* lightsSortData, const int lightCount);
		
	private:
		struct SortRange
		{
			uint32_t begin{};
			uint32_t end{};

			SortRange(uint32_t begin, uint32_t end)
				: begin{ begin }, end{ end }
			{ 
				assert(begin <= end && "Invalid range");
			}
			uint32_t middle() const
			{
				return (begin + end) / 2;
			}
			uint32_t length() const
			{
				return end - begin;
			}
		};
		struct SplitResult
		{
			uint32_t axis{ UINT32_MAX };
			uint32_t index{ UINT32_MAX };

			bool isValid() const { return axis != UINT32_MAX && index != UINT32_MAX; }
		};
		uint32_t buildNodeHierarchy(const SceneData& scene, uint64_t bitmask, uint32_t depth, SortData* sortData, const SortRange& range, Tree& tree);
		SplitResult splitFunction(SortData* sortData, const SortRange& range, const AABB::BBox& bound, float flux);
		NodeAttributes fillBranchNodeAttributes(uint32_t nodeIndex, Tree& tree);
	};
}
