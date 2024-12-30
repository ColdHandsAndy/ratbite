#pragma once

#include <cstdint>

#include "../core/light_tree_types.h"
#include "../core/debug_macros.h"

// Light tree implementatino is based on
// "Importance Sampling of Many Lights on the GPU"
// https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_18.pdf
// and
// "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting"
// https://gpuopen.com/download/publications/Hierarchical_Light_Sampling_with_Accurate_Spherical_Gaussian_Lighting.pdf

// TODO: Add other kinds of lights to the LightTree
struct SceneData;
namespace LightTree
{
	struct Tree
	{
		PackedNode* nodes{};
		int nodeCount{};
		int reservedNodeCount{};
		LightPointer* lightPointers{};
		int lightCount{};
		uint32_t lightCounts[static_cast<int>(LightType::ALL)]{};
		uint64_t* bitmaskSets[static_cast<int>(LightType::ALL)]{};

		void clear()
		{
			if (nodes != nullptr)
			{
				delete[] nodes;
				nodes = nullptr;
				reservedNodeCount = 0;
				nodeCount = 0;
			}
			if (lightPointers != nullptr)
			{
				delete[] lightPointers;
				lightPointers = nullptr;
				lightCount = 0;
			}
			for (int i{ 0 }; i < ARRAYSIZE(lightCounts); ++i)
			{
				lightCounts[i] = 0;
			}
			for (int i{ 0 }; i < ARRAYSIZE(bitmaskSets); ++i)
			{
				if (bitmaskSets[i] != nullptr)
				{
					delete[] bitmaskSets[i];
					bitmaskSets[i] = nullptr;
				}
			}
		}
	};
	struct Builder
	{
		uint32_t maxLightCountPerLeaf{ 8 };
		uint32_t binCount{ 32 };
		bool splitAlongLargestDimensionOnly{ false };
		bool createLeafsASAP{ true };

		Builder()
		{
			if (maxLightCountPerLeaf > KMaxLeafLightCount || maxLightCountPerLeaf == 0)
			{
				maxLightCountPerLeaf = KMaxLeafLightCount;
				R_LOG("Max light count per leaf option is invalid. Max allowed value is chosen.");
			}
		}

		struct SortData
		{
			AABB::BBox bounds{};
			float coneDirection[3]{};
			float cosConeAngle{};
			float flux{};
			LightType lightType{};
			uint32_t lightIndex{};
			union
			{
				struct
				{
					uint32_t modelIndex{};
					uint32_t subsetIndex{};
					uint32_t triangleIndex{};
				} triangleRef{};
			} lightDataRef{};
		};
		Tree build(const SceneData& scene, SortData* lightsSortData, const int lightCount, const int triangleLightCount);
		
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
