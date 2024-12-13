#pragma once

#include <cuda/std/cmath>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include "../core/util.h"

namespace AABB
{
	// Bounded AABB
	struct BBox
	{
		float min[3]{};
		float max[3]{};

		CU_HOSTDEVICE CU_INLINE static BBox getDefault()
		{
			BBox res{};
			res.min[0] = INFINITY;
			res.min[1] = INFINITY;
			res.min[2] = INFINITY;
			res.max[0] = -INFINITY;
			res.max[1] = -INFINITY;
			res.max[2] = -INFINITY;
			return res;
		}
		CU_HOSTDEVICE CU_INLINE void getCenter(float& x, float& y, float& z) const
		{
			x = min[0] + (max[0] - min[0]) * 0.5f; y = min[1] + (max[1] - min[1]) * 0.5f; z = min[2] + (max[2] - min[2]) * 0.5f;
		}
		CU_HOSTDEVICE CU_INLINE void getDimensions(float& x, float& y, float& z) const
		{
			x = max[0] - min[0]; y = max[1] - min[1]; z = max[2] - min[2];
		}
		CU_HOSTDEVICE CU_INLINE float getVolume() const
		{
			float x{}, y{}, z{};
			getDimensions(x, y, z);
			return x * y * z;
		}
	};
	// Centered AABB
	struct CBox
	{
		float center[3]{};
		float extent[3]{}; // Half of the Box dimensions

		CU_HOSTDEVICE CU_INLINE static CBox getDefault()
		{
			CBox res{};
			res.center[0] = 0.0f;
			res.center[1] = 0.0f;
			res.center[2] = 0.0f;
			res.extent[0] = -INFINITY;
			res.extent[1] = -INFINITY;
			res.extent[2] = -INFINITY;
			return res;
		}
		CU_HOSTDEVICE CU_INLINE void getMin(float& x, float& y, float& z) const
		{
			x = center[0] - extent[0]; y = center[1] - extent[1]; z = center[2] - extent[2];
		}
		CU_HOSTDEVICE CU_INLINE void getMax(float& x, float& y, float& z) const
		{
			x = center[0] + extent[0]; y = center[1] + extent[1]; z = center[2] + extent[2];
		}
		CU_HOSTDEVICE CU_INLINE void getDimensions(float& x, float& y, float& z) const
		{
			x = extent[0] * 2.0f; y = extent[1] * 2.0f; z = extent[2] * 2.0f;
		}
		CU_HOSTDEVICE CU_INLINE float getVolume() const
		{
			float x{}, y{}, z{};
			getDimensions(x, y, z);
			return x * y * z;
		}
	};

	CU_HOSTDEVICE CU_INLINE BBox createUnion(const BBox& a, const BBox& b)
	{
		BBox res{};
		res.min[0] = glm::min(a.min[0], b.min[0]);
		res.min[1] = glm::min(a.min[1], b.min[1]);
		res.min[2] = glm::min(a.min[2], b.min[2]);
		res.max[0] = glm::max(a.max[0], b.max[0]);
		res.max[1] = glm::max(a.max[1], b.max[1]);
		res.max[2] = glm::max(a.max[2], b.max[2]);
		return res;
	}
	CU_HOSTDEVICE CU_INLINE CBox createUnion(const CBox& a, const CBox& b)
	{
		CBox res{};
		float aMin[3]{};
		a.getMin(aMin[0], aMin[1], aMin[2]);
		float aMax[3]{};
		a.getMax(aMax[0], aMax[1], aMax[2]);
		float bMin[3]{};
		b.getMin(bMin[0], bMin[1], bMin[2]);
		float bMax[3]{};
		b.getMax(bMax[0], bMax[1], bMax[2]);
		float resMin[3]{
			glm::min(aMin[0], bMin[0]),
			glm::min(aMin[1], bMin[1]),
			glm::min(aMin[2], bMin[2]),
		};
		float resMax[3]{
			glm::max(aMax[0], bMax[0]),
			glm::max(aMax[1], bMax[1]),
			glm::max(aMax[2], bMax[2]),
		};
		float extent[3]{
			resMax[0] - resMin[0],
			resMax[1] - resMin[1],
			resMax[2] - resMin[2],
		};
		res.extent[0] = (resMax[0] - resMin[0]) * 0.5f;
		res.extent[1] = (resMax[1] - resMin[1]) * 0.5f;
		res.extent[2] = (resMax[2] - resMin[2]) * 0.5f;
		res.center[0] = resMin[0] + res.extent[0];
		res.center[1] = resMin[1] + res.extent[1];
		res.center[2] = resMin[2] + res.extent[2];
		return res;
	}
}

namespace Octohedral
{
	CU_HOSTDEVICE CU_INLINE glm::vec2 encode(const glm::vec3& vec)
	{
		const float& x{ vec.x };
		const float& y{ vec.y };
		const float& z{ vec.z };

		glm::vec2 p{ glm::vec2{x, y} * (1.0f / (glm::abs(x) + glm::abs(y) + glm::abs(z))) };
		glm::vec2 res{
			z <= 0.0f
			?
			(glm::vec2{1.0f} - glm::vec2{glm::abs(p.y), glm::abs(p.x)}) * glm::vec2{(p.x >= 0.0f) ? +1.0f : -1.0f, (p.y >= 0.0f) ? +1.0f : -1.0f}
			:
			p
		};
		return res;
	}
	CU_HOSTDEVICE CU_INLINE glm::vec2 encode(float x, float y, float z)
	{
		return encode(glm::vec3{x, y, z});
	}
	CU_HOSTDEVICE CU_INLINE glm::vec2 encode(float* vec)
	{
		return encode(glm::vec3{vec[0], vec[1], vec[2]});
	}
	CU_HOSTDEVICE CU_INLINE glm::vec3 decode(const glm::vec2& encvec)
	{
		const float& u{ encvec.x };
		const float& v{ encvec.y };

		glm::vec3 vec;
		vec.z = 1.0f - glm::abs(u) - glm::abs(v);
		vec.x = u;
		vec.y = v;

		float t{ glm::max(0.0f, -vec.z) };

		vec.x += vec.x >= 0.0f ? -t : t;
		vec.y += vec.y >= 0.0f ? -t : t;

		return glm::normalize(vec);
	}
	CU_HOSTDEVICE CU_INLINE void decode(float encx, float ency, float& x, float& y, float& z)
	{
		glm::vec3 vec{ decode(glm::vec2{encx, ency}) };
		x = vec.x; y = vec.y; z = vec.z;
	}
	CU_HOSTDEVICE CU_INLINE void decode(const float* encvec, float* decvec)
	{
		glm::vec3 vec{ decode(glm::vec2{encvec[0], encvec[1]}) };
		decvec[0] = vec.x; decvec[1] = vec.y; decvec[2] = vec.z;
	}

	CU_HOSTDEVICE CU_INLINE uint32_t encodeU32(const glm::vec3& vec)
	{
		constexpr float uint16MaxFloat{ 65535.99f };
		glm::vec2 pre{ encode(vec) };
		uint32_t x{ static_cast<uint32_t>(glm::clamp(pre.x * 0.5f + 0.5f, 0.0f, 1.0f) * uint16MaxFloat) << 0 };
		uint32_t y{ static_cast<uint32_t>(glm::clamp(pre.y * 0.5f + 0.5f, 0.0f, 1.0f) * uint16MaxFloat) << 16 };
		uint32_t res{ y | x };
		return res;
	}
	CU_HOSTDEVICE CU_INLINE uint32_t encodeU32(float x, float y, float z)
	{
		return encodeU32(glm::vec3{x, y, z});
	}
	CU_HOSTDEVICE CU_INLINE glm::vec3 decodeU32(uint32_t encvec)
	{
		constexpr float uint16MaxFloatNormalizer{ 1.0f / 65535.0f };
		glm::vec2 pre;
		pre.x = static_cast<float>(encvec & 0xFFFF) * uint16MaxFloatNormalizer;
		pre.y = static_cast<float>(encvec >> 16) * uint16MaxFloatNormalizer;
		return decode(pre * 2.0f - 1.0f);
	}
	CU_HOSTDEVICE CU_INLINE void decodeU32(uint32_t encvec, float& x, float& y, float& z)
	{
		glm::vec3 vec{ decodeU32(encvec) };
		x = vec.x; y = vec.y; z = vec.z;
	}
}
