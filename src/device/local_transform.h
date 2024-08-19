#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cmath>
#include <cuda/std/concepts>

#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/util.h"

class LocalTransform
{
public:
	glm::mat3 m_TBN{};
public:
	CU_DEVICE CU_INLINE LocalTransform() = default;
	CU_DEVICE CU_INLINE LocalTransform(const glm::vec3& normal)
		: m_TBN{genToLocalMatrixFromNormal(normal)} {}
	CU_DEVICE CU_INLINE LocalTransform(const glm::mat3& frameMat)
	{
		m_TBN = glm::transpose(frameMat);
	}
	CU_DEVICE CU_INLINE LocalTransform(const glm::quat& frame)
	{
		m_TBN = glm::transpose(glm::mat3_cast(frame));
	}

	CU_DEVICE CU_INLINE static float cos2Theta(const glm::vec3& w) { return w.z * w.z; };
	CU_DEVICE CU_INLINE static float sin2Theta(const glm::vec3& w) { return glm::max(0.0f, 1.0f - cos2Theta(w)); };
	CU_DEVICE CU_INLINE static float tan2Theta(const glm::vec3& w) { return sin2Theta(w) / cos2Theta(w); };
	CU_DEVICE CU_INLINE static float cosTheta(const glm::vec3& w) { return w.z; };
	CU_DEVICE CU_INLINE static float sinTheta(const glm::vec3& w) { return cuda::std::sqrtf(glm::max(0.0f, 1.0f - cos2Theta(w))); };
	CU_DEVICE CU_INLINE static float tanTheta(const glm::vec3& w) { return sinTheta(w) / cosTheta(w); };
	CU_DEVICE CU_INLINE static float cosPhi(const glm::vec3& w) { float sinTh{ sinTheta(w) }; return (sinTh == 0.0f) ? 1.0f : glm::clamp(w.x / sinTh, -1.0f, 1.0f); };
	CU_DEVICE CU_INLINE static float sinPhi(const glm::vec3& w) { float sinTh{ sinTheta(w) }; return (sinTh == 0.0f) ? 0.0f : glm::clamp(w.y / sinTh, -1.0f, 1.0f); };

	CU_DEVICE CU_INLINE static float sin2Theta(float cos2Th) { return glm::max(0.0f, 1.0f - cos2Th); };
	CU_DEVICE CU_INLINE static float tan2Theta(float cos2Th, float sin2Th) { return sin2Th / cos2Th; };
	CU_DEVICE CU_INLINE static float sinTheta(float cos2Th) { return cuda::std::sqrtf(glm::max(0.0f, 1.0f - cos2Th)); };
	CU_DEVICE CU_INLINE static float tanTheta(float cosTh, float sinTh) { return sinTh / cosTh; };
	CU_DEVICE CU_INLINE static float cosPhi(const glm::vec3& w, float sinTh) { return (sinTh == 0.0f) ? 1.0f : glm::clamp(w.x / sinTh, -1.0f, 1.0f); };
	CU_DEVICE CU_INLINE static float sinPhi(const glm::vec3& w, float sinTh) { return (sinTh == 0.0f) ? 0.0f : glm::clamp(w.y / sinTh, -1.0f, 1.0f); };
	
	CU_DEVICE CU_INLINE static glm::mat3 genToLocalMatrixFromNormal(const glm::vec3& normal)
	{
		return glm::transpose(genFromLocalMatrixFromNormal(normal));
	}
	CU_DEVICE CU_INLINE static glm::mat3 genFromLocalMatrixFromNormal(const glm::vec3& normal)
	{
		float sign{ cuda::std::copysignf(1.0f, normal.z) };
		float a{ -1.0f / (sign + normal.z) };
		float b{ normal.x * normal.y * a };
		glm::mat3 TBN;
		TBN[2] = normal;
		TBN[0] = glm::vec3(1.0f + sign * (normal.x * normal.x) * a, sign * b, -sign * normal.x);
		TBN[1] = glm::vec3(b, sign + (normal.y * normal.y) * a, -normal.y);
		return TBN;
	}

	template<typename... Vecs> requires AllSameAs<glm::vec3, Vecs...>
	CU_DEVICE CU_INLINE void toLocal(Vecs&... vecs) const
	{
		transform(vecs...);
	}
	template<typename... Vecs> requires AllSameAs<glm::vec3, Vecs...>
	CU_DEVICE CU_INLINE void fromLocal(Vecs&... vecs) const
	{
		glm::mat3 invTBN{ glm::transpose(m_TBN) };
		transformInv(invTBN, vecs...);
	}
private:
	template<typename... Vecs>
	CU_DEVICE CU_INLINE void transform(glm::vec3& vec, Vecs&... vecs) const
	{
		vec = m_TBN * vec;
		transform(vecs...);
	}
	CU_DEVICE CU_INLINE void transform(glm::vec3& vec) const
	{
		vec = m_TBN * vec;
	}
	template<typename... Vecs>
	CU_DEVICE CU_INLINE void transformInv(const glm::mat3& invTBN, glm::vec3& vec, Vecs&... vecs) const
	{
		vec = invTBN * vec;
		transform(invTBN, vecs...);
	}
	CU_DEVICE CU_INLINE void transformInv(const glm::mat3& invTBN, glm::vec3& vec) const
	{
		vec = invTBN * vec;
	}
};
