#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cmath>
#include <cuda/std/concepts>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/util_macros.h"

namespace geometry
{
	//Modified version of https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_6.pdf
	CU_DEVICE CU_INLINE glm::vec3 offsetRay(const glm::vec3& p, const glm::vec3& n)
	{
		constexpr float originInterval{ 1.0f / 32.0f };
		constexpr float floatScale{ 1.0f / 65536.0f };
		constexpr float intScale{ 256.0f };

		glm::ivec3 intOff{ intScale * n.x, intScale * n.y, intScale * n.z };

		glm::vec3 pIntOff{ __int_as_float(__float_as_int(p.x) + ((p.x < 0.0f) ? -intOff.x : intOff.x)),
			__int_as_float(__float_as_int(p.y) + ((p.y < 0.0f) ? -intOff.y : intOff.y)),
			__int_as_float(__float_as_int(p.z) + ((p.z < 0.0f) ? -intOff.z : intOff.z)) };

		return glm::vec3{ fabsf(p.x) < originInterval ? p.x + floatScale * n.x : pIntOff.x,
			fabsf(p.y) < originInterval ? p.y + floatScale * n.y : pIntOff.y,
			fabsf(p.z) < originInterval ? p.z + floatScale * n.z : pIntOff.z };
	}


	template<typename T, typename... U>
		concept AllSameAs = (cuda::std::same_as<T, U> && ...);

	class LocalTransform
	{
	private:
		glm::mat3 m_TBN{};
	public:
		CU_DEVICE CU_INLINE LocalTransform(const glm::vec3& normal)
		{
			float sign{ cuda::std::copysignf(1.0f, normal.z) };
			float a{ -1.0f / (sign + normal.z) };
			float b{ normal.x * normal.y * a };
			m_TBN[2] = normal;
			m_TBN[0] = glm::vec3(1.0f + sign * (normal.x * normal.x) * a, sign * b, -sign * normal.x);
			m_TBN[1] = glm::vec3(b, sign + (normal.y * normal.y) * a, -normal.y);
			m_TBN = glm::transpose(m_TBN);
		}

		// CU_DEVICE CU_INLINE void toLocal(glm::vec3& a, glm::vec3& b, glm::vec3& c)
		// {
		// 	a = m_TBN * a;
		// 	b = m_TBN * b;
		// 	c = m_TBN * c;
		// }
		// CU_DEVICE CU_INLINE void fromLocal(glm::vec3& a)
		// {
		// 	glm::mat3 invTBN{ glm::transpose(m_TBN) };
		// 	a = invTBN * a;
		// }
		//

		template<typename... Vecs> requires AllSameAs<glm::vec3, Vecs...>
		CU_DEVICE CU_INLINE void toLocal(Vecs&... vecs)
		{
			transform(vecs...);
		}
		template<typename... Vecs> requires AllSameAs<glm::vec3, Vecs...>
		CU_DEVICE CU_INLINE void fromLocal(Vecs&... vecs)
		{
			glm::mat3 invTBN{ glm::transpose(m_TBN) };
			transformInv(invTBN, vecs...);
		}
	private:
		template<typename... Vecs>
		CU_DEVICE CU_INLINE void transform(glm::vec3& vec, Vecs&... vecs)
		{
			vec = m_TBN * vec;
			transform(vecs...);
		}
		CU_DEVICE CU_INLINE void transform(glm::vec3& vec)
		{
			vec = m_TBN * vec;
		}
		template<typename... Vecs>
		CU_DEVICE CU_INLINE void transformInv(const glm::mat3& invTBN, glm::vec3& vec, Vecs&... vecs)
		{
			vec = invTBN * vec;
			transform(invTBN, vecs...);
		}
		CU_DEVICE CU_INLINE void transformInv(const glm::mat3& invTBN, glm::vec3& vec)
		{
			vec = invTBN * vec;
		}
	};
}
