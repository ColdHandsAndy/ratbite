#pragma once

#include <cuda/std/cmath>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/util.h"
#include "../device/util.h"
#include "../device/local_transform.h"

namespace sampling
{
	namespace disk
	{
		CU_DEVICE CU_INLINE glm::vec2 sampleUniform2DConcentric(glm::vec2 uv)
		{
			uv = uv * 2.0f - 1.0f;
			float theta{};
			float r{};
			if (cuda::std::fabs(uv.x) > cuda::std::fabs(uv.y))
			{
				r = uv.x;
				theta = (glm::pi<float>() / 4.0f) * (uv.y / uv.x);
			}
			else
			{
				r = uv.y;
				theta = (glm::pi<float>() / 2.0f) - (glm::pi<float>() / 4.0f) * (uv.x / uv.y);
			}
			uv.x = r * cuda::std::cosf(theta);
			uv.y = r * cuda::std::sinf(theta);
			return uv;
		}
		CU_DEVICE CU_INLINE glm::vec2 sampleUniform2DPolar(const glm::vec2& uv)
		{
			float r{ cuda::std::sqrtf(uv.x) };
			float theta{ 2.0f * glm::pi<float>() * uv.y };
			return glm::vec2{r * cuda::std::cosf(theta), r * cuda::std::sinf(theta) };
		}
		CU_DEVICE CU_INLINE glm::vec3 sampleUniform3D(const glm::vec2& uv, const glm::quat& frame)
		{
			glm::mat3 matFrame{ glm::mat3_cast(frame) };
			glm::vec2 xy{ sampleUniform2DConcentric(uv) };
			return matFrame[0] * xy.x + matFrame[1] * xy.y;
		}
		CU_DEVICE CU_INLINE glm::vec3 sampleUniform3D(const glm::vec2& uv, const glm::mat3& matFrame)
		{
			glm::vec2 xy{ sampleUniform2DConcentric(uv) };
			return matFrame[0] * xy.x + matFrame[1] * xy.y;
		}
	}
	namespace sphere
	{
		CU_DEVICE CU_INLINE glm::vec3 sampleUniformWorldSolidAngle(const glm::vec2& uv, const glm::vec3& o, const glm::vec3& c, float r, float& pdf)
		{
			glm::vec3 cToO{ o - c };
			float d{ glm::length(cToO) };
			cToO /= d;
			if (d - r < -0.001f)
				pdf = 0.0f;

			float sinThetaMax{ r / d };
			float sin2ThetaMax{ sinThetaMax * sinThetaMax };
			float cosThetaMax{ cuda::std::fmaxf(0.0f, cuda::std::sqrtf(1.0f - sin2ThetaMax)) };
			float oneMinusCosThetaMax{ 1.0f - cosThetaMax };

			float cosTheta{ (cosThetaMax - 1.0f) * uv.x + 1.0f };
			float sin2Theta{ 1.0f - cosTheta * cosTheta };
			if (sin2ThetaMax < 0.00068523f)
			{
				sin2Theta = sin2ThetaMax * uv.x;
				cosTheta = std::sqrt(1.0f - sin2Theta);
				oneMinusCosThetaMax = sin2ThetaMax / 2.0f;
			}

			float cosAlpha{ sin2Theta / sinThetaMax + cosTheta * cuda::std::fmaxf(0.0f, cuda::std::sqrtf(1.0f - sin2Theta / (sinThetaMax * sinThetaMax))) };
			float sinAlpha{ cuda::std::fmaxf(0.0f, cuda::std::sqrtf(1.0f - (cosAlpha * cosAlpha))) };

			float phi{ uv.y * 2.0f * glm::pi<float>() };
			glm::vec3 w{ sinAlpha * cuda::std::cosf(phi), sinAlpha * cuda::std::sinf(phi), cosAlpha };

			glm::vec3 u{}, v{};
			utility::createOrthonormalBasis(cToO, u, v);

			glm::vec3 sample{ cToO * w.z + u * w.x + v * w.y };
			sample = r * sample + c;

			pdf = 1.0f / (2.0f * glm::pi<float>() * oneMinusCosThetaMax);

			return sample;
		}
		CU_DEVICE CU_INLINE float pdfUniformSolidAngle(const glm::vec3& o, const glm::vec3& c, float r)
		{
			glm::vec3 l{ c - o };
			float r2{ r * r };
			float d2{ l.x * l.x + l.y * l.y + l.z * l.z };
			// if (d2 - r2 < -0.001f)
			// 	return 0.0f;

			float sin2ThetaMax{ r2 / d2 };
			float cosThetaMax{ cuda::std::fmaxf(0.0f, cuda::std::sqrtf(1.0f - sin2ThetaMax)) };
			float oneMinusCosThetaMax{ 1.0f - cosThetaMax };
			if (sin2ThetaMax < 0.00068523f)
				oneMinusCosThetaMax = sin2ThetaMax / 2.0f;

			return 1.0f / (2.0f * glm::pi<float>() * oneMinusCosThetaMax);
		}
	}
}
