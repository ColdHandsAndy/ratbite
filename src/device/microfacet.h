#pragma once

#include <cuda/std/cmath>
#include <cuda/std/complex>
#include <cuda/std/cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/common.hpp>
#include "glm/geometric.hpp"
#include <glm/gtc/constants.hpp>

#include "../core/util_macros.h"
#include "../device/sampling.h"
#include "../device/spectral.h"
#include "../device/local_transform.h"

namespace microfacet
{
	struct ContextIncident
	{
		float wiTan2Theta{};
		float wiCosPhi{};
		float wiSinPhi{};
	};
	CU_DEVICE CU_INLINE ContextIncident createContextIncident(const glm::vec3& wi)
	{
		return {.wiTan2Theta = LocalTransform::tan2Theta(wi), .wiCosPhi = LocalTransform::cosPhi(wi), .wiSinPhi = LocalTransform::sinPhi(wi)};
	}
	struct ContextOutgoing
	{
		float woTan2Theta{};
		float woCosPhi{};
		float woSinPhi{};
	};
	CU_DEVICE CU_INLINE ContextOutgoing createContextOutgoing(const glm::vec3& wo)
	{
		return {.woTan2Theta = LocalTransform::tan2Theta(wo), .woCosPhi = LocalTransform::cosPhi(wo), .woSinPhi = LocalTransform::sinPhi(wo)};
	}
	struct ContextMicronormal
	{
		float wmCos2Theta{};
		float wmTan2Theta{};
		float wmCosPhi{};
		float wmSinPhi{};
	};
	CU_DEVICE CU_INLINE ContextMicronormal createContextMicronormal(const glm::vec3& wm)
	{
		return {.wmCos2Theta = LocalTransform::cos2Theta(wm), .wmTan2Theta = LocalTransform::tan2Theta(wm), .wmCosPhi = LocalTransform::cosPhi(wm), .wmSinPhi = LocalTransform::sinPhi(wm)};
	}
	struct Microsurface
	{
		float alphaX{};
		float alphaY{};
	};

	CU_DEVICE CU_INLINE float LambdaG(const glm::vec3& w, float alphaX, float alphaY, float cosPhi, float sinPhi, float tan2Theta)
	{
		if (isinf(tan2Theta))
			return 0.0f;
		float aXCos{ cosPhi * alphaX };
		float aYSin{ sinPhi * alphaY };
		float alpha2{ aXCos * aXCos + aYSin * aYSin };
		return (cuda::std::sqrtf(1.0f + alpha2 * tan2Theta) - 1.0f) / 2.0f;
	}
	//Masking function
	CU_DEVICE CU_INLINE float G1(const glm::vec3& wo, const ContextOutgoing& context, const Microsurface& ms)
	{
		return 1.0f / (1.0f +
			LambdaG(wo, ms.alphaX, ms.alphaY, context.woCosPhi, context.woSinPhi, context.woTan2Theta));
	}
	//Masking-Shadowing function
	CU_DEVICE CU_INLINE float G(const glm::vec3& wi, const glm::vec3& wo, const ContextIncident& ctxi, const ContextOutgoing& ctxo, const Microsurface& ms)
	{
		return 1.0f / (1.0f + 
			LambdaG(wi, ms.alphaX, ms.alphaY, ctxi.wiCosPhi, ctxi.wiSinPhi, ctxi.wiTan2Theta) + 
			LambdaG(wo, ms.alphaX, ms.alphaY, ctxo.woCosPhi, ctxo.woSinPhi, ctxo.woTan2Theta));
	}
	//Microfacet distribution function
	CU_DEVICE CU_INLINE float D(const glm::vec3& wm, const ContextMicronormal& ctxm, const Microsurface& ms)
	{
		float tan2Theta{ ctxm.wmTan2Theta };
		if (isinf(tan2Theta))
			return 0.0f;
		float cos4Theta{ ctxm.wmCos2Theta * ctxm.wmCos2Theta };
		if (cos4Theta < 1e-16f)
			return 0.0f;
		float cosPhiByAX{ ctxm.wmCosPhi / ms.alphaX };
		float sinPhiByAY{ ctxm.wmSinPhi / ms.alphaY };
		float e{ tan2Theta * (cosPhiByAX * cosPhiByAX + sinPhiByAY * sinPhiByAY) };
		float oPe{ 1.0f + e };
		return 1.0f / (glm::pi<float>() * ms.alphaX * ms.alphaY * cos4Theta * oPe * oPe);
	}
	//Fresnel function for conductors
	CU_DEVICE CU_INLINE SampledSpectrum FComplex(float mfCosTheta, const SampledSpectrum& eta, const SampledSpectrum& k)
	{
		using Complex = cuda::std::complex<float>;
		auto complexFresnel{ [](float cosTheta, const Complex& ceta)
			{
				cosTheta = glm::clamp(cosTheta, 0.0f, 1.0f);

				float sin2Theta{ 1.0f - cosTheta * cosTheta };
				Complex sin2ThetaT{ sin2Theta / (ceta * ceta) };
				Complex cosThetaT{ cuda::std::sqrt(1.0f - sin2ThetaT) };

				Complex rParl{ (ceta * cosTheta - cosThetaT) / (ceta * cosTheta + cosThetaT) };
				Complex rPerp{ (cosTheta - ceta * cosThetaT) / (cosTheta + ceta * cosThetaT) };
				return (cuda::std::norm(rParl) + cuda::std::norm(rPerp)) / 2.0f;
			} };
		SampledSpectrum res{};
		for (int i{ 0 }; i < SampledSpectrum::getSampleCount(); ++i)
			res[i] = complexFresnel(mfCosTheta, Complex{eta[i], k[i]});
		return res;
	}
	//Fresnel function for dielectrics
	CU_DEVICE CU_INLINE float FReal(float mfCosTheta, float eta)
	{
		mfCosTheta = glm::clamp(mfCosTheta, -1.0f, 1.0f);

		if (mfCosTheta < 0.0f)
		{
			eta = 1.0f / eta;
			mfCosTheta = -mfCosTheta;
		}

		float mfSin2Theta{ 1.0f - mfCosTheta * mfCosTheta };
		float mfSin2ThetaT{ mfSin2Theta / (eta * eta) };
		if (mfSin2ThetaT >= 1.0f)
			return 1.0f;
		float mfCosThetaT{ cuda::std::sqrtf(glm::max(0.0f, 1.0f - mfSin2ThetaT)) };

		float rParl{ (eta * mfCosTheta - mfCosThetaT) / (eta * mfCosTheta + mfCosThetaT) };
		float rPerp{ (mfCosTheta - eta * mfCosThetaT) / (mfCosTheta + eta * mfCosThetaT) };
		return (rParl * rParl + rPerp * rPerp) / 2.0f;
	}

	namespace VNDF
	{
		CU_DEVICE CU_INLINE glm::vec3 sample(const glm::vec3& wo, const ContextOutgoing& ctxo, const Microsurface& ms, const glm::vec2& uv)
		{
			glm::vec3 wh{ glm::normalize(glm::vec3{ms.alphaX * wo.x, ms.alphaY * wo.y, wo.z}) };
			if (wh.z < 0)
				wh = -wh;

			glm::vec3 t{ (wh.z < 0.99999f) ? glm::normalize(glm::cross(glm::vec3{0.0f, 0.0f, 1.0f}, wh)) : glm::vec3{1.0f, 0.0f, 0.0f} };
			glm::vec3 b{ glm::cross(wh, t) };

			glm::vec2 p{ sampling::disk::sampleUniform2DPolar(uv) };

			float h{ cuda::std::sqrtf(1.0f - p.x * p.x) };
			p.y = glm::mix(h, p.y, (1.0f + wh.z) / 2.0f);

			float pz{ cuda::std::sqrtf(cuda::std::fmax(0.0f, 1.0f - (p.x * p.x + p.y * p.y))) };
			glm::vec3 nh{ p.x * t + p.y * b + pz * wh };

			return glm::normalize(glm::vec3{ms.alphaX * nh.x, ms.alphaY * nh.y, cuda::std::fmax(1e-6f, nh.z)});
		}
		CU_DEVICE CU_INLINE float PDF(const glm::vec3& wo, const glm::vec3& wm, const ContextOutgoing& ctxo, const ContextMicronormal& ctxm, const Microsurface& ms)
		{
			return G1(wo, ctxo, ms) / cuda::std::fabs(LocalTransform::cosTheta(wo)) * D(wm, ctxm, ms) * cuda::std::fabs(glm::dot(wo, wm));
		}
	}
}
