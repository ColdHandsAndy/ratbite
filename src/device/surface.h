#pragma once

#include <cuda/std/cmath>
#include <cuda/std/complex>
#include <cuda/std/cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

#include "../core/util.h"
#include "../device/sampling.h"
#include "../device/spectral.h"
#include "../device/local_transform.h"

namespace diffuse
{
	namespace CosWeighted
	{
		CU_DEVICE CU_INLINE glm::vec3 sample(const glm::vec2& uv)
		{
			glm::vec2 d{ sampling::disk::sampleUniform2DPolar(uv) };
			return glm::vec3{d.x, d.y, cuda::std::sqrt(cuda::std::fmax(0.0f, 1.0f - d.x * d.x - d.y * d.y))};
		}
		CU_DEVICE CU_INLINE float PDF(float cosTheta)
		{
			return cosTheta / glm::pi<float>();
		}
	}
}

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
	struct Alpha
	{
		float alphaX{ 0.09f };
		float alphaY{ 0.09f };
		CU_DEVICE CU_INLINE void regularize()
		{
			if (alphaX < 0.3f)
				alphaX = glm::clamp(2.0f * alphaX, 0.1f, 0.3f);
			if (alphaY < 0.3f)
				alphaY = glm::clamp(2.0f * alphaY, 0.1f, 0.3f);
		}
	};

	CU_DEVICE CU_INLINE float LambdaG(float alphaX, float alphaY, float cosPhi, float sinPhi, float tan2Theta)
	{
		if (isinf(tan2Theta))
			return 0.0f;
		float aXCos{ cosPhi * alphaX };
		float aYSin{ sinPhi * alphaY };
		float alpha2{ aXCos * aXCos + aYSin * aYSin };
		return (cuda::std::sqrtf(1.0f + alpha2 * tan2Theta) - 1.0f) / 2.0f;
	}
	//Masking function
	CU_DEVICE CU_INLINE float G1(const ContextOutgoing& context, const Alpha& a)
	{
		return 1.0f / (1.0f +
			LambdaG(a.alphaX, a.alphaY, context.woCosPhi, context.woSinPhi, context.woTan2Theta));
	}
	//Masking-Shadowing function
	CU_DEVICE CU_INLINE float G(const ContextIncident& ctxi, const ContextOutgoing& ctxo, const Alpha& a)
	{
		return 1.0f / (1.0f + 
			LambdaG(a.alphaX, a.alphaY, ctxi.wiCosPhi, ctxi.wiSinPhi, ctxi.wiTan2Theta) + 
			LambdaG(a.alphaX, a.alphaY, ctxo.woCosPhi, ctxo.woSinPhi, ctxo.woTan2Theta));
	}
	//Microfacet distribution function
	CU_DEVICE CU_INLINE float D(const ContextMicronormal& ctxm, const Alpha& ms)
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
	//Fresnel function for dielectrics (Spectrum version)
	CU_DEVICE CU_INLINE SampledSpectrum FReal(float mfCosTheta, SampledSpectrum eta)
	{
		mfCosTheta = glm::clamp(mfCosTheta, -1.0f, 1.0f);

		if (mfCosTheta < 0.0f)
		{
			eta = SampledSpectrum(1.0f) / eta;
			mfCosTheta = -mfCosTheta;
		}

		float mfSin2Theta{ 1.0f - mfCosTheta * mfCosTheta };
		SampledSpectrum mfSin2ThetaT{ SampledSpectrum(mfSin2Theta) / (eta * eta) };
		SampledSpectrum mfCosThetaT{};
		for (int i{ 0 }; i < SampledSpectrum::getSampleCount(); ++i)
			mfCosThetaT[i] = cuda::std::sqrtf(cuda::std::fmax(0.0f, 1.0f - mfSin2ThetaT[i]));

		SampledSpectrum rParl{ (eta * mfCosTheta - mfCosThetaT) / (eta * mfCosTheta + mfCosThetaT) };
		SampledSpectrum rPerp{ (SampledSpectrum(mfCosTheta) - eta * mfCosThetaT) / (SampledSpectrum(mfCosTheta) + eta * mfCosThetaT) };
		SampledSpectrum res{ (rParl * rParl + rPerp * rPerp) / 2.0f };
		for (int i{ 0 }; i < SampledSpectrum::getSampleCount(); ++i)
		{
			if (mfSin2ThetaT[i] >= 1.0f)
				res[i] = 1.0f;
		}
		return res;
	}
	//Fresnel (Schlick) approximation
	CU_DEVICE CU_INLINE float FSchlick(float f0, float mu)
	{
		return f0 + (1.0f - f0) * cuda::std::pow(1.0f - mu, 5.0f);
	}
	CU_DEVICE CU_INLINE glm::vec3 FSchlick(const glm::vec3& f0, float mu)
	{
		return f0 + (1.0f - f0) * cuda::std::pow(1.0f - mu, 5.0f);
	}
	//Fresnel (F82) approximation
	CU_DEVICE CU_INLINE glm::vec3 F82(const glm::vec3& f0, float mu)
	{
		constexpr float muDash{ 1.0f / 7.0f };
		constexpr float3 specularColor{ 1.0f, 1.0f, 1.0f };
		constexpr float specularWeight{ 1.0f };

		glm::vec3 f{ FSchlick(f0, mu) };
		glm::vec3 fDash{ FSchlick(f0, muDash) };

		return specularWeight * (f - ((mu * cuda::std::pow(1.0f - mu, 6.0f)) / (muDash * cuda::std::pow(1.0f - muDash, 6.0f))) * (fDash - glm::vec3{specularColor.x, specularColor.y, specularColor.z} * fDash));
	}
	CU_DEVICE CU_INLINE glm::vec3 FAvgIntegralForF82(const glm::vec3& f0)
	{
		constexpr float3 specularColor{ 1.0f, 1.0f, 1.0f };
		return 0.877105f * f0 + (0.0648148f + 0.0752755f * f0) * glm::vec3{specularColor.x, specularColor.y, specularColor.z} - 0.0171958f;
	}

	namespace VNDF
	{
		enum SamplingMethod
		{
			// "Sampling the GGX Distribution of Visible Normals" - Eric Heitz
			// https://jcgt.org/published/0007/04/01/paper.pdf
			ORIGINAL,
			// "Sampling Visible GGX Normals with Spherical Caps" - Jonathan Dupuy, Anis Benyoub.
			// https://arxiv.org/pdf/2306.05044
			SPHERICAL_CAP,
			// "Bounded VNDF Sampling for Smith–GGX Reflections" - Kenta Eto, Yusuke Tokuyoshi.
			// https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf
			BOUNDED_SPHERICAL_CAP,
			DESC
		};
		template<SamplingMethod Method>
		CU_DEVICE CU_INLINE glm::vec3 sample(const glm::vec3& wo, const Alpha& a, const glm::vec2& uv)
		{
			if constexpr (Method == ORIGINAL)
			{
				glm::vec3 wh{ glm::normalize(glm::vec3{a.alphaX * wo.x, a.alphaY * wo.y, wo.z}) };
				if (wh.z < 0.0f)
					wh = -wh;

				glm::vec3 t{ (wh.z < 0.99999f) ? glm::normalize(glm::cross(glm::vec3{0.0f, 0.0f, 1.0f}, wh)) : glm::vec3{1.0f, 0.0f, 0.0f} };
				glm::vec3 b{ glm::cross(wh, t) };

				glm::vec2 p{ sampling::disk::sampleUniform2DPolar(uv) };

				float h{ cuda::std::sqrtf(1.0f - p.x * p.x) };
				p.y = glm::mix(h, p.y, (1.0f + wh.z) / 2.0f);

				float pz{ cuda::std::sqrtf(cuda::std::fmax(0.0f, 1.0f - (p.x * p.x + p.y * p.y))) };
				glm::vec3 nh{ p.x * t + p.y * b + pz * wh };

				return glm::normalize(glm::vec3{a.alphaX * nh.x, a.alphaY * nh.y, cuda::std::fmax(1e-6f, nh.z)});
			}
			else if constexpr (Method == SPHERICAL_CAP)
			{
				glm::vec3 woStd{ glm::normalize(glm::vec3{wo.x * a.alphaX, wo.y * a.alphaY, wo.z}) };
				if (woStd.z < 0.0f)
					woStd = -woStd;

				float phi{ 2.0f * glm::pi<float>() * uv.x };
				float b{ woStd.z };
				float z{ __fmaf_rn(1.0f - uv.y, 1.0f + b, -b) };
				float sinTheta{ cuda::std::sqrtf(glm::clamp(1.0f - z * z, 0.0f, 1.0f)) };
				glm::vec3 wiStd{ sinTheta * cuda::std::cos(phi), sinTheta * cuda::std::sin(phi), z };
				glm::vec3 wmStd{ woStd + wiStd };
				glm::vec3 wm{ glm::normalize(glm::vec3{wmStd.x * a.alphaX, wmStd.y * a.alphaY, wmStd.z}) };
				return wm;
			}
			else if constexpr (Method == BOUNDED_SPHERICAL_CAP)
			{
				glm::vec3 woStd{ glm::normalize(glm::vec3{wo.x * a.alphaX, wo.y * a.alphaY, wo.z}) };

				float phi{ 2.0f * glm::pi<float>() * uv.x };
				float a1{ cuda::std::fmin(a.alphaX, a.alphaY) };
				float s{ 1.0f + glm::length(glm::vec2{wo.x, wo.y}) };
				float a2{ a1 * a1 };
				float s2{ s * s };
				float k{ (1.0f - a2) * s2 / (s2 + a2 * wo.z * wo.z) };
				float b{ wo.z > 0.0f ? k * woStd.z : woStd.z };
				float z{ __fmaf_rn(1.0f - uv.y, 1.0f + b, -b) };
				float sinTheta{ cuda::std::sqrtf(glm::clamp(1.0f - z * z, 0.0f, 1.0f)) };
				glm::vec3 wiStd{ sinTheta * cuda::std::cos(phi), sinTheta * cuda::std::sin(phi), z };
				glm::vec3 wmStd{ woStd + wiStd };
				glm::vec3 wm{ glm::normalize(glm::vec3{wmStd.x * a.alphaX, wmStd.y * a.alphaY, wmStd.z}) };
				return wm;
			}
			else
				static_assert(false);
			return {};
		}
		template<SamplingMethod Method>
		CU_DEVICE CU_INLINE float PDF(const glm::vec3& wo, const float absDotWoWm, const ContextOutgoing& ctxo, const ContextMicronormal& ctxm, const Alpha& a)
		{
			if constexpr (Method == ORIGINAL)
			{
				return G1(ctxo, a) / cuda::std::fabs(LocalTransform::cosTheta(wo)) * D(ctxm, a) * absDotWoWm;
			}
			else if constexpr (Method == SPHERICAL_CAP)
			{
				return G1(ctxo, a) / cuda::std::fabs(LocalTransform::cosTheta(wo)) * D(ctxm, a) * absDotWoWm;
			}
			else if constexpr (Method == BOUNDED_SPHERICAL_CAP)
			{
				float ndf{ D(ctxm, a) };
				glm::vec2 ao{ glm::vec2{wo.x, wo.y} * glm::vec2{a.alphaX, a.alphaY} };
				float len2{ glm::dot(ao, ao) };
				float t{ cuda::std::sqrt(len2 + wo.z * wo.z) };
				if (wo.z >= 0.0f)
				{
					float a1{ glm::clamp(cuda::std::fmin(a.alphaX, a.alphaY), 0.0f, 1.0f) };
					float s{ 1.0f + glm::length(glm::vec2{wo.x, wo.y}) };
					float a2{ a1 * a1 };
					float s2{ s * s };
					float k{ (1.0f - a2) * s2 / (s2 + a2 * wo.z * wo.z) };
					return ndf * 2.0f * absDotWoWm / (k * wo.z + t); // Jacobian is applied outside of this function, therfore different formula
				}
				return ndf * (t - wo.z) * 2.0f * absDotWoWm / len2; // Jacobian is applied outside of this function, therfore different formula
			}
			else
				static_assert(false);
			return {};
		}
	}
}

namespace microsurface
{
	struct Base
	{
		glm::vec3 color{ 0.8f };
		float metalness{ 1.0f };
	};
	struct Specular
	{
		float ior{ 1.5f };
		float roughness{ 1.0f };
	};
	struct Transmission
	{
		// glm::vec3 color{ 1.0f }; // Transmission color is base color
		float weight{ 0.0f };
	};

	struct Surface
	{
		Base base{};
		Specular specular{};
		Transmission transmission{};
	};
}
