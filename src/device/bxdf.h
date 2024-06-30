#pragma once

#include <cuda/std/cstdint>
#include <cuda/std/complex>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "../core/util_macros.h"
#include "spectral.h"

namespace microfacet
{
	//
	CU_DEVICE CU_INLINE static float localCos2Theta(const glm::vec3& w) { return w.z * w.z; };
	CU_DEVICE CU_INLINE static float localSin2Theta(const glm::vec3& w) { return glm::max(0.0f, 1.0f - localCos2Theta(w)); };
	CU_DEVICE CU_INLINE static float localTan2Theta(const glm::vec3& w) { return localSin2Theta(w) / localCos2Theta(w); };
	CU_DEVICE CU_INLINE static float localCosTheta(const glm::vec3& w) { return w.z; };
	CU_DEVICE CU_INLINE static float localSinTheta(const glm::vec3& w) { return glm::sqrt(glm::max(0.0f, 1.0f - localCos2Theta(w))); };
	CU_DEVICE CU_INLINE static float localTanTheta(const glm::vec3& w) { return localSinTheta(w) / localCosTheta(w); };
	CU_DEVICE CU_INLINE static float localCosPhi(const glm::vec3& w) { float sinTheta{ localSinTheta(w) }; return (sinTheta == 0.0f) ? 1.0f : glm::clamp(w.x / sinTheta, -1.0f, 1.0f); };
	CU_DEVICE CU_INLINE static float localSinPhi(const glm::vec3& w) { float sinTheta{ localSinTheta(w) }; return (sinTheta == 0.0f) ? 0.0f : glm::clamp(w.y / sinTheta, -1.0f, 1.0f); };

	CU_DEVICE CU_INLINE static float localSin2Theta(float cos2Theta) { return glm::max(0.0f, 1.0f - cos2Theta); };
	CU_DEVICE CU_INLINE static float localTan2Theta(float cos2Theta, float sin2Theta) { return sin2Theta / cos2Theta; };
	CU_DEVICE CU_INLINE static float localSinTheta(float cos2Theta) { return glm::sqrt(glm::max(0.0f, 1.0f - cos2Theta)); };
	CU_DEVICE CU_INLINE static float localTanTheta(float cosTheta, float sinTheta) { return sinTheta / cosTheta; };
	CU_DEVICE CU_INLINE static float localCosPhi(const glm::vec3& w, float sinTheta) { return (sinTheta == 0.0f) ? 1.0f : glm::clamp(w.x / sinTheta, -1.0f, 1.0f); };
	CU_DEVICE CU_INLINE static float localSinPhi(const glm::vec3& w, float sinTheta) { return (sinTheta == 0.0f) ? 0.0f : glm::clamp(w.y / sinTheta, -1.0f, 1.0f); };
	//
	
	namespace distribution
	{
		CU_DEVICE CU_INLINE glm::vec3 sample(const glm::vec2& uv)
		{

		}
		CU_DEVICE CU_INLINE float pdf(const glm::vec3& wo, const glm::vec3& wm)
		{

		}
	}

	struct Context
	{
		glm::vec3 wm{};
		glm::vec3 wi{};
		glm::vec3 wo{};
		float mfSamplePDF{};
		float alphaX{};
		float alphaY{};
		//float localCosTheta{};
		//float localSinTheta{};
		//float localTanTheta{};
		float localCos2Theta{};
		//float localSin2Theta{};
		float localTan2Theta{};
		float localSinPhi{};
		float localCosPhi{};
	};

	Context createContext()
	{

	}

	CU_DEVICE CU_INLINE float LambdaG(const glm::vec3& w, float alphaX, float alphaY, float cosPhi, float sinPhi, float tan2Theta)
	{
		if (isinf(tan2Theta))
			return 0.0f;
		float aXCos{ cosPhi * alphaX };
		float aYSin{ sinPhi * alphaY };
		float alpha2{ aXCos * aXCos + aYSin * aYSin };
		return (glm::sqrt(1.0f + alpha2 * tan2Theta) - 1.0f) / 2.0f;
	}

	//Masking function
	CU_DEVICE CU_INLINE float G1(const Context& context)
	{
		return 1.0f / (1.0f +
			LambdaG(context.wo, context.alphaX, context.alphaY, context.localCosPhi, context.localSinPhi, context.localTan2Theta));
	}
	//Masking-Shadowing function
	CU_DEVICE CU_INLINE float G(const Context& context)
	{
		return 1.0f / (1.0f + 
			LambdaG(context.wi, context.alphaX, context.alphaY, context.localCosPhi, context.localSinPhi, context.localTan2Theta) + 
			LambdaG(context.wo, context.alphaX, context.alphaY, context.localCosPhi, context.localSinPhi, context.localTan2Theta));
	}
	//Microfacet distribution function
	CU_DEVICE CU_INLINE float D(const Context& context)
	{
		float tan2Theta{ context.localTan2Theta };
		if (isinf(tan2Theta))
			return 0.0f;
		float cos4Theta{ context.localCos2Theta * context.localCos2Theta };
		if (cos4Theta < 1e-16f)
			return 0.0f;
		float cosPhiByAX{ context.localCosPhi / context.alphaX };
		float sinPhiByAY{ context.localSinPhi / context.alphaY };
		float e{ tan2Theta * (cosPhiByAX * cosPhiByAX + sinPhiByAY * sinPhiByAY) };
		float oPe{ 1.0f + e };
		return 1.0f / (glm::pi<float>() * context.alphaX * context.alphaY * cos4Theta * oPe * oPe);
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
		float mfCosThetaT{ glm::sqrt(glm::max(0.0f, 1.0f - mfSin2ThetaT)) };

		float rParl{ (eta * mfCosTheta - mfCosThetaT) / (eta * mfCosTheta + mfCosThetaT) };
		float rPerp{ (mfCosTheta - eta * mfCosThetaT) / (mfCosTheta + eta * mfCosThetaT) };
		return (rParl * rParl + rPerp * rPerp) / 2.0f;
	}
}