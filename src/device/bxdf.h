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
	struct Context
	{
		glm::vec3 wi{};
		glm::vec3 wo{};
		glm::vec3 wm{};
		float mfSamplePDF{};

		float alphaX{};
		float alphaY{};
		
		float wmCos2Theta{};
		float wmTan2Theta{};
		float wmCosPhi{};
		float wmSinPhi{};

		float wiTan2Theta{};
		float wiCosPhi{};
		float wiSinPhi{};
		
		float woTan2Theta{};
		float woCosPhi{};
		float woSinPhi{};
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
	CU_DEVICE CU_INLINE float G1(const Context& context)
	{
		return 1.0f / (1.0f +
			LambdaG(context.wo, context.alphaX, context.alphaY, context.woCosPhi, context.woSinPhi, context.woTan2Theta));
	}
	//Masking-Shadowing function
	CU_DEVICE CU_INLINE float G(const Context& context)
	{
		return 1.0f / (1.0f + 
			LambdaG(context.wi, context.alphaX, context.alphaY, context.wiCosPhi, context.wiSinPhi, context.wiTan2Theta) + 
			LambdaG(context.wo, context.alphaX, context.alphaY, context.woCosPhi, context.woSinPhi, context.woTan2Theta));
	}
	//Microfacet distribution function
	CU_DEVICE CU_INLINE float D(const Context& context)
	{
		float tan2Theta{ context.wmTan2Theta };
		if (isinf(tan2Theta))
			return 0.0f;
		float cos4Theta{ context.wmCos2Theta * context.wmCos2Theta };
		if (cos4Theta < 1e-16f)
			return 0.0f;
		float cosPhiByAX{ context.wmCosPhi / context.alphaX };
		float sinPhiByAY{ context.wmSinPhi / context.alphaY };
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
		float mfCosThetaT{ cuda::std::sqrtf(glm::max(0.0f, 1.0f - mfSin2ThetaT)) };

		float rParl{ (eta * mfCosTheta - mfCosThetaT) / (eta * mfCosTheta + mfCosThetaT) };
		float rPerp{ (mfCosTheta - eta * mfCosThetaT) / (mfCosTheta + eta * mfCosThetaT) };
		return (rParl * rParl + rPerp * rPerp) / 2.0f;
	}

	namespace VNDF
	{
		CU_DEVICE CU_INLINE void sample(Context& context, const glm::vec2& uv)
		{
			glm::vec3 wh{ glm::normalize(glm::vec3{context.alphaX * context.wo.x, context.alphaY * context.wo.y, context.wo.z}) };
			if (wh.z < 0)
				wh = -wh;

			glm::vec3 t{ (wh.z < 0.99999f) ? glm::normalize(glm::cross(glm::vec3{0.0f, 0.0f, 1.0f}, wh)) : glm::vec3{1.0f, 0.0f, 0.0f} };
			glm::vec3 b{ glm::cross(wh, t) };

			glm::vec2 p{ sampling::disk::sampleUniform2DPolar(uv) };

			float h{ cuda::std::sqrtf(1.0f - p.x * p.x) };
			p.y = glm::mix(h, p.y, (1.0f + wh.z) / 2.0f);

			float pz{ cuda::std::sqrtf(cuda::std::fmax(0.0f, 1.0f - (p.x * p.x + p.y * p.y))) };
			glm::vec3 nh{ p.x * t + p.y * b + pz * wh };

			context.wm = glm::normalize(glm::vec3{context.alphaX * nh.x, context.alphaY * nh.y, cuda::std::fmax(1e-6f, nh.z)});
		}
		CU_DEVICE CU_INLINE void PDF(Context& context)
		{
			const glm::vec3& w{ context.wo };
			const glm::vec3& wm{ context.wm };

			context.mfSamplePDF = G1(context) / cuda::std::fabs(LocalTransform::cosTheta(w)) * D(context) * cuda::std::fabs(glm::dot(w, wm));
		}
	}
}
