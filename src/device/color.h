#pragma once

#include <glm/vec3.hpp>

#include "../core/util.h"
#include "../device/spectral.h"

namespace color
{
	CU_DEVICE CU_INLINE glm::dvec3 toRGB(const DenseSpectrum& sensorSpectralCurveA, 
		const DenseSpectrum& sensorSpectralCurveB, 
		const DenseSpectrum& sensorSpectralCurveC, 
		const SampledWavelengths& lambda, const SampledSpectrum& L)
	{
		return glm::dvec3{ (sensorSpectralCurveA.sample(lambda) * L).average(), (sensorSpectralCurveB.sample(lambda) * L).average(), (sensorSpectralCurveC.sample(lambda) * L).average() };
	}
	
	CU_DEVICE CU_INLINE glm::vec3 sRGBtoLinearRGB(const glm::vec3& sRGB)
	{
		return {sRGB.x < 0.04045f ? sRGB.x / 12.92f : cuda::std::pow((sRGB.x + 0.055f) / 1.055f, 2.4f),
				sRGB.y < 0.04045f ? sRGB.y / 12.92f : cuda::std::pow((sRGB.y + 0.055f) / 1.055f, 2.4f),
				sRGB.z < 0.04045f ? sRGB.z / 12.92f : cuda::std::pow((sRGB.z + 0.055f) / 1.055f, 2.4f)};
	}
	CU_DEVICE CU_INLINE glm::vec3 sRGBfromLinearRGB(const glm::vec3& linearRGB)
	{
		return {linearRGB.x < 0.0031308f ? linearRGB.x * 12.92f : cuda::std::pow(linearRGB.x, 1.0f / 2.4f) * 1.055f - 0.055f,
				linearRGB.y < 0.0031308f ? linearRGB.y * 12.92f : cuda::std::pow(linearRGB.y, 1.0f / 2.4f) * 1.055f - 0.055f,
				linearRGB.z < 0.0031308f ? linearRGB.z * 12.92f : cuda::std::pow(linearRGB.z, 1.0f / 2.4f) * 1.055f - 0.055f};
	}

	CU_DEVICE CU_INLINE SampledSpectrum RGBtoSpectrum(const glm::vec3& rgb, const SampledWavelengths& wavelengths,
			const DenseSpectrum& spectralBasisR, const DenseSpectrum& spectralBasisG, const DenseSpectrum& spectralBasisB)
	{
		SampledSpectrum spectrum;
		SampledSpectrum lambda{ wavelengths.getLambda() };
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			spectrum[i] = (spectralBasisR.sample(lambda[i]) * rgb.r
						 + spectralBasisG.sample(lambda[i]) * rgb.g
						 + spectralBasisB.sample(lambda[i]) * rgb.b);
		}
		return spectrum;
	}
}
