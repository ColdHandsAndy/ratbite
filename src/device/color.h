#pragma once

#include <glm/vec3.hpp>

#include "../core/util.h"
#include "spectral.h"

namespace color
{
	CU_DEVICE CU_INLINE glm::dvec3 toRGB(const DenseSpectrum& sensorSpectralCurveA, 
		const DenseSpectrum& sensorSpectralCurveB, 
		const DenseSpectrum& sensorSpectralCurveC, 
		const SampledWavelengths& lambda, const SampledSpectrum& L)
	{
		return glm::dvec3{ (sensorSpectralCurveA.sample(lambda) * L).average(), (sensorSpectralCurveB.sample(lambda) * L).average(), (sensorSpectralCurveC.sample(lambda) * L).average() };
	}
}
