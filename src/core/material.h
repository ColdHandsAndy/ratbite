#pragma once

#include <cuda/std/cstdint>

struct MaterialData
{
	uint32_t bxdfIndexSBT{};

	uint16_t indexOfRefractSpectrumDataIndex{};
	uint16_t absorpCoefSpectrumDataIndex{};
	uint16_t emissionSpectrumDataIndex{};
	
	float mfRoughnessValue{};

	//Emission - spectrum, scale
	//Index of refraction - spectrum
	//Absorption coefficient(metals) - spectrum
	//Roughness - value(texture)
	//Other...
};
