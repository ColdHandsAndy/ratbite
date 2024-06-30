#pragma once

#include <optix_types.h>
#include <cuda/std/cstdint>

#include <glm/glm.hpp>
#include <glm/ext/quaternion_common.hpp>

#include "../core/util_macros.h"
#include "material.h"
#include "../device/spectral.h"

struct LaunchParameters
{
	uint32_t filmWidth{};
	uint32_t filmHeight{};
	float invFilmWidth{};
	float invFilmHeight{};
	uint32_t maxPathDepth{};
	struct SamplingState
	{
		uint32_t offset{};
		uint32_t count{};
	} samplingState{};

	CUPTR(glm::dvec4) renderingData{};

	glm::vec3 camU{};
	glm::vec3 camV{};
	glm::vec3 camW{};
	float camPerspectiveScaleW{};
	float camPerspectiveScaleH{};

	uint32_t illuminantSpectralDistributionIndex{};
	glm::vec3 diskLightPosition{};
	float diskLightRadius{};
	glm::quat diskFrame{};
	glm::vec3 diskNormal{};
	float diskArea{};
	float lightScale{};
	float diskSurfacePDF{};

	CUPTR(MaterialData) materials{};
	CUPTR(DenseSpectrum) spectrums{};

	CUPTR(DenseSpectrum) sensorSpectralCurveA{};
	CUPTR(DenseSpectrum) sensorSpectralCurveB{};
	CUPTR(DenseSpectrum) sensorSpectralCurveC{};

	OptixTraversableHandle traversable{};
};
