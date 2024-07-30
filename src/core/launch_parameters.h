#pragma once

#include <optix_types.h>
#include <cuda/std/cstdint>

#include <glm/vec3.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/ext/quaternion_float.hpp>

#include "../core/util.h"
#include "../core/material.h"
#include "../core/light.h"
#ifdef __CUDACC__
#include "../device/spectral.h"
#endif // __CUDACC__

struct LaunchParameters
{
	struct ResolutionState
	{
		uint32_t filmWidth{};
		uint32_t filmHeight{};
		float invFilmWidth{};
		float invFilmHeight{};
		float camPerspectiveScaleW{};
		float camPerspectiveScaleH{};
	} resolutionState{};
	uint32_t maxPathLength{};
	struct SamplingState
	{
		uint32_t offset{};
		uint32_t count{};
	} samplingState{};

	CUPTR(glm::dvec4) renderData{};

	struct CameraState
	{
		glm::vec3 camU{};
		glm::vec3 camV{};
		glm::vec3 camW{};

		bool depthOfFieldEnabled{};
		float appertureSize{};
		float focusDistance{};
	} cameraState;

	struct Lights
	{
		float lightCount{};
		CUPTR(uint16_t) orderedCount{};
		CUPTR(DiskLightData) disks{};
		CUPTR(SphereLightData) spheres{};
	} lights;

	CUPTR(MaterialData) materials{};
	CUPTR(DenseSpectrum) spectra{};

	CUPTR(DenseSpectrum) sensorSpectralCurveA{};
	CUPTR(DenseSpectrum) sensorSpectralCurveB{};
	CUPTR(DenseSpectrum) sensorSpectralCurveC{};

	OptixTraversableHandle traversable{};
};
