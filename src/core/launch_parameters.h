#pragma once

#include <optix_types.h>
#include <cuda/std/cstdint>

#include <glm/vec3.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/ext/quaternion_float.hpp>

#include "../core/util.h"
#include "../core/material.h"
#include "../core/light_tree_types.h"

class DenseSpectrum;

struct LaunchParameters
{
	struct ResolutionState
	{
		uint32_t filmWidth{};
		uint32_t filmHeight{};
		float invFilmWidth{};
		float invFilmHeight{};
		float perspectiveScaleW{};
		float perspectiveScaleH{};
	} resolutionState{};
	struct PathState
	{
		uint32_t maxPathDepth{};
		uint16_t maxReflectedPathDepth{};
		uint16_t maxTransmittedPathDepth{};
	} pathState{};
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
	} cameraState{};

	struct LightTree
	{
		CUPTR(::LightTree::PackedNode) nodes{};
		CUPTR(::LightTree::LightPointer) lightPointers{};
		CUPTR(DiskLightData) disks{};
		CUPTR(SphereLightData) spheres{};
		CUPTR(EmissiveTriangleLightData) triangles{};
		CUPTR(uint64_t) bitmasks[KMultiLightTypeCount]{};

		struct EnvironmentMap
		{
			bool enabled{ false };
			float width{};
			float height{};
			cudaTextureObject_t environmentTexture{};
			CUPTR(uint16_t) conditionalCDFIndices{};
			CUPTR(uint16_t) marginalCDFIndices{};
			float integral{};
		} envMap{};
	} lightTree{};


	CUPTR(MaterialData) materials{};
	CUPTR(DenseSpectrum) spectra{};

	struct LookUpTables
	{
		cudaTextureObject_t conductorAlbedo{};
		cudaTextureObject_t dielectricOuterAlbedo{};
		cudaTextureObject_t dielectricInnerAlbedo{};
		cudaTextureObject_t reflectiveDielectricOuterAlbedo{};
		cudaTextureObject_t reflectiveDielectricInnerAlbedo{};
		cudaTextureObject_t sheenLTC{};
	} LUTs{};

	CUPTR(DenseSpectrum) sensorSpectralCurveA{};
	CUPTR(DenseSpectrum) sensorSpectralCurveB{};
	CUPTR(DenseSpectrum) sensorSpectralCurveC{};

	CUPTR(DenseSpectrum) spectralBasisR{};
	CUPTR(DenseSpectrum) spectralBasisG{};
	CUPTR(DenseSpectrum) spectralBasisB{};

	OptixTraversableHandle traversable{};
};
