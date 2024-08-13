#pragma once

#include <cuda/std/cstdint>
#include <cuda.h>

#include "../core/util.h"

enum class IndexType
{
	UINT_16,
	UINT_32,

	DESC
};

struct MaterialData
{
	enum class AttributeTypeBitfield : uint32_t
	{
		NONE      = 0,
		NORMAL    = 1 << 0,
		FRAME     = 1 << 1,
		COLOR     = 1 << 2,
		TEX_COORD = 1 << 3,

		DESC
	};

	uint32_t bxdfIndexSBT{};

	IndexType indexType{};
	CUPTR(void) indices{};
	AttributeTypeBitfield attributes{};
	CUPTR(uint8_t) attributeData{};

	uint16_t indexOfRefractSpectrumDataIndex{};
	uint16_t absorpCoefSpectrumDataIndex{};
	uint16_t emissionSpectrumDataIndex{};
	
	float mfRoughnessValue{};

	// Any
		// Frame -> Normal: texture
	// BxDF (Trowbridge-Reitz)
		// Roughness: value, texture
	// BxDF (Trowbridge-Reitz - Absorbing Dielectric)
		// Color: value, attribute, texture
		// IOR: value

	// Attributes (Normal and Frame are interchangeable)
		// Color: 1 U32
		// Normal: 2 U32 (Orthographic)
		// Frame: 4 U32 (Quaternion 'w' sign stores handedness)
		// TexCoords: 2 F
};
ENABLE_ENUM_BITWISE_OPERATORS(MaterialData::AttributeTypeBitfield);
