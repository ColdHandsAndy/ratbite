#pragma once

#include <cuda/std/cstdint>
#include <cuda.h>
#include <texture_types.h>

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
		NONE        = 0,
		NORMAL      = 1 << 0,
		FRAME       = 1 << 1,
		COLOR       = 1 << 2,
		TEX_COORD_1 = 1 << 3,
		TEX_COORD_2 = 1 << 4,

		DESC
	};
	enum class TextureTypeBitfield : uint32_t
	{
		NONE          = 0,
		BASE_COLOR    = 1 << 0,
		NORMAL        = 1 << 1,
		MET_ROUGH     = 1 << 2,
		TRANSMISSION  = 1 << 3,

		DESC
	};
	enum class FactorTypeBitfield : uint32_t
	{
		NONE          = 0,
		BASE_COLOR    = 1 << 0,
		METALNESS     = 1 << 1,
		ROUGHNESS     = 1 << 2,
		TRANSMISSION  = 1 << 3,
		CUTOFF        = 1 << 4,

		DESC
	};

	uint32_t bxdfIndexSBT{};

	bool doubleSided{};

	IndexType indexType{};
	CUPTR(uint8_t) indices{};
	AttributeTypeBitfield attributes{};
	uint8_t attributeStride{};
	uint8_t normalOffset{};
	uint8_t frameOffset{};
	uint8_t colorOffset{};
	uint8_t texCoord1Offset{};
	uint8_t texCoord2Offset{};
	CUPTR(uint8_t) attributeData{};

	uint16_t indexOfRefractSpectrumDataIndex{};
	uint16_t absorpCoefSpectrumDataIndex{};
	uint16_t emissionSpectrumDataIndex{};
	
	float mfRoughnessValue{};
	float ior{};
	float alphaCutoff{};

	TextureTypeBitfield textures{};
	bool bcTexCoordSetIndex{};
	bool mrTexCoordSetIndex{};
	bool nmTexCoordSetIndex{};
	bool trTexCoordSetIndex{};
	cudaTextureObject_t baseColorTexture{};
	cudaTextureObject_t normalTexture{};
	cudaTextureObject_t pbrMetalRoughnessTexture{};
	cudaTextureObject_t transmissionTexture{};

	FactorTypeBitfield factors{};
	float baseColorFactor[4]{};
	float metalnessFactor{};
	float roughnessFactor{};
	float transmissionFactor{};
};
ENABLE_ENUM_BITWISE_OPERATORS(MaterialData::AttributeTypeBitfield);
ENABLE_ENUM_BITWISE_OPERATORS(MaterialData::TextureTypeBitfield);
ENABLE_ENUM_BITWISE_OPERATORS(MaterialData::FactorTypeBitfield);
