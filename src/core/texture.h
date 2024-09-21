#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "../core/debug_macros.h"

enum class TextureType
{
	R8_UNORM,
	R8G8_UNORM,
	R8G8B8A8_UNORM,
	R16_UINT,
	R16G16_UINT,
	R16G16B16A16_UINT,
	R16_UNORM,
	R16G16_UNORM,
	R16G16B16A16_UNORM,
	R16_FLOAT,
	R16G16_FLOAT,
	R16G16B16A16_FLOAT,
	R32_UINT,
	R32G32_UINT,
	R32G32B32A32_UINT,
	R32_FLOAT,
	R32G32_FLOAT,
	R32G32B32A32_FLOAT,

	DESC
};
enum class TextureAddress
{
	WRAP,
	CLAMP,
	MIRROR,

	DESC
};
enum class TextureFilter
{
	NEAREST,
	LINEAR,

	DESC
};

class CudaImage
{
private:
	cudaArray_t m_image{};

	uint32_t m_width{};
	uint32_t m_height{};
	uint32_t m_depth{};
	uint32_t m_texelByteSize{};

	TextureType m_type{};

	bool m_invalid{ true };

	void initialize(uint32_t width, uint32_t height, uint32_t depth,
			int bitNumX, int bitNumY, int bitNumZ, int bitNumW, cudaChannelFormatKind channelFormatKind)
	{
		m_width = width;
		m_height = height;
		m_depth = depth;
		m_texelByteSize = (bitNumX + bitNumY + bitNumZ + bitNumW) / 8;

		cudaChannelFormatDesc formatDesc{ cudaCreateChannelDesc(bitNumX, bitNumY, bitNumZ, bitNumW, channelFormatKind) };

		if (depth == 1)
			CUDA_CHECK(cudaMallocArray(&m_image, &formatDesc, width, height));
		else
			CUDA_CHECK(cudaMalloc3DArray(&m_image, &formatDesc, {width, height, depth}));

		m_invalid = false;
	}
	void destroy()
	{
		if (!m_invalid)
			CUDA_CHECK(cudaFreeArray(m_image));
	}
public:
	CudaImage() = default;
	CudaImage(uint32_t width, uint32_t height, uint32_t depth, TextureType type) : m_type{type}
	{
		switch (type)
		{
			case TextureType::R8_UNORM:
				initialize(width, height, depth,
						8, 0, 0, 0, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R8G8_UNORM:
				initialize(width, height, depth,
						8, 8, 0, 0, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R8G8B8A8_UNORM:
				initialize(width, height, depth,
						8, 8, 8, 8, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R16_UINT:
				initialize(width, height, depth,
						16, 0, 0, 0, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R16G16_UINT:
				initialize(width, height, depth,
						16, 16, 0, 0, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R16G16B16A16_UINT:
				initialize(width, height, depth,
						16, 16, 16, 16, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R16_UNORM:
				initialize(width, height, depth,
						16, 0, 0, 0, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R16G16_UNORM:
				initialize(width, height, depth,
						16, 16, 0, 0, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R16G16B16A16_UNORM:
				initialize(width, height, depth,
						16, 16, 16, 16, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R16_FLOAT:
				initialize(width, height, depth,
						16, 0, 0, 0, cudaChannelFormatKindFloat);
				break;
			case TextureType::R16G16_FLOAT:
				initialize(width, height, depth,
						16, 16, 0, 0, cudaChannelFormatKindFloat);
				break;
			case TextureType::R16G16B16A16_FLOAT:
				initialize(width, height, depth,
						16, 16, 16, 16, cudaChannelFormatKindFloat);
				break;
			case TextureType::R32_UINT:
				initialize(width, height, depth,
						32, 0, 0, 0, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R32G32_UINT:
				initialize(width, height, depth,
						32, 32, 0, 0, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R32G32B32A32_UINT:
				initialize(width, height, depth,
						32, 32, 32, 32, cudaChannelFormatKindUnsigned);
				break;
			case TextureType::R32_FLOAT:
				initialize(width, height, depth,
						32, 0, 0, 0, cudaChannelFormatKindFloat);
				break;
			case TextureType::R32G32_FLOAT:
				initialize(width, height, depth,
						32, 32, 0, 0, cudaChannelFormatKindFloat);
				break;
			case TextureType::R32G32B32A32_FLOAT:
				initialize(width, height, depth,
						32, 32, 32, 32, cudaChannelFormatKindFloat);
				break;
			default:
				R_ERR_LOG("Invalid texture type passed");
				break;
		}
	}
	CudaImage(CudaImage&& tex)
	{
		m_width = tex.m_width;
		m_height = tex.m_height;
		m_depth = tex.m_depth;
		m_texelByteSize = tex.m_texelByteSize;
		m_image = tex.m_image;

		m_invalid = false;
		tex.m_invalid = true;
	}
	CudaImage& operator=(CudaImage&& tex)
	{
		m_width = tex.m_width;
		m_height = tex.m_height;
		m_depth = tex.m_depth;
		m_texelByteSize = tex.m_texelByteSize;
		m_image = tex.m_image;

		m_invalid = false;
		tex.m_invalid = true;

		return *this;
	}
	~CudaImage()
	{
		destroy();
	}
	CudaImage(const CudaImage&) = delete;
	CudaImage& operator=(const CudaImage&) = delete;

	cudaArray_t getData() const { return m_image; }

	void fillZero(uint32_t wOffset, uint32_t hOffset, uint32_t dOffset, uint32_t width, uint32_t height, uint32_t depth)
	{
		void* zeroBuf{ malloc(width * m_texelByteSize * height * depth) };
		memset(zeroBuf, 0, width * m_texelByteSize * height * depth);
		fill(zeroBuf, wOffset, hOffset, dOffset, width, height, depth, cudaMemcpyHostToDevice);
		free(zeroBuf);
	}
	void fill(void* src, uint32_t wOffset, uint32_t hOffset, uint32_t dOffset, uint32_t width, uint32_t height, uint32_t depth, cudaMemcpyKind memcpyKind)
	{
		R_ASSERT_LOG(!m_invalid, "CudaImage is invalid.");
		R_ASSERT_LOG(wOffset < m_width && hOffset < m_height && dOffset < m_depth, "Offset into the texture is invalid");
		R_ASSERT_LOG(width <= m_width && height <= m_height && depth <= m_depth, "Source data size exceeds the texture's size");
		if (depth == 1)
			CUDA_CHECK(cudaMemcpy2DToArray(m_image, wOffset, hOffset, src, width * m_texelByteSize, width * m_texelByteSize, height, memcpyKind));
		else
		{
			cudaMemcpy3DParms memcpyData{ .srcPtr = cudaPitchedPtr{src, width * m_texelByteSize, width, height},
				.dstArray = m_image,
				.dstPos = cudaPos{wOffset, hOffset, dOffset},
				.extent = cudaExtent{m_width, m_height, m_depth},
				.kind = cudaMemcpyHostToDevice };
			CUDA_CHECK(cudaMemcpy3D(&memcpyData));
		}
	}

	friend class CudaTexture;
};

class CudaTexture
{
private:
	const CudaImage* m_internalImage{};
	cudaTextureObject_t m_texture{};

	bool m_invalid{ true };

	void initialize(cudaTextureAddressMode addressModeX, cudaTextureAddressMode addressModeY, cudaTextureAddressMode addressModeZ, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode,
			bool colorTexture,
			bool normalizedAccess,
			uint32_t maxAnisotropy, cudaTextureFilterMode mipmapFilterMode)
	{
		cudaResourceDesc resDesc{};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = m_internalImage->getData();

		cudaTextureDesc texDesc{};
		texDesc.addressMode[0] = addressModeX;
		texDesc.addressMode[1] = addressModeY;
		texDesc.addressMode[2] = addressModeZ;
		texDesc.filterMode = filterMode;
		texDesc.readMode = readMode;
		texDesc.sRGB = colorTexture ? 1 : 0;
		// texDesc.borderColor[0];
		// texDesc.borderColor[1];
		// texDesc.borderColor[2];
		// texDesc.borderColor[3];
		texDesc.normalizedCoords = normalizedAccess ? 1 : 0;
		texDesc.maxAnisotropy = maxAnisotropy;
		texDesc.mipmapFilterMode = mipmapFilterMode;
		// texDesc.mipmapLevelBias;
		// texDesc.minMipmapLevelClamp;
		// texDesc.maxMipmapLevelClamp;
		// texDesc.disableTrilinearOptimization;
		// texDesc.seamlessCubemap;

		CUDA_CHECK(cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, NULL));

		m_invalid = false;
	}
	void destroy()
	{
		if (!m_invalid)
			CUDA_CHECK(cudaDestroyTextureObject(m_texture));
	}
public:
	CudaTexture() = default;
	CudaTexture(const CudaImage& image,
			TextureAddress addressModeX = TextureAddress::CLAMP, TextureAddress addressModeY = TextureAddress::CLAMP, TextureAddress addressModeZ = TextureAddress::CLAMP,
			TextureFilter filterMode = TextureFilter::LINEAR,
			bool colorTexture = false) : m_internalImage{ &image }
	{
		cudaTextureAddressMode cudaAddressModeX{};
		switch (addressModeX)
		{
			case TextureAddress::WRAP:
				cudaAddressModeX = cudaAddressModeWrap;
				break;
			case TextureAddress::CLAMP:
				cudaAddressModeX = cudaAddressModeClamp;
				break;
			case TextureAddress::MIRROR:
				cudaAddressModeX = cudaAddressModeMirror;
				break;
			default:
				R_ERR_LOG("Unknown address mode passed");
				break;
		}
		cudaTextureAddressMode cudaAddressModeY{};
		switch (addressModeY)
		{
			case TextureAddress::WRAP:
				cudaAddressModeY = cudaAddressModeWrap;
				break;
			case TextureAddress::CLAMP:
				cudaAddressModeY = cudaAddressModeClamp;
				break;
			case TextureAddress::MIRROR:
				cudaAddressModeY = cudaAddressModeMirror;
				break;
			default:
				R_ERR_LOG("Unknown address mode passed");
				break;
		}
		cudaTextureAddressMode cudaAddressModeZ{};
		switch (addressModeZ)
		{
			case TextureAddress::WRAP:
				cudaAddressModeZ = cudaAddressModeWrap;
				break;
			case TextureAddress::CLAMP:
				cudaAddressModeZ = cudaAddressModeClamp;
				break;
			case TextureAddress::MIRROR:
				cudaAddressModeZ = cudaAddressModeMirror;
				break;
			default:
				R_ERR_LOG("Unknown address mode passed");
				break;
		}
		cudaTextureFilterMode cudaFilterMode{};
		switch (filterMode)
		{
			case TextureFilter::NEAREST:
				cudaFilterMode = cudaFilterModePoint;
				break;
			case TextureFilter::LINEAR:
				cudaFilterMode = cudaFilterModeLinear;
				break;
			default:
				R_ERR_LOG("Unknown filter mode passed");
				break;
		}
		switch (image.m_type)
		{
			case TextureType::R8_UNORM:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeNormalizedFloat,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R8G8_UNORM:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeNormalizedFloat,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R8G8B8A8_UNORM:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeNormalizedFloat,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R16_UINT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R16G16_UINT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R16G16B16A16_UINT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R16_UNORM:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeNormalizedFloat,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R16G16_UNORM:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeNormalizedFloat,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R16G16B16A16_UNORM:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeNormalizedFloat,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R16_FLOAT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R16G16_FLOAT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R16G16B16A16_FLOAT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R32_UINT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R32G32_UINT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R32G32B32A32_UINT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R32_FLOAT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R32G32_FLOAT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			case TextureType::R32G32B32A32_FLOAT:
				initialize(cudaAddressModeX, cudaAddressModeY, cudaAddressModeZ, cudaFilterMode, cudaReadModeElementType,
						colorTexture,
						true,
						16, cudaFilterModeLinear);
				break;
			default:
				R_ERR_LOG("Invalid texture type passed");
				break;
		}
	}
	CudaTexture(CudaTexture&& tex)
	{
		m_internalImage = tex.m_internalImage;
		m_texture = tex.m_texture;

		m_invalid = false;
		tex.m_invalid = true;
	}
	CudaTexture& operator=(CudaTexture&& tex)
	{
		m_internalImage = tex.m_internalImage;
		m_texture = tex.m_texture;
	
		m_invalid = false;
		tex.m_invalid = true;

		return *this;
	}
	~CudaTexture()
	{
		destroy();
	}
	CudaTexture(const CudaTexture&) = delete;
	CudaTexture& operator=(const CudaTexture&) = delete;

	cudaTextureObject_t getTextureObject() const { return m_texture; }
};

struct CudaCombinedTexture
{
	CudaImage image{};
	CudaTexture texture{};

	CudaCombinedTexture() = default;
	CudaCombinedTexture(uint32_t width, uint32_t height, uint32_t depth, TextureType type,
			TextureAddress addressModeX = TextureAddress::CLAMP, TextureAddress addressModeY = TextureAddress::CLAMP, TextureAddress addressModeZ = TextureAddress::CLAMP,
			TextureFilter filterMode = TextureFilter::LINEAR,
			bool colorTexture = false)
		: image{width, height, depth, type}, texture{image,
			addressModeX, addressModeY, addressModeZ,
			filterMode,
			colorTexture}
	{}
	CudaCombinedTexture(CudaCombinedTexture&&) = default;
	CudaCombinedTexture& operator=(CudaCombinedTexture&&) = default;
	CudaCombinedTexture(const CudaCombinedTexture&) = delete;
	CudaCombinedTexture& operator=(const CudaCombinedTexture&) = delete;
};
