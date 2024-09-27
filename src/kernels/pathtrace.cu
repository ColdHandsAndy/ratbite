#include <cuda_runtime.h>
#include <optix_device.h>
#include <cuda/std/cstdint>
#include <cuda/std/cmath>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/launch_parameters.h"
#include "../core/util.h"
#include "../core/material.h"
#include "../core/light.h"
#include "../device/util.h"
#include "../device/dir_gen.h"
#include "../device/local_transform.h"
#include "../device/filter.h"
#include "../device/color.h"
#include "../device/quasi_random.h"
#include "../device/spectral.h"
#include "../device/sampling.h"
#include "../device/surface.h"
#include "../device/mis.h"

enum class PathStateBitfield : uint32_t
{
	NO_FLAGS              = 0,
	PREVIOUS_HIT_SPECULAR = 1 << 0,
	CURRENT_HIT_SPECULAR  = 1 << 1,
	REGULARIZED           = 1 << 2,
	RAY_REFRACTED         = 1 << 3,
	PATH_TERMINATED       = 1 << 4,
	TRIANGULAR_GEOMETRY   = 1 << 5,
	RIGHT_HANDED_FRAME    = 1 << 6,
	REFRACTION_HAPPENED   = 1 << 7,
};
ENABLE_ENUM_BITWISE_OPERATORS(PathStateBitfield);

extern "C"
{
	__constant__ LaunchParameters parameters{};
}

struct DirectLightSampleData
{
	SampledSpectrum spectrumSample{};
	glm::vec3 lightDir{};
	float lightSamplePDF{};
	bool occluded{};
};

CU_DEVICE CU_INLINE void unpackTraceData(const LaunchParameters& params, glm::vec3& hP, uint32_t& encHGN,
		PathStateBitfield& stateFlags,
		uint32_t& primitiveIndex, glm::vec2& barycentrics,
		LightType& lightType, uint16_t& lightIndex,
		MaterialData** materialDataPtr,
		glm::mat3& normalTransform,
		uint32_t* pl)
{
	hP = glm::vec3{ __uint_as_float(pl[0]), __uint_as_float(pl[1]), __uint_as_float(pl[2]) };
	encHGN = pl[3];
	uint32_t matIndex{ pl[7] & 0xFFFF };
	*materialDataPtr = params.materials + matIndex;
	stateFlags |= static_cast<PathStateBitfield>((pl[7] >> 16) & 0xFF);
	lightType = static_cast<LightType>(pl[7] >> 24);
	if (static_cast<bool>(stateFlags & PathStateBitfield::TRIANGULAR_GEOMETRY))
	{
		primitiveIndex = pl[4];
		barycentrics = {__uint_as_float(pl[5]), __uint_as_float(pl[6])};
	}
	else
		lightIndex = pl[4];
	normalTransform[0][0] = __uint_as_float(pl[8]);
	normalTransform[1][0] = __uint_as_float(pl[9]);
	normalTransform[2][0] = __uint_as_float(pl[10]);
	normalTransform[0][1] = __uint_as_float(pl[11]);
	normalTransform[1][1] = __uint_as_float(pl[12]);
	normalTransform[2][1] = __uint_as_float(pl[13]);
	normalTransform[0][2] = __uint_as_float(pl[14]);
	normalTransform[1][2] = __uint_as_float(pl[15]);
	normalTransform[2][2] = __uint_as_float(pl[16]);
}
CU_DEVICE CU_INLINE void updateStateFlags(PathStateBitfield& stateFlags)
{
	PathStateBitfield excludeFlags{ PathStateBitfield::CURRENT_HIT_SPECULAR | PathStateBitfield::TRIANGULAR_GEOMETRY | PathStateBitfield::RIGHT_HANDED_FRAME | PathStateBitfield::RAY_REFRACTED };
	PathStateBitfield includeFlags{ static_cast<bool>(stateFlags & PathStateBitfield::CURRENT_HIT_SPECULAR) ?
		PathStateBitfield::PREVIOUS_HIT_SPECULAR : PathStateBitfield::REGULARIZED };
	if (static_cast<bool>(stateFlags & (PathStateBitfield::RAY_REFRACTED | PathStateBitfield::REFRACTION_HAPPENED)))
		includeFlags |= PathStateBitfield::REFRACTION_HAPPENED;

	stateFlags = (stateFlags & (~excludeFlags)) | includeFlags;
}
CU_DEVICE CU_INLINE void resolveSample(SampledSpectrum& L, const SampledSpectrum& pdf)
{
	for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
	{
		L[i] = pdf[i] != 0.0f ? L[i] / pdf[i] : 0.0f;
	}
}
CU_DEVICE CU_INLINE void resolveAttributes(const MaterialData& material, const PathStateBitfield& stateFlags, const glm::vec2& barycentrics, uint32_t primitiveIndex,
		const glm::vec3& geoNormal, const glm::mat3& normalTransform,
		glm::vec2& texCoord1, glm::vec2& texCoord2, glm::vec3& normal, glm::vec3& tangent, bool& flipBitangent)
{
	float baryWeights[3]{ 1.0f - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y };
	uint32_t indices[3];
	switch (material.indexType)
	{
		case IndexType::UINT_16:
			{
				uint16_t* idata{ reinterpret_cast<uint16_t*>(material.indices) + primitiveIndex * 3 };
				indices[0] = idata[0];
				indices[1] = idata[1];
				indices[2] = idata[2];
			}
			break;
		case IndexType::UINT_32:
			{
				uint32_t* idata{ reinterpret_cast<uint32_t*>(material.indices) + primitiveIndex * 3 };
				indices[0] = idata[0];
				indices[1] = idata[1];
				indices[2] = idata[2];
			}
			break;
		default:
			break;
	}
	uint8_t* attribBuffer{ material.attributeData };
	uint32_t attributesStride{ material.attributeStride };

	uint8_t* attr;

	// attr = material.attributeData + material.colorOffset;
	// if (static_cast<bool>(material.attributes & MaterialData::AttributeTypeBitfield::COLOR))
	// 	;
	if (static_cast<bool>(material.attributes & (MaterialData::AttributeTypeBitfield::NORMAL | MaterialData::AttributeTypeBitfield::FRAME)))
	{
		glm::vec3 shadingNormal{};
		glm::vec3 shadingTangent{};
		if (static_cast<bool>(material.attributes & MaterialData::AttributeTypeBitfield::FRAME))
		{
			attr = attribBuffer + material.frameOffset;

			glm::quat frames[3]{
				*reinterpret_cast<glm::quat*>(attr + attributesStride * indices[0]),
				*reinterpret_cast<glm::quat*>(attr + attributesStride * indices[1]),
				*reinterpret_cast<glm::quat*>(attr + attributesStride * indices[2]), };
			flipBitangent = cuda::std::signbit(frames[0].w);
			glm::mat3 frameMats[3]{
				glm::mat3_cast(frames[0]),
				glm::mat3_cast(frames[1]),
				glm::mat3_cast(frames[2]), };
			glm::vec3 normals[3]{ frameMats[0][1], frameMats[1][1], frameMats[2][1] };
			glm::vec3 tangents[3]{ frameMats[0][0], frameMats[1][0], frameMats[2][0] };
			shadingNormal = glm::vec3{
					baryWeights[0] * normals[0]
					+
					baryWeights[1] * normals[1]
					+
					baryWeights[2] * normals[2] };
			shadingTangent =
				baryWeights[0] * tangents[0]
				+
				baryWeights[1] * tangents[1]
				+
				baryWeights[2] * tangents[2];
		}
		else
		{
			attr = attribBuffer + material.normalOffset;

			glm::vec3 normals[3]{
				utility::octohedral::decode(*reinterpret_cast<glm::vec2*>(attr + attributesStride * indices[0])),
				utility::octohedral::decode(*reinterpret_cast<glm::vec2*>(attr + attributesStride * indices[1])),
				utility::octohedral::decode(*reinterpret_cast<glm::vec2*>(attr + attributesStride * indices[2])), };
			shadingNormal = glm::vec3{
				(1.0f - barycentrics.x - barycentrics.y) * normals[0]
				+
				barycentrics.x * normals[1]
				+
				barycentrics.y * normals[2] };
			float sign{ cuda::std::copysignf(1.0f, shadingNormal.z) };
			float a{ -1.0f / (sign + shadingNormal.z) };
			float b{ shadingNormal.x * shadingNormal.y * a };
			shadingTangent = glm::vec3(1.0f + sign * (shadingNormal.x * shadingNormal.x) * a, sign * b, -sign * shadingNormal.x);
		}

		normal = glm::normalize(normalTransform * shadingNormal);
		glm::mat3 vectorTransform{ glm::inverse(glm::transpose(normalTransform)) };
		tangent = glm::normalize(vectorTransform * shadingTangent);
	}
	else
	{
		normal = geoNormal;
		float sign{ cuda::std::copysignf(1.0f, geoNormal.z) };
		float a{ -1.0f / (sign + geoNormal.z) };
		float b{ geoNormal.x * geoNormal.y * a };
		tangent = glm::vec3(1.0f + sign * (geoNormal.x * geoNormal.x) * a, sign * b, -sign * geoNormal.x);
	}
	if (static_cast<bool>(material.attributes & MaterialData::AttributeTypeBitfield::TEX_COORD_1))
	{
		attr = attribBuffer + material.texCoord1Offset;
		texCoord1 =
			baryWeights[0] * (*reinterpret_cast<glm::vec2*>(attr + indices[0] * attributesStride)) +
			baryWeights[1] * (*reinterpret_cast<glm::vec2*>(attr + indices[1] * attributesStride)) +
			baryWeights[2] * (*reinterpret_cast<glm::vec2*>(attr + indices[2] * attributesStride));
	}
	if (static_cast<bool>(material.attributes & MaterialData::AttributeTypeBitfield::TEX_COORD_2))
	{
		attr = attribBuffer + material.texCoord2Offset;
		texCoord2 =
			baryWeights[0] * (*reinterpret_cast<glm::vec2*>(attr + indices[0] * attributesStride)) +
			baryWeights[1] * (*reinterpret_cast<glm::vec2*>(attr + indices[1] * attributesStride)) +
			baryWeights[2] * (*reinterpret_cast<glm::vec2*>(attr + indices[2] * attributesStride));
	}
}
CU_DEVICE CU_INLINE void resolveTextures(const MaterialData& material,
		const PathStateBitfield& stateFlags, const glm::vec3& rayDir,
		const glm::vec2& texCoords1, const glm::vec2& texCoords2, const glm::vec3& normal, const glm::vec3& tangent, bool flipTangent,
		LocalTransform& localTransform, microsurface::Surface& surface, bool& skipHit)
{
	glm::vec3 bitangent{ glm::cross(tangent, normal) * (flipTangent ? -1.0f : 1.0f) };
	glm::mat3 frame{ bitangent, tangent, normal };
	glm::vec3 n{};
	if (static_cast<bool>(material.textures & MaterialData::TextureTypeBitfield::NORMAL))
	{
		float2 uv{ material.nmTexCoordSetIndex ? float2{texCoords2.x, texCoords2.y} : float2{texCoords1.x, texCoords1.y} };
		float4 nm{ tex2D<float4>(material.normalTexture, uv.x, uv.y) };
		n = glm::normalize(glm::vec3{nm.y * 2.0f - 1.0f, nm.x * 2.0f - 1.0f, nm.z * 2.0f - 1.0f});
	}
	else
		n = glm::vec3{0.0f, 0.0f, 1.0f};
	n = frame * n;

	glm::vec3 shadingBitangent{ glm::normalize(glm::cross(tangent, n)) };
	glm::vec3 shadingTangent{ glm::normalize(glm::cross(n, shadingBitangent)) };
	shadingBitangent *= (flipTangent ? -1.0f : 1.0f);
	frame = glm::mat3{shadingBitangent, shadingTangent, n};
	localTransform = LocalTransform{frame};


	bool bcTexture{ static_cast<bool>(material.textures & MaterialData::TextureTypeBitfield::BASE_COLOR) };
	bool bcFactor{ static_cast<bool>(material.factors & MaterialData::FactorTypeBitfield::BASE_COLOR) };
	if (bcTexture)
	{
		float2 uv{ material.nmTexCoordSetIndex ? float2{texCoords2.x, texCoords2.y} : float2{texCoords1.x, texCoords1.y} };
		float4 bcTexData{ tex2D<float4>(material.baseColorTexture, uv.x, uv.y) };
		surface.base.color = glm::vec3{bcTexData.x, bcTexData.y, bcTexData.z};
		if (bcFactor)
			surface.base.color *= glm::vec3{material.baseColorFactor[0], material.baseColorFactor[1], material.baseColorFactor[2]};
		bool cutoff{ static_cast<bool>(material.factors & MaterialData::FactorTypeBitfield::CUTOFF) };
		if (bcTexData.w < material.alphaCutoff && cutoff)
			skipHit = true;
	}
	else if (bcFactor)
		surface.base.color = glm::vec3{material.baseColorFactor[0], material.baseColorFactor[1], material.baseColorFactor[2]};

	bool mrTexture{ static_cast<bool>(material.textures & MaterialData::TextureTypeBitfield::MET_ROUGH) };
	bool metFactor{ static_cast<bool>(material.factors & MaterialData::FactorTypeBitfield::METALNESS) };
	bool roughFactor{ static_cast<bool>(material.factors & MaterialData::FactorTypeBitfield::ROUGHNESS) };
	if (mrTexture)
	{
		float2 uv{ material.mrTexCoordSetIndex ? float2{texCoords2.x, texCoords2.y} : float2{texCoords1.x, texCoords1.y} };
		float4 mrTexData{ tex2D<float4>(material.pbrMetalRoughnessTexture, uv.x, uv.y) };
		surface.base.metalness = mrTexData.z * (metFactor ? material.metalnessFactor : 1.0f);
		surface.specular.roughness = mrTexData.y * (roughFactor ? material.roughnessFactor : 1.0f);
	}
	else
	{
		surface.base.metalness = metFactor ? material.metalnessFactor : surface.base.metalness;
		surface.specular.roughness = roughFactor ? material.roughnessFactor : surface.specular.roughness;
	}

	bool trTexture{ static_cast<bool>(material.textures & MaterialData::TextureTypeBitfield::TRANSMISSION) };
	bool trFactor{ static_cast<bool>(material.factors & MaterialData::FactorTypeBitfield::TRANSMISSION) };
	if (trTexture)
	{
		float2 uv{ material.trTexCoordSetIndex ? float2{texCoords2.x, texCoords2.y} : float2{texCoords1.x, texCoords1.y} };
		float trTexData{ tex2D<float>(material.transmissionTexture, uv.x, uv.y) };
		surface.transmission.weight = trTexData;
	}
	else
	{
		surface.transmission.weight = trFactor ? material.transmissionFactor : surface.transmission.weight;
	}
	surface.specular.ior = material.ior;
}

CU_DEVICE CU_INLINE SampledSpectrum emittedLightEval(const LaunchParameters& params, const SampledWavelengths& wavelengths, PathStateBitfield stateFlags,
		LightType type, uint16_t index, 
		const SampledSpectrum& throughputWeight, float bxdfPDF,
		uint32_t depth, const glm::vec3& rayOrigin, const glm::vec3& rayDir, float sqrdDistToLight)
{
	if (type == LightType::NONE)
		return SampledSpectrum{0.0f};

	float lightStructurePDF{ 1.0f / params.lights.lightCount };
	float lightPDF{};
	float lightPowerScale{};
	uint16_t emissionSpectrumDataIndex{};

	switch (type)
	{
		case LightType::DISK:
			{
				const DiskLightData& disk{ params.lights.disks[index] };
				glm::vec3 norm{ glm::mat3_cast(disk.frame)[2] };
				float lCos{ -glm::dot(rayDir, norm) };
				float surfacePDF{ 1.0f / (glm::pi<float>() * disk.radius * disk.radius) };
				lightPDF = surfacePDF * sqrdDistToLight / lCos;
				lightPowerScale = disk.powerScale * (lCos > 0.0f ? 1.0f : 0.0f);
				emissionSpectrumDataIndex = params.materials[disk.materialIndex].emissionSpectrumDataIndex;
			}
			break;
		case LightType::SPHERE:
			{
				const SphereLightData& sphere{ params.lights.spheres[index] };
				lightPDF = sampling::sphere::pdfUniformSolidAngle(rayOrigin, sphere.position, sphere.radius);
				lightPowerScale = sphere.powerScale;
				emissionSpectrumDataIndex = params.materials[sphere.materialIndex].emissionSpectrumDataIndex;
			}
			break;
		default:
			break;
	}
	lightPDF *= lightStructurePDF;

	SampledSpectrum Le{ params.spectra[emissionSpectrumDataIndex].sample(wavelengths) * lightPowerScale };
	float emissionWeight{};
	if (depth == 0 || static_cast<bool>(stateFlags & PathStateBitfield::PREVIOUS_HIT_SPECULAR))
		emissionWeight = 1.0f;
	else
		emissionWeight = MIS::powerHeuristic(1, bxdfPDF, 1, lightPDF);

	return throughputWeight * Le * emissionWeight;
}
CU_DEVICE CU_INLINE DirectLightSampleData sampledLightEval(const LaunchParameters& params, const SampledWavelengths& wavelengths,
		const glm::vec3& hitPoint, const glm::vec3& hitNormal,
		const glm::vec3& rand)
{
	DirectLightSampleData dlSampleData{};
	LightType type{};
	uint16_t index{};

	float lightStructurePDF{ params.lights.lightCount };
	uint16_t sampledLightIndex{ static_cast<uint16_t>((params.lights.lightCount - 0.0001f) * rand.z) };
	uint32_t lightC{ 0 };
	for (int i{ 0 }; i < KSampleableLightCount; ++i)
	{
		lightC += params.lights.orderedCount[i];
		if (sampledLightIndex < lightC)
		{
			type = KOrderedTypes[i];
			index = sampledLightIndex - (lightC - params.lights.orderedCount[i]);
			break;
		}
	}
	lightStructurePDF = 1.0f / lightStructurePDF;

	if (lightC == 0)
	{
		dlSampleData.occluded = true;
		return dlSampleData;
	}

	float dToL{};
	float lightPDF{};
	switch (type)
	{
		case LightType::DISK:
			{
				const DiskLightData& disk{ params.lights.disks[index] };
				dlSampleData.spectrumSample = params.spectra[params.materials[disk.materialIndex].emissionSpectrumDataIndex].sample(wavelengths) * disk.powerScale;
				glm::mat3 matframe{ glm::mat3_cast(disk.frame) };
				glm::vec3 lSmplPos{ disk.position + sampling::disk::sampleUniform3D(glm::vec2{rand.x, rand.y}, matframe) * disk.radius };
				glm::vec3 rToLight{ lSmplPos - hitPoint };
				float sqrdToLight{ rToLight.x * rToLight.x + rToLight.y * rToLight.y + rToLight.z * rToLight.z };
				dToL = cuda::std::sqrtf(sqrdToLight);
				dlSampleData.lightDir = rToLight / dToL;
				float lCos{ -glm::dot(matframe[2], dlSampleData.lightDir) };

				dlSampleData.occluded = lCos <= 0.0f;

				float surfacePDF{ 1.0f / (glm::pi<float>() * disk.radius * disk.radius) };
				lightPDF = surfacePDF * sqrdToLight / lCos;
			}
			break;
		case LightType::SPHERE:
			{
				const SphereLightData& sphere{ params.lights.spheres[index] };
				dlSampleData.spectrumSample = params.spectra[params.materials[sphere.materialIndex].emissionSpectrumDataIndex].sample(wavelengths) * sphere.powerScale;
				glm::vec3 lSmplPos{ sampling::sphere::sampleUniformWorldSolidAngle(glm::vec2{rand.x, rand.y}, hitPoint, sphere.position, sphere.radius, lightPDF)};

				glm::vec3 rToLight{ lSmplPos - hitPoint };
				dToL = glm::length(rToLight);
				dlSampleData.lightDir = rToLight / dToL;

				if (lightPDF == 0.0f || dToL <= 0.0f)
					dlSampleData.occluded = true;
			}
			break;
		default:
			break;
	}
	dlSampleData.lightSamplePDF = lightPDF * lightStructurePDF;

	bool offsetOutside{ glm::dot(hitNormal, dlSampleData.lightDir) > 0.0f };
	glm::vec3 rO{ utility::offsetPoint(hitPoint, offsetOutside ? hitNormal : -hitNormal) };
	const glm::vec3& lD{ dlSampleData.lightDir };
	if (!dlSampleData.occluded)
	{
		optixTraverse(parameters.traversable,
				{ rO.x, rO.y, rO.z },
				{ lD.x, lD.y, lD.z },
				0.0f,
				dToL - 0.5f,
				0.0f,
				0xFF,
				OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				0,
				0,
				0);
		dlSampleData.occluded = optixHitObjectIsHit();
	}

	return dlSampleData;
}

extern "C" __global__ void __closesthit__triangle()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

	uint32_t materialIndex{ *reinterpret_cast<uint32_t*>(optixGetSbtDataPointer()) };
	const MaterialData& material{ parameters.materials[materialIndex] };
	uint32_t primitiveIndex{ optixGetPrimitiveIndex() };
	float2 barycentrics{ optixGetTriangleBarycentrics() };

	float3 vertexObjectData[3];
	optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0.0f, vertexObjectData);
	
	float3 hposO;
	hposO.x = vertexObjectData[0].x * (1.0f - barycentrics.x - barycentrics.y) + vertexObjectData[1].x * barycentrics.x + vertexObjectData[2].x * barycentrics.y;
	hposO.y = vertexObjectData[0].y * (1.0f - barycentrics.x - barycentrics.y) + vertexObjectData[1].y * barycentrics.x + vertexObjectData[2].y * barycentrics.y;
	hposO.z = vertexObjectData[0].z * (1.0f - barycentrics.x - barycentrics.y) + vertexObjectData[1].z * barycentrics.x + vertexObjectData[2].z * barycentrics.y;

	float WFO[12]{};
	optixGetObjectToWorldTransformMatrix(WFO);
	float OFW[12]{};
	optixGetWorldToObjectTransformMatrix(OFW);

	float3 hpos{ hposO.x * WFO[0] + hposO.y * WFO[1] + hposO.z * WFO[2]  + WFO[3], 
				 hposO.x * WFO[4] + hposO.y * WFO[5] + hposO.z * WFO[6]  + WFO[7], 
				 hposO.x * WFO[8] + hposO.y * WFO[9] + hposO.z * WFO[10] + WFO[11] };

	float3 u{ vertexObjectData[2].x - vertexObjectData[0].x, vertexObjectData[2].y - vertexObjectData[0].y, vertexObjectData[2].z - vertexObjectData[0].z };
	float3 v{ vertexObjectData[1].x - vertexObjectData[0].x, vertexObjectData[1].y - vertexObjectData[0].y, vertexObjectData[1].z - vertexObjectData[0].z };
	float3 geometryNormal{u.y * v.z - u.z * v.y,
						  u.z * v.x - u.x * v.z,
						  u.x * v.y - u.y * v.x};
	geometryNormal = {geometryNormal.x * OFW[0] + geometryNormal.y * OFW[4] + geometryNormal.z * OFW[8], 
				      geometryNormal.x * OFW[1] + geometryNormal.y * OFW[5] + geometryNormal.z * OFW[9], 
				      geometryNormal.x * OFW[2] + geometryNormal.y * OFW[6] + geometryNormal.z * OFW[10]};
	// bool flipNormals{ optixIsFrontFaceHit() && material.doubleSided };
	bool flipNormals{ false };
	float normalizeDiv{ (flipNormals ? -1.0f : 1.0f) * __frsqrt_rn(geometryNormal.x * geometryNormal.x + geometryNormal.y * geometryNormal.y + geometryNormal.z * geometryNormal.z) };
	geometryNormal = {geometryNormal.x * normalizeDiv, geometryNormal.y * normalizeDiv, geometryNormal.z * normalizeDiv};
	uint32_t encGeometryNormal{ utility::octohedral::encodeU32(glm::vec3{geometryNormal.x, geometryNormal.y, geometryNormal.z}) };


	optixSetPayload_0(__float_as_uint(hpos.x));
	optixSetPayload_1(__float_as_uint(hpos.y));
	optixSetPayload_2(__float_as_uint(hpos.z));
	optixSetPayload_3(encGeometryNormal);
	optixSetPayload_4(primitiveIndex);
	optixSetPayload_5(__float_as_uint(barycentrics.x));
	optixSetPayload_6(__float_as_uint(barycentrics.y));
	optixSetPayload_7(
			(static_cast<uint32_t>(LightType::NONE) << 24) |
			(static_cast<uint32_t>(PathStateBitfield::TRIANGULAR_GEOMETRY) << 16) |
			(materialIndex & 0xFFFF));
	optixSetPayload_8(__float_as_uint(OFW[0]));
	optixSetPayload_9(__float_as_uint(OFW[4]));
	optixSetPayload_10(__float_as_uint(OFW[8]));
	optixSetPayload_11(__float_as_uint(OFW[1]));
	optixSetPayload_12(__float_as_uint(OFW[5]));
	optixSetPayload_13(__float_as_uint(OFW[9]));
	optixSetPayload_14(__float_as_uint(OFW[2]));
	optixSetPayload_15(__float_as_uint(OFW[6]));
	optixSetPayload_16(__float_as_uint(OFW[10]));
}
extern "C" __global__ void __intersection__disk()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

	float3 rOT{ optixGetWorldRayOrigin() };
	float3 rDT{ optixGetWorldRayDirection() };
	glm::vec3 rO{ rOT.x, rOT.y, rOT.z };
	glm::vec3 rD{ rDT.x, rDT.y, rDT.z };

	uint32_t lightIndex{ optixGetPrimitiveIndex() };

	const DiskLightData& dl{ parameters.lights.disks[lightIndex] };

	glm::mat3 matFrame{ glm::mat3_cast(dl.frame) };
	glm::vec3 dC{ dl.position };
	//glm::vec3 dT{ matFrame[0] };
	//glm::vec3 dB{ matFrame[1] };
	glm::vec3 dN{ matFrame[2] };
	dN = glm::dot(rD, dN) < 0.0f ? dN : -dN;
	float dR{ dl.radius };

	glm::vec3 o{ rO - dC };
	float t{ -glm::dot(dN, o) / glm::dot(rD, dN) };
	glm::vec3 rhP{ o + rD * t };

	bool intersect{ glm::dot(rhP, rhP) < dR * dR };
	if (intersect)
	{
		uint32_t encGeometryNormal{ utility::octohedral::encodeU32(dN) };
		glm::vec3 hP{ rhP + dC };
		optixSetPayload_0(__float_as_uint(hP.x));
		optixSetPayload_1(__float_as_uint(hP.y));
		optixSetPayload_2(__float_as_uint(hP.z));
		optixSetPayload_3(encGeometryNormal);
		optixSetPayload_4(lightIndex);
		optixSetPayload_7((static_cast<uint32_t>(LightType::DISK) << 24) | (dl.materialIndex & 0xFFFF));
		optixReportIntersection(t, 0);
	}
}
extern "C" __global__ void __closesthit__disk()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);
}
extern "C" __global__ void __closesthit__sphere()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

	const uint32_t primID{ optixGetPrimitiveIndex() };

	uint32_t lightIndex{ primID };
	const SphereLightData& sl{ parameters.lights.spheres[lightIndex] };
	glm::vec4 sphere{ sl.position, sl.radius };

	float3 rOT{ optixGetWorldRayOrigin() };
	float3 rDT{ optixGetWorldRayDirection() };
	glm::vec3 rO{ rOT.x, rOT.y, rOT.z };
	glm::vec3 rD{ rDT.x, rDT.y, rDT.z };

	glm::vec3 hP{ rO + rD * optixGetRayTmax() };
	glm::vec3 dN{ glm::normalize(hP - glm::vec3{sphere.x, sphere.y, sphere.z}) };
	glm::vec3 oc{ rO - glm::vec3{sphere.x, sphere.y, sphere.z} };
	float d2{ oc.x * oc.x + oc.y * oc.y + oc.z * oc.z };
	if (d2 < sphere.w * sphere.w)
		dN = -dN;

	uint32_t encGeometryNormal{ utility::octohedral::encodeU32(dN) };
	optixSetPayload_0(__float_as_uint(hP.x));
	optixSetPayload_1(__float_as_uint(hP.y));
	optixSetPayload_2(__float_as_uint(hP.z));
	optixSetPayload_3(encGeometryNormal);
	optixSetPayload_4(lightIndex);
	optixSetPayload_7((static_cast<uint32_t>(LightType::SPHERE) << 24) | (sl.materialIndex & 0xFFFF));
}

extern "C" __global__ void __miss__miss()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0); 
	optixSetPayload_7(static_cast<uint32_t>(LightType::SKY) << 24);
}

extern "C" __device__ void __direct_callable__PureDielectricBxDF(const MaterialData& materialData, const DirectLightSampleData& directLightData, const QRNG::State& qrngState,
		const LocalTransform& local, const microsurface::Surface& surface,
		SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathStateBitfield& stateFlags, float& refractionScale)
{
// 	glm::vec3 locWo{ -rD };
// 	glm::vec3 locLi{ directLightData.lightDir };
// 	local.toLocal(locWo, locLi);
//
// 	glm::vec3 rand{ QRNG::Sobol::sample3D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF_0) };
//
// 	float alpha{ utility::roughnessToAlpha(materialData.mfRoughnessValue) };
// 	microfacet::Alpha alphaMS{ .alphaX = alpha, .alphaY = alpha };
// 	if (static_cast<bool>(stateFlags & PathStateBitfield::REGULARIZED))
// 		alphaMS.regularize();
// 	glm::vec3 wo{ locWo };
// 	const float cosThetaO{ LocalTransform::cosTheta(wo) };
// 	glm::vec3 wi{};
// 	glm::vec3 wm{};
// 	microfacet::ContextOutgoing ctxo{ microfacet::createContextOutgoing(wo) };
//
// 	SampledSpectrum eta{ parameters.spectra[materialData.indexOfRefractSpectrumDataIndex].sample(wavelengths) };
//
// 	if (alpha < 0.001f || eta[0] == 1.0f)
// 	{
// 		SampledSpectrum R{ microfacet::FReal(cosThetaO, eta) };
// 		float p{ R[0] };
// 		if (rand.z < p)
// 		{
// 			wi = {-wo.x, -wo.y, wo.z};
// 			bxdfPDF = p;
// 			float pdfSum{ bxdfPDF };
// 			for (int i{ 1 }; i < SampledSpectrum::getSampleCount(); ++i)
// 			{
// 				if (wavelengths.getPDF()[i] != 0.0f)
// 					pdfSum += R[i];
// 			}
// 			throughputWeight *= R * wavelengths.getActiveCount() * (1.0f / pdfSum);
// 		}
// 		else
// 		{
// 			wavelengths.terminateAllSecondary();
// 			float T{ glm::clamp(1.0f - R[0], 0.0f, 1.0f) };
// 			bool valid;
// 			float& etaRel{ refractionScale };
// 			wi = utility::refract(wo, glm::vec3{0.0f, 0.0f, 1.0f}, eta[0], valid, &etaRel);
// 			if (!valid)
// 			{
// 				stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
// 				return;
// 			}
// 			bxdfPDF = cuda::std::fmax(0.0f, 1.0f - p);
// 			throughputWeight *= T / (etaRel * etaRel) / bxdfPDF;
// 		}
// 		stateFlags = wi.z < 0.0f ? stateFlags | PathStateBitfield::INSIDE_OBJECT : stateFlags & (~PathStateBitfield::INSIDE_OBJECT);
// 		stateFlags = stateFlags | PathStateBitfield::CURRENT_HIT_SPECULAR;
// 		local.fromLocal(wi);
// 		rD = wi;
// 		return;
// 	}
//
// 	if (!directLightData.occluded)
// 	{
// 		wi = locLi;
// 		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
// 		float cosThetaI{ LocalTransform::cosTheta(wi) };
// 		const float cosFactor{ cuda::std::fabs(cosThetaI) };
// 		float t{ cosThetaO * cosThetaI };
// 		bool reflect{ t > 0.0f };
// 		float G{ microfacet::G(ctxi, ctxo, alphaMS) };
// 		float lbxdfPDF;
//
// 		SampledSpectrum f{};
// #pragma unroll
// 		for (int i{ 0 }; i < SampledSpectrum::getSampleCount(); ++i)
// 		{
// 			if (wavelengths.getPDF()[i] != 0.0f)
// 			{
// 				float etaRel{ 1.0f };
// 				if (!reflect)
// 					etaRel = cosThetaO > 0.0f ? eta[i] : 1.0f / eta[i];
// 				wm = wi * etaRel + wo;
// 				wm = glm::normalize(wm);
// 				wm = wm.z > 0.0f ? wm : -wm;
// 				float dotWmWo{ glm::dot(wm, wo) };
// 				float dotWmWi{ glm::dot(wm, wi) };
// 				if (t != 0.0f && !(dotWmWo * cosThetaO < 0.0f || dotWmWi * cosThetaI < 0.0f))
// 				{
// 					microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };
// 					const float R{ microfacet::FReal(dotWmWo, eta[i]) };
// 					const float T{ glm::clamp(1.0f - R, 0.0f, 1.0f) };
// 					const float pR{ R / (R + T) };
// 					const float pT{ cuda::std::fmax(0.0f, 1.0f - pR) };
// 					const float wowmAbsDot{ cuda::std::fabs(dotWmWo) };
//
// 					if (reflect)
// 					{
// 						f[i] = microfacet::D(ctxm, alphaMS) * R * G
// 							   / (4.0f * cosThetaO * cosThetaI);
// 						if (i == 0)
// 							lbxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) / (4.0f * wowmAbsDot) * pR;
// 					}
// 					else
// 					{
// 						float t{ dotWmWi + dotWmWo / etaRel };
// 						float denom{ t * t };
// 						float dwmdwi{ cuda::std::fabs(dotWmWi) / denom };
// 						f[i] = microfacet::D(ctxm, alphaMS) * T * G
// 							   * cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom))
// 							   / (etaRel * etaRel);
// 						if (i == 0)
// 							lbxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) * dwmdwi * pT;
// 					}
// 				}
// 			}
// 		}
// 		L += directLightData.spectrumSample * f * throughputWeight
// 			* (cosFactor * MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, lbxdfPDF) / directLightData.lightSamplePDF);
// 	}
//
// 	SampledSpectrum R{};
//
// 	wm = microfacet::VNDF::sample<microfacet::VNDF::SPHERICAL_CAP>(wo, alphaMS, rand);
// 	microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };
//
// 	float dotWmWo{ glm::dot(wo, wm) };
// 	float absDotWmWo{ cuda::std::fabs(dotWmWo) };
// 	const float heroR{ microfacet::FReal(dotWmWo, eta[0]) };
// 	const float heroT{ glm::clamp(1.0f - heroR, 0.0f, 1.0f) };
// 	const float pR{ heroR };
//
// 	SampledSpectrum f;
// 	float cosFactor;
// 	float condPDFCount{ 0.0f };
// 	float condPDFSum{ 0.0f };
// 	if (rand.z < pR)
// 	{
// 		wi = utility::reflect(wo, wm);
// 		if (wo.z * wi.z <= 0.0f)
// 		{
// 			stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
// 			return;
// 		}
// 		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
// 		const float cosThetaI{ LocalTransform::cosTheta(wi) };
// 		R[0] = heroR;
// 		for (int i{ 1 }; i < SampledSpectrum::getSampleCount(); ++i)
// 			R[i] = microfacet::FReal(dotWmWo, eta[i]);
// 		f = microfacet::D(ctxm, alphaMS) * R * microfacet::G(ctxi, ctxo, alphaMS)
// 			/ (4.0f * cosThetaO * cosThetaI);
// 		float pdfTerm{ microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, absDotWmWo, ctxo, ctxm, alphaMS) / (4.0f * absDotWmWo) };
// 		bxdfPDF = pdfTerm * heroR;
// 		condPDFCount = wavelengths.getActiveCount();
// 		condPDFSum = bxdfPDF;
// 		for (int i{ 1 }; i < SampledSpectrum::getSampleCount(); ++i)
// 		{
// 			if (wavelengths.getPDF()[i] != 0.0f)
// 				condPDFSum += pdfTerm * R[i];
// 		}
// 		cosFactor = cuda::std::fabs(cosThetaI);
// 	}
// 	else
// 	{
// 		bool valid;
// 		float& etaRel{ refractionScale };
// 		wi = utility::refract(wo, wm, eta[0], valid, &etaRel);
// 		if (wo.z * wi.z >= 0.0f || !valid)
// 		{
// 			stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
// 			return;
// 		}
// 		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
// 		const float cosThetaI{ LocalTransform::cosTheta(wi) };
// 		float dotWmWi{ glm::dot(wm, wi) };
// 		float t{ dotWmWi + dotWmWo / etaRel };
// 		float denom{ t * t };
// 		float dwmdwi{ cuda::std::fabs(dotWmWi) / denom };
// 		float G{ microfacet::G(ctxi, ctxo, alphaMS) };
// 		float fh{ microfacet::D(ctxm, alphaMS) * heroT * G
// 				  * cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom)) };
// 		const float pT{ cuda::std::fmax(0.0f, 1.0f - pR) };
// 		bxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, absDotWmWo, ctxo, ctxm, alphaMS) * dwmdwi * pT;
// 		condPDFSum = bxdfPDF;
// 		cosFactor = cuda::std::fabs(cosThetaI);
// 		f[0] = fh;
// 		for (int i{ 1 }; i < SampledSpectrum::getSampleCount(); ++i)
// 		{
// 			if (wavelengths.getPDF()[i] != 0.0f)
// 			{
// 				etaRel = cosThetaO > 0.0f ? eta[i] : 1.0f / eta[i];
// 				wm = glm::normalize(wi * etaRel + wo);
// 				wm = wm.z > 0.0f ? wm : -wm;
// 				ctxm = microfacet::createContextMicronormal(wm);
// 				dotWmWo = glm::dot(wm, wo);
// 				dotWmWi = glm::dot(wm, wi);
// 				absDotWmWo = cuda::std::fabs(dotWmWo);
// 				const float secR{ microfacet::FReal(dotWmWo, eta[i]) };
// 				const float secT{ glm::clamp(1.0f - secR, 0.0f, 1.0f) };
// 				t = dotWmWi + dotWmWo / etaRel;
// 				denom = t * t;
// 				dwmdwi = cuda::std::fabs(dotWmWi) / denom;
// 				f[i] = microfacet::D(ctxm, alphaMS) * secT * G
// 					* cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom));
// 				float pdf{ microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, absDotWmWo, ctxo, ctxm, alphaMS) * dwmdwi * secT };
// 				if (pdf <= 0.0f || (!(dotWmWo * cosThetaO < 0.0f || dotWmWi * cosThetaI < 0.0f)))
// 					wavelengths.terminateSecondary(i);
// 				else
// 					condPDFSum += pdf;
// 			}
// 		}
// 		condPDFCount = wavelengths.getActiveCount();
// 		f /= (etaRel * etaRel);
// 	}
//
// 	if (bxdfPDF == 0.0f)
// 	{
// 		stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
// 		return;
// 	}
//
// 	throughputWeight *= condPDFCount * (1.0f / (condPDFSum)) * f * cosFactor;
//
// 	glm::vec3 locWi{ wi };
// 	stateFlags = locWi.z < 0.0f ? stateFlags | PathStateBitfield::INSIDE_OBJECT : stateFlags & (~PathStateBitfield::INSIDE_OBJECT);
// 	local.fromLocal(locWi);
// 	rD = glm::normalize(locWi);
}
extern "C" __device__ void __direct_callable__PureConductorBxDF(const MaterialData& materialData, const DirectLightSampleData& directLightData, const QRNG::State& qrngState,
		const LocalTransform& local, const microsurface::Surface& surface,
		SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathStateBitfield& stateFlags, float& refractionScale)
{
	glm::vec3 locWo{ -rD };
	glm::vec3 locLi{ directLightData.lightDir };
	local.toLocal(locWo, locLi);

	if (locWo.z == 0.0f)
	{
		stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
		return;
	}

	SampledSpectrum eta{ parameters.spectra[materialData.indexOfRefractSpectrumDataIndex].sample(wavelengths) };
	SampledSpectrum k{ parameters.spectra[materialData.absorpCoefSpectrumDataIndex].sample(wavelengths) };

	float alpha{ utility::roughnessToAlpha(materialData.mfRoughnessValue) };
	microfacet::Alpha alphaMS{ .alphaX = alpha, .alphaY = alpha };
	if (static_cast<bool>(stateFlags & PathStateBitfield::REGULARIZED))
		alphaMS.regularize();
	glm::vec3 wo{ locWo };
	glm::vec3 wi{};
	glm::vec3 wm{};
	microfacet::ContextOutgoing ctxo{ microfacet::createContextOutgoing(wo) };

	if (alpha < 0.001f)
	{
		wi = glm::vec3{-wo.x, -wo.y, wo.z};
		float absCosTheta{ cuda::std::fabs(LocalTransform::cosTheta(wi)) };
		throughputWeight *= microfacet::FComplex(absCosTheta, eta, k);
		bxdfPDF = 1.0f;
		stateFlags = stateFlags | PathStateBitfield::CURRENT_HIT_SPECULAR;
		local.fromLocal(wi);
		rD = glm::normalize(wi);
		return;
	}

	if (!directLightData.occluded)
	{
		wi = locLi;
		if (wo.z * wi.z > 0.0f)
		{
			microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
			wm = glm::normalize(wi + wo);
			microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };

			const float wowmAbsDot{ cuda::std::fabs(glm::dot(wo, wm)) };
			SampledSpectrum f{ microfacet::D(ctxm, alphaMS) * microfacet::FComplex(wowmAbsDot, eta, k) * microfacet::G(ctxi, ctxo, alphaMS) 
							   / cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi)) };
			float cosFactor{ cuda::std::fabs(LocalTransform::cosTheta(wi)) };
			wm = wm.z > 0.0f ? wm : -wm;
			float lbxdfPDF{ microfacet::VNDF::PDF<microfacet::VNDF::BOUNDED_SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) / (4.0f * wowmAbsDot) };
			L += directLightData.spectrumSample * f * throughputWeight * cosFactor
				* MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, lbxdfPDF)
				/ directLightData.lightSamplePDF;
		}
	}

	glm::vec2 rand{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF_0) };
	wm = microfacet::VNDF::sample<microfacet::VNDF::BOUNDED_SPHERICAL_CAP>(wo, alphaMS, rand);
	microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };

	wi = utility::reflect(wo, wm);
	microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };

	if (wo.z * wi.z <= 0.0f)
	{
		stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
		return;
	}

	const float wowmAbsDot{ cuda::std::fabs(glm::dot(wo, wm)) };

	SampledSpectrum f{ microfacet::D(ctxm, alphaMS) * microfacet::FComplex(wowmAbsDot, eta, k) * microfacet::G(ctxi, ctxo, alphaMS) 
					   / cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi)) };
	float cosFactor{ cuda::std::fabs(LocalTransform::cosTheta(wi)) };
	bxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::BOUNDED_SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) / (4.0f * wowmAbsDot);
	throughputWeight *= f * cosFactor / bxdfPDF;

	glm::vec3 locWi{ wi };
	local.fromLocal(locWi);
	rD = glm::normalize(locWi);
}
extern "C" __device__ void __direct_callable__ComplexSurface_BxDF(const MaterialData& materialData, const DirectLightSampleData& directLightData, const QRNG::State& qrngState,
		const LocalTransform& local, const microsurface::Surface& surface,
		SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathStateBitfield& stateFlags, float& refractionScale)
{
	glm::vec3 locWo{ -rD };
	glm::vec3 locLi{ directLightData.lightDir };
	local.toLocal(locWo, locLi);

	if (locWo.z == 0.0f)
	{
		stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
		return;
	}

	float roughness{ cuda::std::fmax(0.001f, surface.specular.roughness) };
	if (roughness < 0.01f)
		stateFlags = stateFlags | PathStateBitfield::CURRENT_HIT_SPECULAR;
	float alpha{ utility::roughnessToAlpha(roughness) };
	microfacet::Alpha alphaMS{ .alphaX = alpha, .alphaY = alpha };

	// if (static_cast<bool>(stateFlags & PathStateBitfield::REGULARIZED))
	// 	alphaMS.regularize();
	glm::vec3 wo{ locWo };
	microfacet::ContextOutgoing ctxo{ microfacet::createContextOutgoing(wo) };
	glm::vec3 wm{};
	glm::vec3 wi{};

	float directionalAlbedoConductor{ tex2D<float>(parameters.LUTs.conductorAlbedo, LocalTransform::cosTheta(wo), roughness) };
	float eta{ wo.z > 0.0f ? surface.specular.ior / 1.0f : 1.0f / surface.specular.ior };
	float f0Sr{ (1.0f - eta) / (1.0f + eta) };
	float f0{ f0Sr * f0Sr };
	float energyCompensationTermDielectric{ 1.0f / tex3D<float>(eta > 1.0f ? parameters.LUTs.dielectricOuterAlbedo : parameters.LUTs.dielectricInnerAlbedo,
			LocalTransform::cosTheta(wo), roughness, cuda::std::sqrt(cuda::std::abs(f0Sr))) };
	float energyPreservationTermDiffuse{ 1.0f - tex3D<float>(eta > 1.0f ? parameters.LUTs.reflectiveDielectricOuterAlbedo : parameters.LUTs.reflectiveDielectricInnerAlbedo,
			LocalTransform::cosTheta(wo), roughness, cuda::std::sqrt(cuda::std::abs(f0Sr))) };

	if (!directLightData.occluded)
	{
		wi = locLi;
		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };

		const float cosThetaI{ LocalTransform::cosTheta(wi) };
		const float cosThetaO{ LocalTransform::cosTheta(wo) };
		float cosFactor{ cuda::std::fabs(cosThetaI) };

		bool reflect{ cosThetaI * cosThetaO > 0.0f };
		float etaR{ 1.0f };
		if (!reflect)
			etaR = eta;
		wm = glm::normalize(wi * etaR + wo);
		microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };

		const float wowmAbsDot{ cuda::std::fabs(glm::dot(wo, wm)) };

		float FDielectric{ microfacet::FSchlick(f0, wowmAbsDot) };
		float Ds{ microfacet::D(ctxm, alphaMS) };
		float Gs{ microfacet::G(ctxi, ctxo, alphaMS) };

		float metalness{ surface.base.metalness };
		float translucency{ surface.transmission.weight };
		float cSpecWeight{ metalness };
		float dSpecWeight{ (1.0f - metalness) * FDielectric };
		float diffWeight{ (1.0f - metalness) * (1.0f - translucency) * (1.0f - FDielectric) };
		float dTransWeight{ (1.0f - metalness) * translucency * (1.0f - FDielectric) };

		float lbxdfSpecPDF{ microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) / (4.0f * wowmAbsDot) };
		float lbxdfDiffPDF{ diffuse::CosWeighted::PDF(LocalTransform::cosTheta(wi)) };
		float dotWmWo{ glm::dot(wm, wo) };
		float dotWmWi{ glm::dot(wm, wi) };

		if (reflect)
		{
			SampledSpectrum FConductor{ color::RGBtoSpectrum(microfacet::F82(surface.base.color, wowmAbsDot) *
					(1.0f + microfacet::FAvgIntegralForF82(surface.base.color) * ((1.0f - directionalAlbedoConductor) / directionalAlbedoConductor)),
					wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) };

			SampledSpectrum flConductor{ Ds * Gs * FConductor / cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi)) };
			float flDielectric{ Ds * Gs * energyCompensationTermDielectric // FDielectric is included in dSpecWeight
				/ cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi)) };
			SampledSpectrum flDiffuse{ energyPreservationTermDiffuse *
				color::RGBtoSpectrum(surface.base.color, wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) / glm::pi<float>() };


			float bxdfPDF{ cSpecWeight * lbxdfSpecPDF + dSpecWeight * lbxdfSpecPDF + diffWeight * lbxdfDiffPDF };
			float mis{ MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, bxdfPDF) };
			L += directLightData.spectrumSample * throughputWeight * cosFactor * mis * (
					cSpecWeight * flConductor
					+
					dSpecWeight * flDielectric
					+
					diffWeight * flDiffuse
					) / directLightData.lightSamplePDF;
		}
		else if (surface.transmission.weight != 0.0f && !(dotWmWo * cosThetaO <= 0.0f || dotWmWi * cosThetaI <= 0.0f))
		{
			float t{ dotWmWi + dotWmWo / eta };
			float denom{ t * t };
			float dwmdwi{ cuda::std::fabs(dotWmWi) / denom };
			float lbxdfTransPDF{ microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) * dwmdwi };
			float bxdfPDF{ dTransWeight * lbxdfTransPDF };
			float mis{ MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, bxdfPDF) };
			SampledSpectrum flTransmission{ Ds * (1.0f - FDielectric) * Gs * energyCompensationTermDielectric
				* cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom)) };
			flTransmission /= (eta * eta);
			L += directLightData.spectrumSample * throughputWeight * flTransmission * cosFactor * mis / directLightData.lightSamplePDF;
		}
	}

	glm::vec3 rand{ QRNG::Sobol::sample3D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF_0) };
	wm = microfacet::VNDF::sample<microfacet::VNDF::SPHERICAL_CAP>(wo, alphaMS, rand);
	microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };
	const float wowmAbsDot{ cuda::std::abs(glm::dot(wo, wm)) };

	float Ds{ microfacet::D(ctxm, alphaMS) };
	float FDielectric{ microfacet::FSchlick(f0, wowmAbsDot) };

	float conductorP{ surface.base.metalness };
	float dielectricSpecP{ cuda::std::fmax(0.0f, FDielectric * (1.0f - conductorP)) };
	float translucency{ surface.transmission.weight };
	float transmissionP{ cuda::std::fmax(0.0f, (1.0f - FDielectric) * translucency * (1.0f - conductorP)) };
	float diffuseP{ cuda::std::fmax(0.0f, (1.0f - FDielectric) * (1.0f - translucency) * (1.0f - conductorP)) };

	if (rand.z < conductorP + dielectricSpecP)
	{
		wi = glm::normalize(utility::reflect(wo, wm));
		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
		if (wo.z * wi.z <= 0.0f)
		{
			stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
			return;
		}
		float Gs{ microfacet::G(ctxi, ctxo, alphaMS) };
		float spec{ Ds * Gs / cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi)) };
		SampledSpectrum f;
		float pdfP;
		if (rand.z < conductorP)
		{
			SampledSpectrum FConductor{ color::RGBtoSpectrum(microfacet::F82(surface.base.color, wowmAbsDot) *
					(1.0f + microfacet::FAvgIntegralForF82(surface.base.color) * ((1.0f - directionalAlbedoConductor) / directionalAlbedoConductor)),
					wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) };
			f = spec * FConductor;
			pdfP = conductorP;
		}
		else
		{
			f = spec * FDielectric * energyCompensationTermDielectric;
			pdfP = dielectricSpecP;
		}
		float cosFactor{ cuda::std::fabs(LocalTransform::cosTheta(wi)) };
		bxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) * pdfP / (4.0f * wowmAbsDot);
		throughputWeight *= f * cosFactor / bxdfPDF;
	}
	else
	{
		if (rand.z < conductorP + dielectricSpecP + transmissionP)
		{
			bool valid;
			wi = utility::refract(wo, wm, eta, valid);
			refractionScale *= eta;
			if (wo.z * wi.z >= 0.0f || !valid)
			{
				stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
				return;
			}
			microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
			float Gs{ microfacet::G(ctxi, ctxo, alphaMS) };
			const float cosThetaI{ LocalTransform::cosTheta(wi) };
			const float cosThetaO{ LocalTransform::cosTheta(wo) };
			float dotWmWo{ glm::dot(wm, wo) };
			float dotWmWi{ glm::dot(wm, wi) };
			float t{ dotWmWi + dotWmWo / eta };
			float denom{ t * t };
			float dwmdwi{ cuda::std::fabs(dotWmWi) / denom };
			float f{ Ds * (1.0f - FDielectric) * Gs
				* cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom)) };
			f /= (eta * eta);
			bxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) * dwmdwi * transmissionP;
			float cosFactor{ cuda::std::fabs(cosThetaI) };
			throughputWeight *= color::RGBtoSpectrum(surface.base.color, wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB)
				* f * energyCompensationTermDielectric * cosFactor / bxdfPDF;
			stateFlags |= PathStateBitfield::RAY_REFRACTED;
		}
		else
		{
			glm::vec2 randD{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF_1) };
			wi = diffuse::CosWeighted::sample(randD);
			if (wo.z * wi.z <= 0.0f)
			{
				stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
				return;
			}
			float cosFactor{ cuda::std::fabs(LocalTransform::cosTheta(wi)) };
			SampledSpectrum f{ color::RGBtoSpectrum(surface.base.color, wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) / glm::pi<float>() };
			bxdfPDF = diffuse::CosWeighted::PDF(LocalTransform::cosTheta(wi)) * diffuseP;
			throughputWeight *= f * energyPreservationTermDiffuse * cosFactor / bxdfPDF;
		}
	}

	if (bxdfPDF <= 0.0f || isinf(bxdfPDF))
	{
		stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
		return;
	}

	glm::vec3 locWi{ wi };
	local.fromLocal(locWi);
	rD = glm::normalize(locWi);
}

extern "C" __global__ void __raygen__main()
{
	uint3 li{ optixGetLaunchIndex() };
	glm::vec2 pixelCoordinate{ static_cast<float>(li.x), static_cast<float>(li.y) };

	QRNG::State qrngState{ parameters.samplingState.offset, QRNG::getPixelHash(li.x, li.y) };

	LaunchParameters::ResolutionState& resState{ parameters.resolutionState };

	glm::dvec4 result{ parameters.samplingState.offset != 0 ? parameters.renderData[li.y * resState.filmWidth + li.x] : glm::dvec4{0.0} };
	uint32_t sample{ 0 };
	bool terminated{ false };
	do
	{
		const glm::vec2 subsample{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::FILTER) };
		const glm::vec2 subsampleCoordinate{ pixelCoordinate + filter::gaussian::sampleDistribution(subsample) };
		const glm::vec2 lensSample{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::LENS) };
		Ray ray;
		if (parameters.cameraState.depthOfFieldEnabled)
		{
			ray = generateThinLensCamera(subsampleCoordinate,
					lensSample, parameters.cameraState.focusDistance, parameters.cameraState.appertureSize,
					glm::vec2{resState.invFilmWidth, resState.invFilmHeight}, glm::vec2{resState.camPerspectiveScaleW, resState.camPerspectiveScaleH},
					parameters.cameraState.camU, parameters.cameraState.camV, parameters.cameraState.camW);
		}
		else
		{
			ray = generatePinholeCameraDirection(subsampleCoordinate,
					glm::vec2{resState.invFilmWidth, resState.invFilmHeight}, glm::vec2{resState.camPerspectiveScaleW, resState.camPerspectiveScaleH},
					parameters.cameraState.camU, parameters.cameraState.camV, parameters.cameraState.camW);
		}
		glm::vec3& rO{ ray.o };
		glm::vec3& rD{ ray.d };

		SampledWavelengths wavelengths{ SampledWavelengths::sampleVisible(QRNG::Sobol::sample1D(qrngState, QRNG::DimensionOffset::WAVELENGTH)) };
		SampledSpectrum L{ 0.0f };
		SampledSpectrum throughputWeight{ 1.0f };
		float refractionScale{ 1.0f };
		float bxdfPDF{ 1.0f };
		LightType lightType{ LightType::NONE };
		uint16_t lightIndex{};
		PathStateBitfield stateFlags{ 0 };
		bool continuePath{};
		uint32_t depth{ 0 };
		do
		{
			uint32_t pl[17];
			optixTrace(OPTIX_PAYLOAD_TYPE_ID_0, //Payload type
					   parameters.traversable, //Traversable handle
					   { rO.x, rO.y, rO.z }, //Ray origin
					   { rD.x, rD.y, rD.z }, //Ray direction
					   0.0f, //Min "t"
					   FLT_MAX, //Max "t"
					   0.0f, //Time
					   0xFF, //Visibility mask
					   0, //Flags
					   0, //SBT offset
					   1, //SBT stride
					   0, //SBT miss program index
					   pl[0], pl[1], pl[2], pl[3], pl[4], pl[5], pl[6], pl[7], pl[8], pl[9], pl[10], pl[11], pl[12], pl[13], pl[14], pl[15], pl[16]); //Payload

			glm::vec3 hP; //Hit position
			uint32_t encHitGNormal;
			uint32_t primitiveIndex;
			glm::vec2 barycentrics;
			MaterialData* material;
			glm::mat3 normalTransform;
			unpackTraceData(parameters, hP, encHitGNormal,
					stateFlags,
					primitiveIndex, barycentrics,
					lightType, lightIndex, &material, normalTransform,
					pl);

			if (lightType == LightType::SKY)
			{
				L += throughputWeight * color::RGBtoSpectrum(glm::vec3{0.246f, 0.623f, 0.956f} * 0.002f, wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB);

				goto breakPath;
			}

			glm::vec3 toHit{ hP - rO };
			L += emittedLightEval(parameters, wavelengths, stateFlags,
					lightType, lightIndex,
					throughputWeight, bxdfPDF,
					depth, rO, rD, toHit.x * toHit.x + toHit.y * toHit.y + toHit.z * toHit.z);

			glm::vec3 hNG{ utility::octohedral::decodeU32(encHitGNormal) }; //Hit normal geometry
			DirectLightSampleData directLightData{ sampledLightEval(parameters, wavelengths,
						hP, hNG,
						QRNG::Sobol::sample3D(qrngState, QRNG::DimensionOffset::LIGHT)) };

			bool skipHit{ false };

			// Launch BxDF evaluation
			LocalTransform localTransform;
			microsurface::Surface surface{};
			if (static_cast<bool>(stateFlags & PathStateBitfield::TRIANGULAR_GEOMETRY))
			{
				glm::vec2 texC1;
				glm::vec2 texC2;
				glm::vec3 normal;
				glm::vec3 tangent;
				bool flipTangent{ false };
				resolveAttributes(*material, stateFlags, barycentrics, primitiveIndex, hNG, normalTransform,
						texC1, texC2, normal, tangent, flipTangent);
				resolveTextures(*material, stateFlags, rD, texC1, texC2, normal, tangent, flipTangent, localTransform, surface, skipHit);
			}
			else
				localTransform = LocalTransform{hNG};

			hNG = cuda::std::copysign(1.0f, -glm::dot(hNG, rD)) * hNG; // Turn into hit relative normal

			if (!skipHit)
			{
				optixDirectCall<void, 
					const MaterialData&, const DirectLightSampleData&, const QRNG::State&,
					const LocalTransform&, const microsurface::Surface&,
					SampledSpectrum&, SampledWavelengths&, SampledSpectrum&,
					float&, glm::vec3&, PathStateBitfield&, float&>
						(material->bxdfIndexSBT,
						 *material, directLightData, qrngState,
						 localTransform, surface,
						 L, wavelengths, throughputWeight,
						 bxdfPDF, rD, stateFlags, refractionScale);
			}

			// TODO: Debug color output
			//
			// glm::vec3 c{};
			// if ((static_cast<uint32_t>(texC1.x * 50.0f) % 2) == ((static_cast<uint32_t>(texC1.y * 50.0f) % 2) == 0 ? 1 : 0))
			// 	c = glm::vec3{0.8f};
			// else
			// 	c = glm::vec3{0.5f};
			// L = color::RGBtoSpectrum(c, wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB);
			//

			if (float tMax{ throughputWeight.max() * refractionScale }; tMax < 1.0f && depth > 0)
			{
				float q{ cuda::std::fmax(0.0f, 1.0f - tMax) };
				q = q * q;
				if (QRNG::Sobol::sample1D(qrngState, QRNG::DimensionOffset::ROULETTE) < q)
					goto breakPath;
				throughputWeight /= 1.0f - q;
			}

			hNG = static_cast<bool>(stateFlags & PathStateBitfield::RAY_REFRACTED) || skipHit ? -hNG : hNG;
			float rCorrection{ glm::dot(hNG, rD) };
			if (rCorrection < 0.0f)
				rD = glm::normalize(rD + (hNG * (-rCorrection + 0.01f)));
			rO = utility::offsetPoint(hP, hNG);

			continuePath = ++depth < parameters.pathState.maxPathDepth;
			if (!static_cast<bool>(stateFlags & PathStateBitfield::CURRENT_HIT_SPECULAR) && !skipHit)
			{
				if (static_cast<bool>(stateFlags & PathStateBitfield::REFRACTION_HAPPENED))
					continuePath = depth < parameters.pathState.maxTransmittedPathDepth;
				else
					continuePath = depth < parameters.pathState.maxReflectedPathDepth;
			}

			qrngState.advanceBounce();
			updateStateFlags(stateFlags);
			terminated = static_cast<bool>(stateFlags & PathStateBitfield::PATH_TERMINATED);
		} while (continuePath && !terminated);
	breakPath:
		qrngState.advanceSample();

		if (!terminated)
		{
			resolveSample(L, wavelengths.getPDF());
			result += glm::dvec4{color::toRGB(*parameters.sensorSpectralCurveA, *parameters.sensorSpectralCurveB, *parameters.sensorSpectralCurveC,
					wavelengths, L), filter::computeFilterWeight(subsample)};
		}
		else
		{
			result += glm::dvec4{0.0, 0.0, 0.0, filter::computeFilterWeight(subsample)};
		}
	} while (++sample < parameters.samplingState.count);
	parameters.renderData[li.y * resState.filmWidth + li.x] = result;
}
