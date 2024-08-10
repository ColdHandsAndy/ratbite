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
#include "../device/microfacet.h"
#include "../device/mis.h"

enum class PathStateBitfield : uint32_t
{
	NO_FLAGS = 0u,
	PREVIOUS_HIT_SPECULAR = 1u,
	CURRENT_HIT_SPECULAR = 2u,
	MISS = 4u,
	TRANSMISSION = 8u,
	EMISSIVE_OBJECT_HIT = 16u,
	REGULARIZED = 32u,
	SECONDARY_SPECTRAL_SAMPLES_TERMINATED = 64u,
	INSIDE_OBJECT = 128u,
	PATH_TERMINATED = 256u,
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
		uint32_t& primitiveIndex, glm::vec2& barycentrics,
		LightType& lightType, uint16_t& lightIndex,
		MaterialData** materialDataPtr,
		glm::quat& shadingFrame,
		uint32_t pl0, uint32_t pl1, uint32_t pl2, uint32_t pl3, uint32_t pl4, uint32_t pl5, uint32_t pl6, uint32_t pl7, uint32_t pl8, uint32_t pl9, uint32_t pl10, uint32_t pl11)
{
	hP = glm::vec3{ __uint_as_float(pl0), __uint_as_float(pl1), __uint_as_float(pl2) };
	encHGN = pl3;
	uint32_t matIndex{ pl7 & 0xFFFF };
	*materialDataPtr = params.materials + matIndex;
	lightType = static_cast<LightType>(pl7 >> 16);
	if (lightType != LightType::NONE)
		lightIndex = pl4;
	else
	{
		primitiveIndex = pl4;
		barycentrics = {__uint_as_float(pl5), __uint_as_float(pl6)};
	}
	shadingFrame.x = __uint_as_float(pl8);
	shadingFrame.y = __uint_as_float(pl9);
	shadingFrame.z = __uint_as_float(pl10);
	shadingFrame.w = __uint_as_float(pl11);
}
CU_DEVICE CU_INLINE void updateStateFlags(PathStateBitfield& stateFlags)
{
	PathStateBitfield excludeFlags{ PathStateBitfield::EMISSIVE_OBJECT_HIT | PathStateBitfield::CURRENT_HIT_SPECULAR };
	PathStateBitfield includeFlags{ static_cast<bool>(stateFlags & PathStateBitfield::CURRENT_HIT_SPECULAR) ?
		PathStateBitfield::PREVIOUS_HIT_SPECULAR : PathStateBitfield::REGULARIZED };

	stateFlags = (stateFlags & (~excludeFlags)) | includeFlags;
}
CU_DEVICE CU_INLINE void resolveSample(SampledSpectrum& L, const SampledSpectrum& pdf)
{
	for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
	{
		L[i] = pdf[i] != 0.0f ? L[i] / pdf[i] : 0.0f;
	}
}
CU_DEVICE CU_INLINE void resolveAttributes(const MaterialData& material, const glm::vec2& barycentrics, uint32_t primitiveIndex, glm::quat& shadingFrame)
{
	// uint8_t* attribBuffer{ material.attributeData };
	// glm::uvec3 indices;
	// switch (material.indexType)
	// {
	// 	case IndexType::UINT_16:
	// 		{
	// 			uint16_t* idata{ reinterpret_cast<uint16_t*>(material.indices) + primitiveIndex * 3 };
	// 			indices = glm::uvec3{idata[0], idata[1], idata[2]};
	// 		}
	// 		break;
	// 	case IndexType::UINT_32:
	// 		{
	// 			uint32_t* idata{ reinterpret_cast<uint32_t*>(material.indices) + primitiveIndex * 3 };
	// 			indices = glm::uvec3{idata[0], idata[1], idata[2]};
	// 		}
	// 		break;
	// 	default:
	// 		break;
	// }
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

	DirectLightSampleData dlSampleData{};
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
	glm::vec3 rO{ utility::offsetRay(hitPoint, offsetOutside ? hitNormal : -hitNormal) };
	const glm::vec3& lD{ dlSampleData.lightDir };
	if (!dlSampleData.occluded)
	{
		optixTraverse(parameters.traversable,
				{ rO.x, rO.y, rO.z },
				{ lD.x, lD.y, lD.z },
				0.0f,
				dToL - 0.1f,
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

	float2 barycentrics{ optixGetTriangleBarycentrics() };

	float3 vertexObjectData[3];
	optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0.0f, vertexObjectData);
	
	float3 hposO;
	hposO.x = vertexObjectData[0].x * (1.0f - barycentrics.x - barycentrics.y) + vertexObjectData[1].x * barycentrics.x + vertexObjectData[2].x * barycentrics.y;
	hposO.y = vertexObjectData[0].y * (1.0f - barycentrics.x - barycentrics.y) + vertexObjectData[1].y * barycentrics.x + vertexObjectData[2].y * barycentrics.y;
	hposO.z = vertexObjectData[0].z * (1.0f - barycentrics.x - barycentrics.y) + vertexObjectData[1].z * barycentrics.x + vertexObjectData[2].z * barycentrics.y;

	float WFO[12]{};
	optixGetObjectToWorldTransformMatrix(WFO);
	float3 hpos{ hposO.x * WFO[0] + hposO.y * WFO[1] + hposO.z * WFO[2]  + WFO[3], 
				 hposO.x * WFO[4] + hposO.y * WFO[5] + hposO.z * WFO[6]  + WFO[7], 
				 hposO.x * WFO[8] + hposO.y * WFO[9] + hposO.z * WFO[10] + WFO[11] };

	float3 u{ vertexObjectData[2].x - vertexObjectData[0].x, vertexObjectData[2].y - vertexObjectData[0].y, vertexObjectData[2].z - vertexObjectData[0].z };
	float3 v{ vertexObjectData[1].x - vertexObjectData[0].x, vertexObjectData[1].y - vertexObjectData[0].y, vertexObjectData[1].z - vertexObjectData[0].z };
	float3 geometryNormal{u.y * v.z - u.z * v.y,
						  u.z * v.x - u.x * v.z,
						  u.x * v.y - u.y * v.x};
	geometryNormal = optixTransformNormalFromObjectToWorldSpace(geometryNormal);
	float normalizeDiv{ __frsqrt_rn(geometryNormal.x * geometryNormal.x + geometryNormal.y * geometryNormal.y + geometryNormal.z * geometryNormal.z) };
	geometryNormal = {geometryNormal.x * normalizeDiv, geometryNormal.y * normalizeDiv, geometryNormal.z * normalizeDiv};
	uint32_t encGeometryNormal{ utility::octohedral::encodeU32(glm::vec3{geometryNormal.x, geometryNormal.y, geometryNormal.z}) };

	uint32_t primitiveIndex{ optixGetPrimitiveIndex() };
	uint32_t materialIndex{ *reinterpret_cast<uint32_t*>(optixGetSbtDataPointer()) };


	const MaterialData& material{ parameters.materials[materialIndex] };
	uint8_t* attribBuffer{ material.attributeData };
	glm::uvec3 indices;
	switch (material.indexType)
	{
		case IndexType::UINT_16:
			{
				uint16_t* idata{ reinterpret_cast<uint16_t*>(material.indices) + primitiveIndex * 3 };
				indices = glm::uvec3{idata[0], idata[1], idata[2]};
			}
			break;
		case IndexType::UINT_32:
			{
				uint32_t* idata{ reinterpret_cast<uint32_t*>(material.indices) + primitiveIndex * 3 };
				indices = glm::uvec3{idata[0], idata[1], idata[2]};
			}
			break;
		default:
			break;
	}
	// Normals
	constexpr uint32_t attributesStride{ sizeof(glm::vec4) };
	constexpr uint32_t attribOffset{ 0 };
	uint8_t* normalAtt{ attribBuffer + attribOffset };
	glm::vec3 shadingNormal{
		(1.0f - barycentrics.x - barycentrics.y) * (*reinterpret_cast<glm::vec3*>(normalAtt + attributesStride * indices[0]))
		+
		barycentrics.x * (*reinterpret_cast<glm::vec3*>(normalAtt + attributesStride * indices[1]))
		+
		barycentrics.y * (*reinterpret_cast<glm::vec3*>(normalAtt + attributesStride * indices[2])) };

	float3 sNW{ optixTransformNormalFromObjectToWorldSpace(float3{shadingNormal.x, shadingNormal.y, shadingNormal.z}) };
	shadingNormal = glm::normalize(glm::vec3{sNW.x, sNW.y, sNW.z});

	glm::mat3 frameMat{ LocalTransform::genFromLocalMatrixFromNormal(shadingNormal) };
	glm::quat frame{ glm::quat_cast(frameMat) };


	optixSetPayload_0(__float_as_uint(hpos.x));
	optixSetPayload_1(__float_as_uint(hpos.y));
	optixSetPayload_2(__float_as_uint(hpos.z));
	optixSetPayload_3(encGeometryNormal);
	optixSetPayload_4(primitiveIndex);
	optixSetPayload_5(__float_as_uint(barycentrics.x));
	optixSetPayload_6(__float_as_uint(barycentrics.y));
	optixSetPayload_7((static_cast<uint32_t>(LightType::NONE) << 16u) | (materialIndex & 0xFFFF));
	optixSetPayload_8(__float_as_uint(frame.x));
	optixSetPayload_9(__float_as_uint(frame.y));
	optixSetPayload_10(__float_as_uint(frame.z));
	optixSetPayload_11(__float_as_uint(frame.w));
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
		optixSetPayload_7((static_cast<uint32_t>(LightType::DISK) << 16u) | (dl.materialIndex & 0xFFFF));
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
	optixSetPayload_7((static_cast<uint32_t>(LightType::SPHERE) << 16u) | (sl.materialIndex & 0xFFFF));
}

extern "C" __global__ void __miss__miss()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0); 
	optixSetPayload_7(static_cast<uint32_t>(LightType::SKY) << 16);
}

extern "C" __device__ void __direct_callable__DielectricBxDF(const MaterialData& materialData, const DirectLightSampleData& directLightData, const QRNG::State& qrngState, const glm::quat& shadingFrame,
	SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathStateBitfield& stateFlags, float& refractionScale, uint32_t depth)
{
	LocalTransform local{ shadingFrame } ;

	glm::vec3 locWo{ -rD };
	glm::vec3 locLi{ directLightData.lightDir };
	local.toLocal(locWo, locLi);

	glm::vec3 rand{ QRNG::Sobol::sample3D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF) };

	float alpha{ utility::roughnessToAlpha(materialData.mfRoughnessValue) };
	microfacet::Microsurface ms{ .alphaX = alpha, .alphaY = alpha };
	if (static_cast<bool>(stateFlags & PathStateBitfield::REGULARIZED))
		ms.regularize();
	glm::vec3 wo{ locWo };
	const float cosThetaO{ LocalTransform::cosTheta(wo) };
	glm::vec3 wi{};
	glm::vec3 wm{};
	microfacet::ContextOutgoing ctxo{ microfacet::createContextOutgoing(wo) };

	SampledSpectrum eta{ parameters.spectra[materialData.indexOfRefractSpectrumDataIndex].sample(wavelengths) };

	if (alpha < 0.001f || eta[0] == 1.0f)
	{
		SampledSpectrum R{ microfacet::FReal(cosThetaO, eta) };
		float p{ R[0] };
		if (rand.z < p)
		{
			wi = {-wo.x, -wo.y, wo.z};
			bxdfPDF = p;
			float pdfSum{ bxdfPDF };
			for (int i{ 1 }; i < SampledSpectrum::getSampleCount(); ++i)
			{
				if (wavelengths.getPDF()[i] != 0.0f)
					pdfSum += R[i];
			}
			throughputWeight *= R * wavelengths.getActiveCount() * (1.0f / pdfSum);
		}
		else
		{
			wavelengths.terminateAllSecondary();
			float T{ glm::clamp(1.0f - R[0], 0.0f, 1.0f) };
			bool valid;
			float& etaRel{ refractionScale };
			wi = utility::refract(wo, glm::vec3{0.0f, 0.0f, 1.0f}, eta[0], valid, &etaRel);
			if (!valid)
			{
				stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
				return;
			}
			bxdfPDF = cuda::std::fmax(0.0f, 1.0f - p);
			throughputWeight *= T / (etaRel * etaRel) / bxdfPDF;
		}
		stateFlags = wi.z < 0.0f ? stateFlags | PathStateBitfield::INSIDE_OBJECT : stateFlags & (~PathStateBitfield::INSIDE_OBJECT);
		stateFlags = stateFlags | PathStateBitfield::CURRENT_HIT_SPECULAR;
		local.fromLocal(wi);
		rD = wi;
		return;
	}

	if (!directLightData.occluded)
	{
		wi = locLi;
		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
		float cosThetaI{ LocalTransform::cosTheta(wi) };
		const float cosFactor{ cuda::std::fabs(cosThetaI) };
		float t{ cosThetaO * cosThetaI };
		bool reflect{ t > 0.0f };
		float G{ microfacet::G(wi, wo, ctxi, ctxo, ms) };
		float lbxdfPDF;

		SampledSpectrum f{};
#pragma unroll
		for (int i{ 0 }; i < SampledSpectrum::getSampleCount(); ++i)
		{
			if (wavelengths.getPDF()[i] != 0.0f)
			{
				float etaRel{ 1.0f };
				if (!reflect)
					etaRel = cosThetaO > 0.0f ? eta[i] : 1.0f / eta[i];
				wm = wi * etaRel + wo;
				wm = glm::normalize(wm);
				wm = wm.z > 0.0f ? wm : -wm;
				float dotWmWo{ glm::dot(wm, wo) };
				float dotWmWi{ glm::dot(wm, wi) };
				if (t != 0.0f && !(dotWmWo * cosThetaO < 0.0f || dotWmWi * cosThetaI < 0.0f))
				{
					microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };
					const float R{ microfacet::FReal(dotWmWo, eta[i]) };
					const float T{ glm::clamp(1.0f - R, 0.0f, 1.0f) };
					const float pR{ R / (R + T) };
					const float pT{ cuda::std::fmax(0.0f, 1.0f - pR) };
					const float wowmAbsDot{ cuda::std::fabs(dotWmWo) };

					if (reflect)
					{
						f[i] = microfacet::D(wm, ctxm, ms) * R * G
							   / (4.0f * cosThetaO * cosThetaI);
						if (i == 0)
							lbxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wm, wowmAbsDot, ctxo, ctxm, ms) / (4.0f * wowmAbsDot) * pR;
					}
					else
					{
						float t{ dotWmWi + dotWmWo / etaRel };
						float denom{ t * t };
						float dwmdwi{ cuda::std::fabs(dotWmWi) / denom };
						f[i] = microfacet::D(wm, ctxm, ms) * T * G
							   * cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom))
							   / (etaRel * etaRel);
						if (i == 0)
							lbxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wm, wowmAbsDot, ctxo, ctxm, ms) * dwmdwi * pT;
					}
				}
			}
		}
		L += directLightData.spectrumSample * f * throughputWeight
			* (cosFactor * MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, lbxdfPDF) / directLightData.lightSamplePDF);
	}

	SampledSpectrum R{};

	wm = microfacet::VNDF::sample<microfacet::VNDF::SPHERICAL_CAP>(wo, ms, rand);
	microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };

	float dotWmWo{ glm::dot(wo, wm) };
	float absDotWmWo{ cuda::std::fabs(dotWmWo) };
	const float heroR{ microfacet::FReal(dotWmWo, eta[0]) };
	const float heroT{ glm::clamp(1.0f - heroR, 0.0f, 1.0f) };
	const float pR{ heroR };

	SampledSpectrum f;
	float cosFactor;
	float condPDFCount{ 0.0f };
	float condPDFSum{ 0.0f };
	if (rand.z < pR)
	{
		wi = utility::reflect(wo, wm);
		if (wo.z * wi.z <= 0.0f)
		{
			stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
			return;
		}
		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
		const float cosThetaI{ LocalTransform::cosTheta(wi) };
		R[0] = heroR;
		for (int i{ 1 }; i < SampledSpectrum::getSampleCount(); ++i)
			R[i] = microfacet::FReal(dotWmWo, eta[i]);
		f = microfacet::D(wm, ctxm, ms) * R * microfacet::G(wi, wo, ctxi, ctxo, ms)
			/ (4.0f * cosThetaO * cosThetaI);
		float pdfTerm{ microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wm, absDotWmWo, ctxo, ctxm, ms) / (4.0f * absDotWmWo) };
		bxdfPDF = pdfTerm * heroR;
		condPDFCount = wavelengths.getActiveCount();
		condPDFSum = bxdfPDF;
		for (int i{ 1 }; i < SampledSpectrum::getSampleCount(); ++i)
		{
			if (wavelengths.getPDF()[i] != 0.0f)
				condPDFSum += pdfTerm * R[i];
		}
		cosFactor = cuda::std::fabs(cosThetaI);
	}
	else
	{
		bool valid;
		float& etaRel{ refractionScale };
		wi = utility::refract(wo, wm, eta[0], valid, &etaRel);
		if (wo.z * wi.z >= 0.0f || !valid)
		{
			stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
			return;
		}
		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
		const float cosThetaI{ LocalTransform::cosTheta(wi) };
		float dotWmWi{ glm::dot(wm, wi) };
		float t{ dotWmWi + dotWmWo / etaRel };
		float denom{ t * t };
		float dwmdwi{ cuda::std::fabs(dotWmWi) / denom };
		float G{ microfacet::G(wi, wo, ctxi, ctxo, ms) };
		float fh{ microfacet::D(wm, ctxm, ms) * heroT * G
				  * cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom)) };
		const float pT{ cuda::std::fmax(0.0f, 1.0f - pR) };
		bxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wm, absDotWmWo, ctxo, ctxm, ms) * dwmdwi * pT;
		condPDFSum = bxdfPDF;
		cosFactor = cuda::std::fabs(cosThetaI);
		f[0] = fh;
		for (int i{ 1 }; i < SampledSpectrum::getSampleCount(); ++i)
		{
			if (wavelengths.getPDF()[i] != 0.0f)
			{
				etaRel = cosThetaO > 0.0f ? eta[i] : 1.0f / eta[i];
				wm = glm::normalize(wi * etaRel + wo);
				wm = wm.z > 0.0f ? wm : -wm;
				ctxm = microfacet::createContextMicronormal(wm);
				dotWmWo = glm::dot(wm, wo);
				dotWmWi = glm::dot(wm, wi);
				absDotWmWo = cuda::std::fabs(dotWmWo);
				const float secR{ microfacet::FReal(dotWmWo, eta[i]) };
				const float secT{ glm::clamp(1.0f - secR, 0.0f, 1.0f) };
				t = dotWmWi + dotWmWo / etaRel;
				denom = t * t;
				dwmdwi = cuda::std::fabs(dotWmWi) / denom;
				f[i] = microfacet::D(wm, ctxm, ms) * secT * G
					* cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom));
				float pdf{ microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wm, absDotWmWo, ctxo, ctxm, ms) * dwmdwi * secT };
				if (pdf <= 0.0f || (!(dotWmWo * cosThetaO < 0.0f || dotWmWi * cosThetaI < 0.0f)))
					wavelengths.terminateSecondary(i);
				else
					condPDFSum += pdf;
			}
		}
		condPDFCount = wavelengths.getActiveCount();
		f /= (etaRel * etaRel);
	}

	if (bxdfPDF == 0.0f)
	{
		stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
		return;
	}

	throughputWeight *= condPDFCount * (1.0f / (condPDFSum)) * f * cosFactor;

	glm::vec3 locWi{ wi };
	stateFlags = locWi.z < 0.0f ? stateFlags | PathStateBitfield::INSIDE_OBJECT : stateFlags & (~PathStateBitfield::INSIDE_OBJECT);
	local.fromLocal(locWi);
	rD = glm::normalize(locWi);
}
extern "C" __device__ void __direct_callable__ConductorBxDF(const MaterialData& materialData, const DirectLightSampleData& directLightData, const QRNG::State& qrngState, const glm::quat& shadingFrame,
	SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathStateBitfield& stateFlags, float& refractionScale, uint32_t depth)
{
	LocalTransform local{ shadingFrame } ;

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
	microfacet::Microsurface ms{ .alphaX = alpha, .alphaY = alpha };
	if (static_cast<bool>(stateFlags & PathStateBitfield::REGULARIZED))
		ms.regularize();
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
			SampledSpectrum f{ microfacet::D(wm, ctxm, ms) * microfacet::FComplex(wowmAbsDot, eta, k) * microfacet::G(wi, wo, ctxi, ctxo, ms) 
							   / cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi)) };
			float cosFactor{ cuda::std::fabs(LocalTransform::cosTheta(wi)) };
			wm = wm.z > 0.0f ? wm : -wm;
			float lbxdfPDF{ microfacet::VNDF::PDF<microfacet::VNDF::BOUNDED_SPHERICAL_CAP>(wo, wm, wowmAbsDot, ctxo, ctxm, ms) / (4.0f * wowmAbsDot) };
			L += directLightData.spectrumSample * f * cosFactor * throughputWeight
				* MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, lbxdfPDF)
				/ directLightData.lightSamplePDF;
		}
	}

	glm::vec2 rand{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF) };
	wm = microfacet::VNDF::sample<microfacet::VNDF::BOUNDED_SPHERICAL_CAP>(wo, ms, rand);
	microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };

	wi = utility::reflect(wo, wm);
	microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };

	if (wo.z * wi.z <= 0.0f)
	{
		stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
		return;
	}

	const float wowmAbsDot{ cuda::std::fabs(glm::dot(wo, wm)) };

	SampledSpectrum f{ microfacet::D(wm, ctxm, ms) * microfacet::FComplex(wowmAbsDot, eta, k) * microfacet::G(wi, wo, ctxi, ctxo, ms) 
					   / cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi)) };
	float cosFactor{ cuda::std::fabs(LocalTransform::cosTheta(wi)) };
	bxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::BOUNDED_SPHERICAL_CAP>(wo, wm, wowmAbsDot, ctxo, ctxm, ms) / (4.0f * wowmAbsDot);
	throughputWeight *= f * cosFactor / bxdfPDF;

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
		const glm::vec2 sampleCoordinate{ pixelCoordinate + filter::gaussian::sampleDistribution(subsample) };
		const glm::vec2 lensSample{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::LENS) };
		Ray ray;
		if (parameters.cameraState.depthOfFieldEnabled)
		{
			ray = generateThinLensCamera(sampleCoordinate,
					lensSample, parameters.cameraState.focusDistance, parameters.cameraState.appertureSize,
					glm::vec2{resState.invFilmWidth, resState.invFilmHeight}, glm::vec2{resState.camPerspectiveScaleW, resState.camPerspectiveScaleH},
					parameters.cameraState.camU, parameters.cameraState.camV, parameters.cameraState.camW);
		}
		else
		{
			ray = generatePinholeCameraDirection(sampleCoordinate,
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
		uint32_t depth{ 0 };
		do
		{
			uint32_t pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7, pl8, pl9, pl10, pl11;
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
					   pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7, pl8, pl9, pl10, pl11); //Payload

			glm::vec3 hP; //Hit position
			uint32_t encHitGNormal;
			uint32_t primitiveIndex;
			glm::vec2 barycentrics;
			MaterialData* material;
			glm::quat shadingFrame;
			unpackTraceData(parameters, hP, encHitGNormal,
					primitiveIndex, barycentrics,
					lightType, lightIndex, &material, shadingFrame,
					pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7, pl8, pl9, pl10, pl11);

			if (lightType == LightType::SKY)
			{
				/*SampledSpectrum Lo{ sampleSpectrum(wavelengths) };
				float emissionWeight{ 1.0f };
				if (depth != 0 && !(stateFlags & PathStateBitfield::PREVIOUS_HIT_SPECULAR))
					emissionWeight = MIS::powerHeuristic(1, bxdfPDF, 1, lightPDF);
				L += throughputWeight * Lo * emissionWeight;*/
				
				goto breakPath;
			}

			glm::vec3 toHit{ hP - rO };
			L += emittedLightEval(parameters, wavelengths, stateFlags,
					lightType, lightIndex,
					throughputWeight, bxdfPDF,
					depth, rO, rD, toHit.x * toHit.x + toHit.y * toHit.y + toHit.z * toHit.z);

			glm::vec3 hNG{ utility::octohedral::decodeU32(encHitGNormal)}; //Hit normal geometry
			DirectLightSampleData directLightData{ sampledLightEval(parameters, wavelengths,
						hP, hNG,
						QRNG::Sobol::sample3D(qrngState, QRNG::DimensionOffset::LIGHT)) };


			// Launch BxDF evaluation
			resolveAttributes(*material, barycentrics, primitiveIndex, shadingFrame);
			if (lightType != LightType::NONE)
				shadingFrame = glm::quat_cast(LocalTransform::genFromLocalMatrixFromNormal(hNG));
			optixDirectCall<void, 
				const MaterialData&, const DirectLightSampleData&, const QRNG::State&,
				const glm::quat&,
				SampledSpectrum&, SampledWavelengths&, SampledSpectrum&,
				float&, glm::vec3&, PathStateBitfield&, float&, uint32_t>
				(material->bxdfIndexSBT,
				 *material, directLightData, qrngState,
				 shadingFrame,
				 L, wavelengths, throughputWeight,
				 bxdfPDF, rD, stateFlags, refractionScale, depth);

			rO = utility::offsetRay(hP, static_cast<bool>(stateFlags & PathStateBitfield::INSIDE_OBJECT) ? -hNG : hNG); // For nested transmissive objects need to implement some kind of object stack or node system


			qrngState.advanceBounce();
			updateStateFlags(stateFlags);
			terminated = static_cast<bool>(stateFlags & PathStateBitfield::PATH_TERMINATED);

			SampledSpectrum refScaleThroughput{ throughputWeight };
			if (float tMax{ refScaleThroughput.max() * refractionScale }; tMax < 1.0f && depth > 0)
			{
				float q{ cuda::std::fmax(0.0f, 1.0f - tMax) };
				q = q * q;
				if (QRNG::Sobol::sample1D(qrngState, QRNG::DimensionOffset::ROULETTE) < q)
					goto breakPath;
				throughputWeight /= 1.0f - q;
			}
		} while (++depth < parameters.maxPathLength && !terminated);
	breakPath:
		qrngState.advanceSample();

		if (!terminated)
		{
			resolveSample(L, wavelengths.getPDF());
			result += glm::dvec4{color::toRGB(*parameters.sensorSpectralCurveA, *parameters.sensorSpectralCurveB, *parameters.sensorSpectralCurveC,
					wavelengths, L), filter::computeFilterWeight(subsample)};
		}
	} while (++sample < parameters.samplingState.count);
	parameters.renderData[li.y * resState.filmWidth + li.x] = result;
}
