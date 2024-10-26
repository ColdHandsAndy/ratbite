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
	FINISHED              = 1 << 8,
};
ENABLE_ENUM_BITWISE_OPERATORS(PathStateBitfield);

extern "C"
{
	__constant__ LaunchParameters parameters{};
}

struct Path
{
	Ray ray{};
	SampledWavelengths wavelengths{};
	SampledSpectrum L{};
	SampledSpectrum throughput{};
	float refractionScale{};
	float bxdfPDF{};
	PathStateBitfield stateFlags{};
	uint32_t depth{};
};
struct Interaction
{
	glm::vec3 hitPos{};
	LightType lightType{};
	uint16_t lightIndex{};
	MaterialData* material{};
	glm::vec3 geometryNormal{};
	bool hitFromInside{};
	bool skipped{ false };
	union PrimitiveData
	{
		struct Triangle
		{
			uint32_t index{};
			glm::vec3 vertices[3]{};
			glm::vec3 hitPosInterp{};
			glm::vec2 barycentrics{};
			glm::vec3 shadingNormal{};
		} triangle;
	} primitive;
};
struct DirectLightSampleData
{
	SampledSpectrum spectrumSample{};
	glm::vec3 lightDir{};
	float lightSamplePDF{};
	bool occluded{};
};


extern "C" __global__ void __closesthit__triangle()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

	uint32_t materialIndex{ *reinterpret_cast<uint32_t*>(optixGetSbtDataPointer()) };
	uint32_t primitiveIndex{ optixGetPrimitiveIndex() };
	float2 barycentrics{ optixGetTriangleBarycentrics() };

	float3 verticesObj[3];
	optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0.0f, verticesObj);
	
	float WFO[12]{};
	optixGetObjectToWorldTransformMatrix(WFO);
	float OFW[12]{};
	optixGetWorldToObjectTransformMatrix(OFW);

	float3 u{ verticesObj[2].x - verticesObj[0].x, verticesObj[2].y - verticesObj[0].y, verticesObj[2].z - verticesObj[0].z };
	float3 v{ verticesObj[1].x - verticesObj[0].x, verticesObj[1].y - verticesObj[0].y, verticesObj[1].z - verticesObj[0].z };
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

	optixSetPayload_0(__float_as_uint(verticesObj[0].x * WFO[0] + verticesObj[0].y * WFO[1] + verticesObj[0].z * WFO[2]   + WFO[3]));
	optixSetPayload_1(__float_as_uint(verticesObj[0].x * WFO[4] + verticesObj[0].y * WFO[5] + verticesObj[0].z * WFO[6]   + WFO[7]));
	optixSetPayload_2(__float_as_uint(verticesObj[0].x * WFO[8] + verticesObj[0].y * WFO[9] + verticesObj[0].z * WFO[10]  + WFO[11]));
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
	optixSetPayload_17(__float_as_uint(verticesObj[1].x * WFO[0] + verticesObj[1].y * WFO[1] + verticesObj[1].z * WFO[2]   + WFO[3]));
	optixSetPayload_18(__float_as_uint(verticesObj[1].x * WFO[4] + verticesObj[1].y * WFO[5] + verticesObj[1].z * WFO[6]   + WFO[7]));
	optixSetPayload_19(__float_as_uint(verticesObj[1].x * WFO[8] + verticesObj[1].y * WFO[9] + verticesObj[1].z * WFO[10]  + WFO[11]));
	optixSetPayload_20(__float_as_uint(verticesObj[2].x * WFO[0] + verticesObj[2].y * WFO[1] + verticesObj[2].z * WFO[2]   + WFO[3]));
	optixSetPayload_21(__float_as_uint(verticesObj[2].x * WFO[4] + verticesObj[2].y * WFO[5] + verticesObj[2].z * WFO[6]   + WFO[7]));
	optixSetPayload_22(__float_as_uint(verticesObj[2].x * WFO[8] + verticesObj[2].y * WFO[9] + verticesObj[2].z * WFO[10]  + WFO[11]));
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

	if (static_cast<bool>(stateFlags & PathStateBitfield::REGULARIZED))
		alphaMS.regularize();
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
		if (wm.z < 0.0f)
			wm = -wm;
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
	float sin2ThetaI{ cuda::std::fmax(0.0f, 1.0f - wowmAbsDot * wowmAbsDot) };
	float sin2ThetaT{ sin2ThetaI / (eta * eta) };
	if (sin2ThetaT >= 1.0f)
		FDielectric = 1.0f;

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
			wi = utility::refract(wo, wm, eta);
			refractionScale *= eta;
			if (wo.z * wi.z >= 0.0f)
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


CU_DEVICE CU_INLINE void unpackInteractionData(const LaunchParameters& params, uint32_t* pl,
		Path& path, Interaction& interaction, glm::mat3& worldFromObjectNormal)
{
	interaction.geometryNormal = utility::octohedral::decodeU32(pl[3]);
	uint32_t matIndex{ pl[7] & 0xFFFF };
	interaction.material = params.materials + matIndex;
	
	path.stateFlags |= static_cast<PathStateBitfield>((pl[7] >> 16) & 0xFF);
	interaction.lightType = static_cast<LightType>(pl[7] >> 24);
	if (static_cast<bool>(path.stateFlags & PathStateBitfield::TRIANGULAR_GEOMETRY))
	{
		interaction.primitive.triangle.index = pl[4];
		glm::vec2& barycentrics{ interaction.primitive.triangle.barycentrics };
		barycentrics = { __uint_as_float(pl[5]), __uint_as_float(pl[6]) };
		glm::vec3& vert0{ interaction.primitive.triangle.vertices[0] };
		glm::vec3& vert1{ interaction.primitive.triangle.vertices[1] };
		glm::vec3& vert2{ interaction.primitive.triangle.vertices[2] };
		vert0 = glm::vec3{ __uint_as_float(pl[0]), __uint_as_float(pl[1]), __uint_as_float(pl[2]) };
		vert1 = glm::vec3{ __uint_as_float(pl[17]), __uint_as_float(pl[18]), __uint_as_float(pl[19]) };
		vert2 = glm::vec3{ __uint_as_float(pl[20]), __uint_as_float(pl[21]), __uint_as_float(pl[22]) };
		glm::vec3& hitPos{ interaction.hitPos };
		hitPos.x = vert0.x * (1.0f - barycentrics.x - barycentrics.y) + vert1.x * barycentrics.x + vert2.x * barycentrics.y;
		hitPos.y = vert0.y * (1.0f - barycentrics.x - barycentrics.y) + vert1.y * barycentrics.x + vert2.y * barycentrics.y;
		hitPos.z = vert0.z * (1.0f - barycentrics.x - barycentrics.y) + vert1.z * barycentrics.x + vert2.z * barycentrics.y;
	}
	else
	{
		interaction.hitPos = glm::vec3{ __uint_as_float(pl[0]), __uint_as_float(pl[1]), __uint_as_float(pl[2]) };
		interaction.lightIndex = pl[4];
	}

	interaction.primitive.triangle.hitPosInterp = interaction.hitPos;
	interaction.hitFromInside = -glm::dot(interaction.geometryNormal, path.ray.d) < 0.0f;

	worldFromObjectNormal[0][0] = __uint_as_float(pl[8]);
	worldFromObjectNormal[1][0] = __uint_as_float(pl[9]);
	worldFromObjectNormal[2][0] = __uint_as_float(pl[10]);
	worldFromObjectNormal[0][1] = __uint_as_float(pl[11]);
	worldFromObjectNormal[1][1] = __uint_as_float(pl[12]);
	worldFromObjectNormal[2][1] = __uint_as_float(pl[13]);
	worldFromObjectNormal[0][2] = __uint_as_float(pl[14]);
	worldFromObjectNormal[1][2] = __uint_as_float(pl[15]);
	worldFromObjectNormal[2][2] = __uint_as_float(pl[16]);
}
CU_DEVICE CU_INLINE void updateStateFlags(PathStateBitfield& stateFlags)
{
	PathStateBitfield excludeFlags{ PathStateBitfield::CURRENT_HIT_SPECULAR | PathStateBitfield::TRIANGULAR_GEOMETRY | PathStateBitfield::RIGHT_HANDED_FRAME | PathStateBitfield::RAY_REFRACTED };
	PathStateBitfield includeFlags{ static_cast<bool>(stateFlags & PathStateBitfield::CURRENT_HIT_SPECULAR) ?
		PathStateBitfield::PREVIOUS_HIT_SPECULAR : PathStateBitfield::REGULARIZED };
	if (static_cast<bool>(stateFlags & (PathStateBitfield::RAY_REFRACTED)))
		includeFlags |= PathStateBitfield::REFRACTION_HAPPENED;

	stateFlags = (stateFlags & (~excludeFlags)) | includeFlags;
}

extern "C" __global__ void __raygen__main()
{
	const uint3 li{ optixGetLaunchIndex() };
	const glm::vec2 pixelCoordinate{ static_cast<float>(li.x), static_cast<float>(li.y) };
	const uint32_t renderDataIndex{ li.y * parameters.resolutionState.filmWidth + li.x };

	glm::dvec4 result{ parameters.samplingState.offset != 0 ? parameters.renderData[renderDataIndex] : glm::dvec4{0.0} };
	uint32_t sample{ 0 };
	QRNG::State qrngState{ parameters.samplingState.offset, QRNG::getPixelHash(li.x, li.y) };
	// Sample loop (Processing multiple samples and store them)
	do
	{
		//Defining the path state
		Path path{
			.wavelengths = SampledWavelengths::sampleVisible(QRNG::Sobol::sample1D(qrngState, QRNG::DimensionOffset::WAVELENGTH)),
			.L = { 0.0f },
			.throughput = { 1.0f },
			.refractionScale = 1.0f,
			.bxdfPDF = 1.0f,
			.stateFlags = PathStateBitfield::NO_FLAGS,
			.depth = 0, };
		{
			// Generate camera ray
			const glm::vec2 subsample{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::FILTER) };
			const glm::vec2 subsampleCoordinate{ pixelCoordinate + filter::gaussian::sampleDistribution(subsample) };
			Ray& ray{ path.ray };
			if (parameters.cameraState.depthOfFieldEnabled)
			{
				const LaunchParameters::ResolutionState& resState{ parameters.resolutionState };
				const glm::vec2 lensSample{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::LENS) };
				ray = generateThinLensCamera(subsampleCoordinate,
						lensSample, parameters.cameraState.focusDistance, parameters.cameraState.appertureSize,
						glm::vec2{resState.invFilmWidth, resState.invFilmHeight}, glm::vec2{resState.perspectiveScaleW, resState.perspectiveScaleH},
						parameters.cameraState.camU, parameters.cameraState.camV, parameters.cameraState.camW);
			}
			else
			{
				const LaunchParameters::ResolutionState& resState{ parameters.resolutionState };
				ray = generatePinholeCameraDirection(subsampleCoordinate,
						glm::vec2{resState.invFilmWidth, resState.invFilmHeight}, glm::vec2{resState.perspectiveScaleW, resState.perspectiveScaleH},
						parameters.cameraState.camU, parameters.cameraState.camV, parameters.cameraState.camW);
			}
		}

		// Trace loop (Processing the path and gathering transfered radiance)
		do
		{
			uint32_t pl[23];
			optixTrace(OPTIX_PAYLOAD_TYPE_ID_0, //Payload type
					   parameters.traversable, //Traversable handle
					   { path.ray.o.x, path.ray.o.y, path.ray.o.z }, //Ray origin
					   { path.ray.d.x, path.ray.d.y, path.ray.d.z }, //Ray direction
					   0.0f, //Min "t"
					   FLT_MAX, //Max "t"
					   0.0f, //Time
					   0xFF, //Visibility mask
					   0, //Flags
					   0, //SBT offset
					   1, //SBT stride
					   0, //SBT miss program index
					   pl[0], pl[1], pl[2],
					   pl[3], pl[4], pl[5], pl[6], pl[7],
					   pl[8], pl[9], pl[10], pl[11], pl[12], pl[13], pl[14], pl[15], pl[16],
					   pl[17], pl[18], pl[19], pl[20], pl[21], pl[22]);

			// Unpack path interaction data
			Interaction interaction{};
			glm::mat3 worldFromObjectNormal{};
			unpackInteractionData(parameters, pl,
					path, interaction, worldFromObjectNormal);

			// Hit emission estimation
			if (interaction.lightType != LightType::NONE)
			{
				glm::vec3 toHit{ interaction.hitPos - path.ray.o };
				float sqrdDistToLight{ toHit.x * toHit.x + toHit.y * toHit.y + toHit.z * toHit.z };

				float lightStructurePDF{ 1.0f / (parameters.lights.lightCount + (parameters.envMap.enabled ? 1.0f : 0.0f)) };
				float lightPDF{};
				SampledSpectrum Le{};
				bool noEmission{ false };

				switch (interaction.lightType)
				{
					case LightType::DISK:
						{
							const DiskLightData& disk{ parameters.lights.disks[interaction.lightIndex] };
							glm::vec3 norm{ glm::mat3_cast(disk.frame)[2] };
							float lCos{ -glm::dot(path.ray.d, norm) };
							float surfacePDF{ 1.0f / (glm::pi<float>() * disk.radius * disk.radius) };
							lightPDF = surfacePDF * sqrdDistToLight / lCos;
							float lightPowerScale{ disk.powerScale * (lCos > 0.0f ? 1.0f : 0.0f) };
							uint32_t emissionSpectrumDataIndex{ parameters.materials[disk.materialIndex].emissionSpectrumDataIndex };
							Le = parameters.spectra[emissionSpectrumDataIndex].sample(path.wavelengths) * lightPowerScale;
						}
						break;
					case LightType::SPHERE:
						{
							const SphereLightData& sphere{ parameters.lights.spheres[interaction.lightIndex] };
							lightPDF = sampling::sphere::pdfUniformSolidAngle(path.ray.o, sphere.position, sphere.radius);
							float lightPowerScale{ sphere.powerScale };
							uint32_t emissionSpectrumDataIndex{ parameters.materials[sphere.materialIndex].emissionSpectrumDataIndex };
							Le = parameters.spectra[emissionSpectrumDataIndex].sample(path.wavelengths) * lightPowerScale;
						}
						break;
					case LightType::SKY:
						{
							if (parameters.envMap.enabled)
							{
								float phi{ cuda::std::atan2(path.ray.d.x, path.ray.d.z) };
								float theta{ cuda::std::acos(path.ray.d.y) };
								float4 skyMap{ tex2D<float4>(parameters.envMap.environmentTexture,
										0.5f + phi / (2.0f * glm::pi<float>()), theta / glm::pi<float>()) };
								glm::vec3 skyColor{ skyMap.x, skyMap.y, skyMap.z };
								float surfacePDF{ 1.0f / (4.0f * glm::pi<float>()) };
								lightPDF = surfacePDF * ((skyColor.x + skyColor.y + skyColor.z) / 3.0f)
									* cuda::std::sin(theta) // Applying Cartesian to spherical Jacobian
									/ parameters.envMap.integral;
								Le = color::RGBtoSpectrum(skyColor, path.wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) * 0.01f;
								path.stateFlags |= PathStateBitfield::FINISHED;
							}
							else
							{
								path.stateFlags |= PathStateBitfield::FINISHED;
								noEmission = true;
							}
						}
						break;
					default:
						break;
				}

				if (!noEmission)
				{
					lightPDF *= lightStructurePDF;

					float emissionWeight{};
					if (path.depth == 0 || static_cast<bool>(path.stateFlags & PathStateBitfield::PREVIOUS_HIT_SPECULAR))
						emissionWeight = 1.0f;
					else
						emissionWeight = MIS::powerHeuristic(1, path.bxdfPDF, 1, lightPDF);

					path.L += path.throughput * Le * emissionWeight;
				}
			}

			if (static_cast<bool>(path.stateFlags & PathStateBitfield::FINISHED))
				goto finishPath;

			// Extract vertex and material data
			LocalTransform localTransform{};
			microsurface::Surface surface{};
			if (static_cast<bool>(path.stateFlags & PathStateBitfield::TRIANGULAR_GEOMETRY))
			{
				glm::vec2 texC1;
				glm::vec2 texC2;
				glm::vec3 normal;
				glm::vec3 tangent;
				bool flipTangent{ false };

				// Unpack vertex attributes
				{
					glm::vec2 bary{ interaction.primitive.triangle.barycentrics.x, interaction.primitive.triangle.barycentrics.y };
					float baryWeights[3]{ 1.0f - bary.x - bary.y, bary.x, bary.y };
					uint32_t indices[3];
					switch (interaction.material->indexType)
					{
						case IndexType::UINT_16:
							{
								uint16_t* idata{ reinterpret_cast<uint16_t*>(interaction.material->indices)
									+ interaction.primitive.triangle.index * 3 };
								indices[0] = idata[0];
								indices[1] = idata[1];
								indices[2] = idata[2];
							}
							break;
						case IndexType::UINT_32:
							{
								uint32_t* idata{ reinterpret_cast<uint32_t*>(interaction.material->indices)
									+ interaction.primitive.triangle.index * 3 };
								indices[0] = idata[0];
								indices[1] = idata[1];
								indices[2] = idata[2];
							}
							break;
						default:
							break;
					}
					uint8_t* attribBuffer{ interaction.material->attributeData };
					uint32_t attributesStride{ interaction.material->attributeStride };

					uint8_t* attr;

					// attr = material.attributeData + material.colorOffset;
					// if (static_cast<bool>(material.attributes & MaterialData::AttributeTypeBitfield::COLOR))
					// 	;
					if (static_cast<bool>(interaction.material->attributes & (MaterialData::AttributeTypeBitfield::NORMAL | MaterialData::AttributeTypeBitfield::FRAME)))
					{
						glm::vec3 shadingNormal{};
						glm::vec3 shadingTangent{};
						glm::vec3 normals[3]{};
						if (static_cast<bool>(interaction.material->attributes & MaterialData::AttributeTypeBitfield::FRAME))
						{
							attr = attribBuffer + interaction.material->frameOffset;

							glm::quat frames[3]{
								*reinterpret_cast<glm::quat*>(attr + attributesStride * indices[0]),
								*reinterpret_cast<glm::quat*>(attr + attributesStride * indices[1]),
								*reinterpret_cast<glm::quat*>(attr + attributesStride * indices[2]), };
							flipTangent = cuda::std::signbit(frames[0].w);
							glm::mat3 frameMats[3]{
								glm::mat3_cast(frames[0]),
								glm::mat3_cast(frames[1]),
								glm::mat3_cast(frames[2]), };
							normals[0] = frameMats[0][1];
							normals[1] = frameMats[1][1];
							normals[2] = frameMats[2][1];
							glm::vec3 tangents[3]{ frameMats[0][0], frameMats[1][0], frameMats[2][0] };
							shadingNormal =
								baryWeights[0] * normals[0]
								+
								baryWeights[1] * normals[1]
								+
								baryWeights[2] * normals[2];
							shadingTangent =
								baryWeights[0] * tangents[0]
								+
								baryWeights[1] * tangents[1]
								+
								baryWeights[2] * tangents[2];
						}
						else
						{
							attr = attribBuffer + interaction.material->normalOffset;

							normals[0] = utility::octohedral::decode(*reinterpret_cast<glm::vec2*>(attr + attributesStride * indices[0]));
							normals[1] = utility::octohedral::decode(*reinterpret_cast<glm::vec2*>(attr + attributesStride * indices[1]));
							normals[2] = utility::octohedral::decode(*reinterpret_cast<glm::vec2*>(attr + attributesStride * indices[2]));
							shadingNormal = glm::vec3{
								baryWeights[0] * normals[0]
									+
									baryWeights[1] * normals[1]
									+
									baryWeights[2] * normals[2] };
							float sign{ cuda::std::copysignf(1.0f, shadingNormal.z) };
							float a{ -1.0f / (sign + shadingNormal.z) };
							float b{ shadingNormal.x * shadingNormal.y * a };
							shadingTangent = glm::vec3(1.0f + sign * (shadingNormal.x * shadingNormal.x) * a, sign * b, -sign * shadingNormal.x);
						}

						// Offset hit position according to shading normals https://jo.dreggn.org/home/2021_terminator.pdf
						const glm::vec3& A{ interaction.primitive.triangle.vertices[0] };
						const glm::vec3& B{ interaction.primitive.triangle.vertices[1] };
						const glm::vec3& C{ interaction.primitive.triangle.vertices[2] };
						const glm::vec3& P{ interaction.hitPos };
						const glm::vec3 nA{ glm::normalize(worldFromObjectNormal * normals[0]) };
						const glm::vec3 nB{ glm::normalize(worldFromObjectNormal * normals[1]) };
						const glm::vec3 nC{ glm::normalize(worldFromObjectNormal * normals[2]) };
						const float u{ baryWeights[0] };
						const float v{ baryWeights[1] };
						const float w{ baryWeights[2] };
						glm::vec3 tmpu = P - A, tmpv = P - B, tmpw = P - C;
						float dotu = cuda::std::fmin(0.0f, glm::dot(tmpu, nA));
						float dotv = cuda::std::fmin(0.0f, glm::dot(tmpv, nB));
						float dotw = cuda::std::fmin(0.0f, glm::dot(tmpw, nC));
						tmpu -= dotu * nA;
						tmpv -= dotv * nB;
						tmpw -= dotw * nC;
						glm::vec3 Pp = P + u * tmpu + v * tmpv + w * tmpw;
						interaction.primitive.triangle.hitPosInterp = Pp;

						normal = glm::normalize(worldFromObjectNormal * shadingNormal);
						glm::mat3 vectorTransform{ glm::inverse(glm::transpose(worldFromObjectNormal)) };
						tangent = glm::normalize(vectorTransform * shadingTangent);
					}
					else
					{
						normal = interaction.geometryNormal;
						float sign{ cuda::std::copysignf(1.0f, interaction.geometryNormal.z) };
						float a{ -1.0f / (sign + interaction.geometryNormal.z) };
						float b{ interaction.geometryNormal.x * interaction.geometryNormal.y * a };
						tangent = { 1.0f + sign * (interaction.geometryNormal.x * interaction.geometryNormal.x) * a,
								sign * b,
								-sign * interaction.geometryNormal.x };
					}
					if (static_cast<bool>(interaction.material->attributes & MaterialData::AttributeTypeBitfield::TEX_COORD_1))
					{
						attr = attribBuffer + interaction.material->texCoord1Offset;
						texC1 =
							baryWeights[0] * (*reinterpret_cast<glm::vec2*>(attr + indices[0] * attributesStride)) +
							baryWeights[1] * (*reinterpret_cast<glm::vec2*>(attr + indices[1] * attributesStride)) +
							baryWeights[2] * (*reinterpret_cast<glm::vec2*>(attr + indices[2] * attributesStride));
					}
					if (static_cast<bool>(interaction.material->attributes & MaterialData::AttributeTypeBitfield::TEX_COORD_2))
					{
						attr = attribBuffer + interaction.material->texCoord2Offset;
						texC2 =
							baryWeights[0] * (*reinterpret_cast<glm::vec2*>(attr + indices[0] * attributesStride)) +
							baryWeights[1] * (*reinterpret_cast<glm::vec2*>(attr + indices[1] * attributesStride)) +
							baryWeights[2] * (*reinterpret_cast<glm::vec2*>(attr + indices[2] * attributesStride));
					}
				}
				// Unpack textures
				{
					glm::vec3 bitangent{ glm::cross(tangent, normal) * (flipTangent ? -1.0f : 1.0f) };
					glm::mat3 frame{ bitangent, tangent, normal };
					glm::vec3 n{};
					if (static_cast<bool>(interaction.material->textures & MaterialData::TextureTypeBitfield::NORMAL))
					{
						float2 uv{ interaction.material->nmTexCoordSetIndex ? float2{texC2.x, texC2.y} : float2{texC1.x, texC1.y} };
						float4 nm{ tex2D<float4>(interaction.material->normalTexture, uv.x, uv.y) };
						n = glm::normalize(glm::vec3{nm.y * 2.0f - 1.0f, nm.x * 2.0f - 1.0f, nm.z * 2.0f - 1.0f});
					}
					else
						n = glm::vec3{0.0f, 0.0f, 1.0f};
					n = frame * n;

					// Shading normal correction
					float dotNsRi{ glm::dot(n, path.ray.d) };
					float dotNgRi{ glm::dot(interaction.geometryNormal, path.ray.d) };
					bool incFromOutside{ dotNsRi > 0.0f && dotNgRi < 0.0f };
					bool incFromInside{ dotNsRi < 0.0f && dotNgRi > 0.0f };
					constexpr float correctionBias{ 0.001f };
					if (incFromOutside || incFromInside)
						n = glm::normalize(n - (path.ray.d * (dotNsRi + (incFromOutside ? correctionBias : -correctionBias))));

					glm::vec3 shadingBitangent{ glm::normalize(glm::cross(tangent, n)) };
					glm::vec3 shadingTangent{ glm::normalize(glm::cross(n, shadingBitangent)) };
					shadingBitangent *= (flipTangent ? -1.0f : 1.0f);
					frame = glm::mat3{shadingBitangent, shadingTangent, n};
					localTransform = LocalTransform{frame};
					interaction.primitive.triangle.shadingNormal = n;

					bool bcTexture{ static_cast<bool>(interaction.material->textures & MaterialData::TextureTypeBitfield::BASE_COLOR) };
					bool bcFactor{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::BASE_COLOR) };
					if (bcTexture)
					{
						float2 uv{ interaction.material->nmTexCoordSetIndex ? float2{texC2.x, texC2.y} : float2{texC1.x, texC1.y} };
						float4 bcTexData{ tex2D<float4>(interaction.material->baseColorTexture, uv.x, uv.y) };
						surface.base.color = glm::vec3{bcTexData.x, bcTexData.y, bcTexData.z};
						if (bcFactor)
							surface.base.color
								*= glm::vec3{interaction.material->baseColorFactor[0], interaction.material->baseColorFactor[1], interaction.material->baseColorFactor[2]};
						bool cutoff{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::CUTOFF) };
						if (bcTexData.w < interaction.material->alphaCutoff && cutoff)
							interaction.skipped = true;
					}
					else if (bcFactor)
						surface.base.color
							= glm::vec3{interaction.material->baseColorFactor[0], interaction.material->baseColorFactor[1], interaction.material->baseColorFactor[2]};

					bool mrTexture{ static_cast<bool>(interaction.material->textures & MaterialData::TextureTypeBitfield::MET_ROUGH) };
					bool metFactor{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::METALNESS) };
					bool roughFactor{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::ROUGHNESS) };
					if (mrTexture)
					{
						float2 uv{ interaction.material->mrTexCoordSetIndex ? float2{texC2.x, texC2.y} : float2{texC1.x, texC1.y} };
						float4 mrTexData{ tex2D<float4>(interaction.material->pbrMetalRoughnessTexture, uv.x, uv.y) };
						surface.base.metalness = mrTexData.z * (metFactor ? interaction.material->metalnessFactor : 1.0f);
						surface.specular.roughness = mrTexData.y * (roughFactor ? interaction.material->roughnessFactor : 1.0f);
					}
					else
					{
						surface.base.metalness = metFactor ? interaction.material->metalnessFactor : surface.base.metalness;
						surface.specular.roughness = roughFactor ? interaction.material->roughnessFactor : surface.specular.roughness;
					}

					bool trTexture{ static_cast<bool>(interaction.material->textures & MaterialData::TextureTypeBitfield::TRANSMISSION) };
					bool trFactor{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::TRANSMISSION) };
					if (trTexture)
					{
						float2 uv{ interaction.material->trTexCoordSetIndex ? float2{texC2.x, texC2.y} : float2{texC1.x, texC1.y} };
						float trTexData{ tex2D<float>(interaction.material->transmissionTexture, uv.x, uv.y) };
						surface.transmission.weight = trTexData;
					}
					else
					{
						surface.transmission.weight = trFactor ? interaction.material->transmissionFactor : surface.transmission.weight;
					}
					surface.specular.ior = interaction.material->ior;
				}
			}
			else
			{
				localTransform = LocalTransform{interaction.geometryNormal};
			}

			// Light sampling and BxDF evaluation
			if (!interaction.skipped)
			{
				// Next event estimation data
				DirectLightSampleData directLightData{};
				if (parameters.lights.lightCount == 0.0f && !parameters.envMap.enabled)
				{
					directLightData.occluded = true;
				}
				else
				{
					glm::vec3 rand{ QRNG::Sobol::sample3D(qrngState, QRNG::DimensionOffset::LIGHT) };

					// Sample light structure
					LightType type{};
					uint16_t index{};
					float lightStructurePDF{ 1.0f / (parameters.lights.lightCount + (parameters.envMap.enabled ? 1.0f : 0.0f)) };
					uint16_t sampledLightIndex{ static_cast<uint16_t>((parameters.lights.lightCount + (parameters.envMap.enabled ? 1.0f : 0.0f) - 0.0001f) * rand.z) };
					if (sampledLightIndex != static_cast<uint16_t>(parameters.lights.lightCount + 0.001f))
					{
						uint32_t lightC{ 0 };
						for (int i{ 0 }; i < KSampleableLightCount; ++i)
						{
							lightC += parameters.lights.orderedCount[i];
							if (sampledLightIndex < lightC)
							{
								type = KOrderedTypes[i];
								index = sampledLightIndex - (lightC - parameters.lights.orderedCount[i]);
								break;
							}
						}
					}
					else
					{
						type = LightType::SKY;
					}

					// Take the light sample
					glm::vec3 lightRayOrigin{};
					float dToL{};
					float lightPDF{};
					bool triangularGeometry{ static_cast<bool>(path.stateFlags & PathStateBitfield::TRIANGULAR_GEOMETRY) };
					bool surfaceCanTransmit{ surface.transmission.weight != 0.0f };
					switch (type)
					{
						case LightType::DISK:
							{
								const DiskLightData& disk{ parameters.lights.disks[index] };
								directLightData.spectrumSample
									= parameters.spectra[parameters.materials[disk.materialIndex].emissionSpectrumDataIndex].sample(path.wavelengths) * disk.powerScale;
								glm::mat3 matframe{ glm::mat3_cast(disk.frame) };
								glm::vec3 lSmplPos{ disk.position + sampling::disk::sampleUniform3D(glm::vec2{rand.x, rand.y}, matframe) * disk.radius };

								const glm::vec3& gn{ interaction.geometryNormal };
								const glm::vec3& sn{ triangularGeometry ? interaction.primitive.triangle.shadingNormal : gn };
								const glm::vec3 sr{ lSmplPos - interaction.hitPos };
								const bool inSample{ glm::dot(sn, sr) < 0.0f };
								if ((inSample && !surfaceCanTransmit) || (inSample && (glm::dot(gn, sr) > 0.0f)))
									directLightData.occluded = true;
								lightRayOrigin = utility::offsetPoint(
										inSample || !triangularGeometry ? interaction.hitPos : interaction.primitive.triangle.hitPosInterp,
										inSample ? -gn : gn);

								glm::vec3 rToLight{ lSmplPos - lightRayOrigin };
								float sqrdToLight{ rToLight.x * rToLight.x + rToLight.y * rToLight.y + rToLight.z * rToLight.z };
								dToL = cuda::std::sqrtf(sqrdToLight);
								directLightData.lightDir = rToLight / dToL;
								float lCos{ -glm::dot(matframe[2], directLightData.lightDir) };

								directLightData.occluded = lCos <= 0.0f;

								float surfacePDF{ 1.0f / (glm::pi<float>() * disk.radius * disk.radius) };
								lightPDF = surfacePDF * sqrdToLight / lCos;
							}
							break;
						case LightType::SPHERE:
							{
								const SphereLightData& sphere{ parameters.lights.spheres[index] };
								directLightData.spectrumSample = parameters.spectra[parameters.materials[sphere.materialIndex].emissionSpectrumDataIndex].sample(path.wavelengths) * sphere.powerScale;
								glm::vec3 lSmplPos{ sampling::sphere::sampleUniformWorldSolidAngle(glm::vec2{rand.x, rand.y}, interaction.hitPos, sphere.position, sphere.radius, lightPDF) };

								const glm::vec3& gn{ interaction.geometryNormal };
								const glm::vec3& sn{ triangularGeometry ? interaction.primitive.triangle.shadingNormal : gn };
								const glm::vec3 sr{ lSmplPos - interaction.hitPos };
								const bool inSample{ glm::dot(sn, sr) < 0.0f };
								if ((inSample && !surfaceCanTransmit) || (inSample && (glm::dot(gn, sr) > 0.0f)))
									directLightData.occluded = true;
								lightRayOrigin = utility::offsetPoint(
										inSample || !triangularGeometry ? interaction.hitPos : interaction.primitive.triangle.hitPosInterp,
										inSample ? -gn : gn);

								glm::vec3 rToLight{ lSmplPos - lightRayOrigin };
								dToL = glm::length(rToLight);
								directLightData.lightDir = rToLight / dToL;

								if (lightPDF == 0.0f || dToL <= 0.0f)
									directLightData.occluded = true;
							}
							break;
						case LightType::SKY:
							{
								// Importance sampling environment map with interpolated inverted CDF indices
								glm::vec2 fullC{
									cuda::std::fmin(rand.x * (parameters.envMap.width - 1.0f), parameters.envMap.width - 1.0001f),
									cuda::std::fmin(rand.y * (parameters.envMap.height - 1.0f), parameters.envMap.height - 1.0001f) };
								glm::vec2 floorC{ glm::floor(fullC) };
								glm::vec2 fractC{ fullC - floorC };
								glm::uvec2 floorCI{ floorC };

								glm::vec2 cdfIndicesM{
									parameters.envMap.marginalCDFIndices[floorCI.y],
									parameters.envMap.marginalCDFIndices[floorCI.y + 1] };
								glm::vec2 cdfIndicesC{
									parameters.envMap.conditionalCDFIndices[static_cast<uint32_t>(cdfIndicesM.x * parameters.envMap.width) + floorCI.x],
									parameters.envMap.conditionalCDFIndices[static_cast<uint32_t>(cdfIndicesM.x * parameters.envMap.width) + floorCI.x + 1] };

								glm::vec2 impSample{
									glm::mix(cdfIndicesC.x, cdfIndicesC.y, fractC.x) * (1.0f / parameters.envMap.width),
										glm::mix(cdfIndicesM.x, cdfIndicesM.y, fractC.y) * (1.0f / parameters.envMap.height) };

								float phi{ 2.0f * glm::pi<float>() * impSample.x };
								float theta{ glm::pi<float>() * impSample.y };
								directLightData.lightDir = glm::vec3{
									-cuda::std::sin(theta) * cuda::std::sin(phi),
										cuda::std::cos(theta),
										-cuda::std::sin(theta) * cuda::std::cos(phi) };
								float4 rgbSample{ tex2D<float4>(parameters.envMap.environmentTexture, impSample.x, impSample.y) };
								directLightData.spectrumSample = color::RGBtoSpectrum(glm::vec3{rgbSample.x, rgbSample.y, rgbSample.z},
										path.wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) * 0.01f;
								float surfacePDF{ 1.0f / (4.0f * glm::pi<float>()) };
								lightPDF = surfacePDF * ((rgbSample.x + rgbSample.y + rgbSample.z) / 3.0f)
									* cuda::std::sin(theta) // Applying Cartesian to spherical Jacobian
									/ parameters.envMap.integral;
								dToL = FLT_MAX;
								lightRayOrigin = utility::offsetPoint(
										!triangularGeometry ? interaction.hitPos : interaction.primitive.triangle.hitPosInterp,
										interaction.geometryNormal);
							}
							break;
						default:
							break;
					}
					directLightData.lightSamplePDF = lightPDF * lightStructurePDF;

					const glm::vec3& rO{ lightRayOrigin };
					const glm::vec3& lD{ directLightData.lightDir };
					if (!directLightData.occluded)
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
						directLightData.occluded = optixHitObjectIsHit();
					}
				}

				// BxDF evaluation
				optixDirectCall<void, 
					const MaterialData&, const DirectLightSampleData&, const QRNG::State&,
					const LocalTransform&, const microsurface::Surface&,
					SampledSpectrum&, SampledWavelengths&, SampledSpectrum&,
					float&, glm::vec3&, PathStateBitfield&, float&>
						(interaction.material->bxdfIndexSBT,
						 *interaction.material, directLightData, qrngState,
						 localTransform, surface,
						 path.L, path.wavelengths, path.throughput,
						 path.bxdfPDF, path.ray.d, path.stateFlags, path.refractionScale);
			}

			// Russian roulette
			if (float tMax{ path.throughput.max() * path.refractionScale }; tMax < 1.0f && path.depth > 0)
			{
				float q{ cuda::std::fmax(0.0f, 1.0f - tMax) };
				q = q * q;
				if (QRNG::Sobol::sample1D(qrngState, QRNG::DimensionOffset::ROULETTE) < q)
					goto finishPath;
				path.throughput /= 1.0f - q;
			}

			// Offset new ray origin and correct direction
			{
				const bool refracted{ static_cast<bool>(path.stateFlags & PathStateBitfield::RAY_REFRACTED) };

				const bool triangularGeometry{ static_cast<bool>(path.stateFlags & PathStateBitfield::TRIANGULAR_GEOMETRY) };
				const bool in{ refracted || interaction.skipped };
				const glm::vec3& chosenOrigin{ (in == interaction.hitFromInside) && triangularGeometry
					? interaction.primitive.triangle.hitPosInterp : interaction.hitPos };
				const glm::vec3& chosenOffset{ in == interaction.hitFromInside
					? interaction.geometryNormal : -interaction.geometryNormal };
				path.ray.o = utility::offsetPoint(chosenOrigin, chosenOffset);

				const float dotNgRo{ glm::dot(interaction.geometryNormal, path.ray.d) };
				constexpr float correctionBias{ 0.001f };
				if (dotNgRo > 0.0f
					&&
					(interaction.hitFromInside != refracted)
					&&
					!interaction.skipped)
					path.ray.d = glm::normalize(path.ray.d - (interaction.geometryNormal * (dotNgRo + correctionBias)));
			}

			// Check if path depth threshold was passed
			{
				bool continuePath{ ++path.depth < parameters.pathState.maxPathDepth };
				if (!static_cast<bool>(path.stateFlags & PathStateBitfield::CURRENT_HIT_SPECULAR) && !interaction.skipped)
				{
					if (static_cast<bool>(path.stateFlags & PathStateBitfield::REFRACTION_HAPPENED))
						continuePath = path.depth < parameters.pathState.maxTransmittedPathDepth;
					else
						continuePath = path.depth < parameters.pathState.maxReflectedPathDepth;
				}
				path.stateFlags |= continuePath ? PathStateBitfield::NO_FLAGS : PathStateBitfield::FINISHED;
			}

			// Preparing to next trace or terminating
			{
				updateStateFlags(path.stateFlags);
				qrngState.advanceBounce();
			}
		} while (!static_cast<bool>(path.stateFlags & PathStateBitfield::FINISHED)
				 &&
				 !static_cast<bool>(path.stateFlags & PathStateBitfield::PATH_TERMINATED));
	finishPath:
		qrngState.advanceSample();

		// Add the sample
		if (!static_cast<bool>(path.stateFlags & PathStateBitfield::PATH_TERMINATED))
		{
			// Resolve the sample
			SampledSpectrum& L{ path.L };
			const SampledSpectrum& pdf{ path.wavelengths.getPDF() };
			for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
			{
				L[i] = pdf[i] != 0.0f ? L[i] / pdf[i] : 0.0f;
			}

			result += glm::dvec4{color::toRGB(*parameters.sensorSpectralCurveA, *parameters.sensorSpectralCurveB, *parameters.sensorSpectralCurveC,
					path.wavelengths, path.L), 1.0f};
		}
		else
		{
			result += glm::dvec4{0.0, 0.0, 0.0, 1.0f};
		}
	} while (++sample < parameters.samplingState.count);

	parameters.renderData[renderDataIndex] = result;
}
