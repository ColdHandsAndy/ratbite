#include <cuda_runtime.h>
#include <optix_device.h>
#include <cuda/std/cstdint>
#include <cuda/std/cmath>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/launch_parameters.h"
#include "../core/math_util.h"
#include "../core/util.h"
#include "../core/material.h"
#include "../core/light_tree_types.h"
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
	glm::vec3 lastShadingNormal{};
	float refractionScale{};
	float bxdfPDF{};
	PathStateBitfield stateFlags{};
	uint32_t depth{};
};
struct Interaction
{
	glm::vec3 hitPos{};
	LightType lightType{};
	uint32_t lightIndex{};
	MaterialData* material{};
	glm::vec3 geometryNormal{};
	bool backHit{};
	bool skipped{ false };
	union PrimitiveData
	{
		struct Triangle
		{
			uint32_t index{};
			glm::vec3 vertices[3]{};
			glm::vec3 hitPosInterp{};
			glm::vec2 barycentrics{};
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

	float3 u{ verticesObj[1].x - verticesObj[0].x, verticesObj[1].y - verticesObj[0].y, verticesObj[1].z - verticesObj[0].z };
	float3 v{ verticesObj[2].x - verticesObj[0].x, verticesObj[2].y - verticesObj[0].y, verticesObj[2].z - verticesObj[0].z };
	float3 geometryNormal{u.y * v.z - u.z * v.y,
						  u.z * v.x - u.x * v.z,
						  u.x * v.y - u.y * v.x};
	geometryNormal = {geometryNormal.x * OFW[0] + geometryNormal.y * OFW[4] + geometryNormal.z * OFW[8], 
				      geometryNormal.x * OFW[1] + geometryNormal.y * OFW[5] + geometryNormal.z * OFW[9], 
				      geometryNormal.x * OFW[2] + geometryNormal.y * OFW[6] + geometryNormal.z * OFW[10]};
	float normalizeDiv{ __frsqrt_rn(geometryNormal.x * geometryNormal.x + geometryNormal.y * geometryNormal.y + geometryNormal.z * geometryNormal.z) };
	geometryNormal = {geometryNormal.x * normalizeDiv, geometryNormal.y * normalizeDiv, geometryNormal.z * normalizeDiv};
	uint32_t encGeometryNormal{ Octohedral::encodeU32(glm::vec3{geometryNormal.x, geometryNormal.y, geometryNormal.z}) };

	LightTree::LightPointer lightPointer{};
	lightPointer.pack(LightType::NONE, 0);
	if (materialIndex == 0xFFFFFFFF)
	{
		uint32_t emissivePrimIndex{ primitiveIndex };
		lightPointer.pack(LightType::TRIANGLE, emissivePrimIndex);

		primitiveIndex = parameters.lightTree.triangles[emissivePrimIndex].primitiveDataIndex;
		materialIndex = parameters.lightTree.triangles[emissivePrimIndex].materialIndex;
	}

	optixSetPayload_0(__float_as_uint(verticesObj[0].x * WFO[0] + verticesObj[0].y * WFO[1] + verticesObj[0].z * WFO[2]   + WFO[3]));
	optixSetPayload_1(__float_as_uint(verticesObj[0].x * WFO[4] + verticesObj[0].y * WFO[5] + verticesObj[0].z * WFO[6]   + WFO[7]));
	optixSetPayload_2(__float_as_uint(verticesObj[0].x * WFO[8] + verticesObj[0].y * WFO[9] + verticesObj[0].z * WFO[10]  + WFO[11]));
	optixSetPayload_3(encGeometryNormal);
	optixSetPayload_4(primitiveIndex);
	optixSetPayload_5(__float_as_uint(barycentrics.x));
	optixSetPayload_6(__float_as_uint(barycentrics.y));
	optixSetPayload_7(materialIndex);
	optixSetPayload_8(lightPointer.lptr);
	optixSetPayload_9(__float_as_uint(OFW[0]));
	optixSetPayload_10(__float_as_uint(OFW[4]));
	optixSetPayload_11(__float_as_uint(OFW[8]));
	optixSetPayload_12(__float_as_uint(OFW[1]));
	optixSetPayload_13(__float_as_uint(OFW[5]));
	optixSetPayload_14(__float_as_uint(OFW[9]));
	optixSetPayload_15(__float_as_uint(OFW[2]));
	optixSetPayload_16(__float_as_uint(OFW[6]));
	optixSetPayload_17(__float_as_uint(OFW[10]));
	optixSetPayload_18(__float_as_uint(verticesObj[1].x * WFO[0] + verticesObj[1].y * WFO[1] + verticesObj[1].z * WFO[2]   + WFO[3]));
	optixSetPayload_19(__float_as_uint(verticesObj[1].x * WFO[4] + verticesObj[1].y * WFO[5] + verticesObj[1].z * WFO[6]   + WFO[7]));
	optixSetPayload_20(__float_as_uint(verticesObj[1].x * WFO[8] + verticesObj[1].y * WFO[9] + verticesObj[1].z * WFO[10]  + WFO[11]));
	optixSetPayload_21(__float_as_uint(verticesObj[2].x * WFO[0] + verticesObj[2].y * WFO[1] + verticesObj[2].z * WFO[2]   + WFO[3]));
	optixSetPayload_22(__float_as_uint(verticesObj[2].x * WFO[4] + verticesObj[2].y * WFO[5] + verticesObj[2].z * WFO[6]   + WFO[7]));
	optixSetPayload_23(__float_as_uint(verticesObj[2].x * WFO[8] + verticesObj[2].y * WFO[9] + verticesObj[2].z * WFO[10]  + WFO[11]));
}
extern "C" __global__ void __intersection__disk()
{
	// optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);
	//
	// float3 rOT{ optixGetWorldRayOrigin() };
	// float3 rDT{ optixGetWorldRayDirection() };
	// glm::vec3 rO{ rOT.x, rOT.y, rOT.z };
	// glm::vec3 rD{ rDT.x, rDT.y, rDT.z };
	//
	// uint32_t lightIndex{ optixGetPrimitiveIndex() };
	//
	// const DiskLightData& dl{ parameters.lightTree.disks[lightIndex] };
	//
	// glm::mat3 matFrame{ glm::mat3_cast(dl.frame) };
	// glm::vec3 dC{ dl.position };
	// //glm::vec3 dT{ matFrame[0] };
	// //glm::vec3 dB{ matFrame[1] };
	// glm::vec3 dN{ matFrame[2] };
	// dN = glm::dot(rD, dN) < 0.0f ? dN : -dN;
	// float dR{ dl.radius };
	//
	// glm::vec3 o{ rO - dC };
	// float t{ -glm::dot(dN, o) / glm::dot(rD, dN) };
	// glm::vec3 rhP{ o + rD * t };
	//
	// LightTree::LightPointer lightPointer{};
	// lightPointer.pack(LightType::DISK, lightIndex);
	//
	// bool intersect{ glm::dot(rhP, rhP) < dR * dR };
	// if (intersect)
	// {
	// 	uint32_t encGeometryNormal{ Octohedral::encodeU32(dN) };
	// 	glm::vec3 hP{ rhP + dC };
	// 	optixSetPayload_0(__float_as_uint(hP.x));
	// 	optixSetPayload_1(__float_as_uint(hP.y));
	// 	optixSetPayload_2(__float_as_uint(hP.z));
	// 	optixSetPayload_3(encGeometryNormal);
	// 	optixSetPayload_7(dl.materialIndex);
	// 	optixSetPayload_8(lightPointer.lptr);
	// 	optixReportIntersection(t, 0);
	// }
}
extern "C" __global__ void __closesthit__disk()
{
	// optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);
}
extern "C" __global__ void __closesthit__sphere()
{
	// optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);
	//
	// const uint32_t primID{ optixGetPrimitiveIndex() };
	//
	// uint32_t lightIndex{ primID };
	// const SphereLightData& sl{ parameters.lightTree.spheres[lightIndex] };
	// glm::vec4 sphere{ sl.position, sl.radius };
	//
	// float3 rOT{ optixGetWorldRayOrigin() };
	// float3 rDT{ optixGetWorldRayDirection() };
	// glm::vec3 rO{ rOT.x, rOT.y, rOT.z };
	// glm::vec3 rD{ rDT.x, rDT.y, rDT.z };
	//
	// glm::vec3 hP{ rO + rD * optixGetRayTmax() };
	// glm::vec3 dN{ glm::normalize(hP - glm::vec3{sphere.x, sphere.y, sphere.z}) };
	// glm::vec3 oc{ rO - glm::vec3{sphere.x, sphere.y, sphere.z} };
	// float d2{ oc.x * oc.x + oc.y * oc.y + oc.z * oc.z };
	// if (d2 < sphere.w * sphere.w)
	// 	dN = -dN;
	//
	// LightTree::LightPointer lightPointer{};
	// lightPointer.pack(LightType::SPHERE, lightIndex);
	//
	// uint32_t encGeometryNormal{ Octohedral::encodeU32(dN) };
	// optixSetPayload_0(__float_as_uint(hP.x));
	// optixSetPayload_1(__float_as_uint(hP.y));
	// optixSetPayload_2(__float_as_uint(hP.z));
	// optixSetPayload_3(encGeometryNormal);
	// optixSetPayload_7(sl.materialIndex);
	// optixSetPayload_8(lightPointer.lptr);
}

extern "C" __global__ void __miss__miss()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0); 
	LightTree::LightPointer lightPointer{};
	lightPointer.pack(LightType::SKY, 0);
	optixSetPayload_8(lightPointer.lptr);
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
	/*
	   OpenPBR Surface Specification (https://academysoftwarefoundation.github.io/OpenPBR/):

	   -------------------------------------------------
	                         FUZZ                       
	   -------------------------------------------------
	                         COAT                       
	   -------------------------------------------------
	   METAL | TRANSLUCENT | SUBSURFACE | GLOSSY-DIFFUSE
	   -------------------------------------------------
	                       <--------------------------->
						   Opaque
						   Base
	   <-----><---------------------------------------->
	   Metal  Dielectric
	   Base   Base

					┌─────┐
				   ┌┤ PBR ├─┐
				   ││(MIX)│ │
			 1 - a │└─────┘ │ a
				┌──┘        └────┐
			┌───┴───┐        ┌───┴───┐
			│Ambinet│       ┌┤Surface├─┐
			│Medium │       ││(LAYER)│ │
			└───────┘     1 │└───────┘ │ F
						  ┌─┘          └──┐
					┌─────┴─────┐       ┌─┴──┐
				   ┌┤Coated Base├─┐     │FUZZ│
				   ││  (LAYER)  │ │     └────┘
				 1 │└───────────┘ │ C
				 ┌─┘              └──┐
			┌────┴────┐            ┌─┴──┐
		  ┌─┤Base     ├─┐          │COAT│
		  │ │Substrate│ │          └────┘
		  │ │  (MIX)  │ │
		M │ └─────────┘ │  1 - M
	   ┌──┘             └─────────┐
	┌──┴──┐                 ┌─────┴────┐
	│Metal│               ┌─┤Dielectric│──┐
	└─────┘               │ │Base      │  │
						  │ │  (MIX)   │  │
					   T  │ └──────────┘  │  1 - T
					┌─────┘               └─────────┐
			  ┌─────┴─────┐                   ┌─────┴─────┐
			  │Translucent│                 ┌─│Opaque Base│──┐
			  │Base       │                 │ │   (MIX)   │  │
			  └───────────┘             S   │ └───────────┘  │ 1 - S
									 ┌──────┘                └──────┐
								 ┌───┴──────┐              ┌────────┴────────┐
								 │Subsurface│              │     (LAYER)     │
								 └──────────┘              │Diffuse     Gloss│
														   └─────────────────┘
	TODO:
	* Subsurface scattering (Have to learn more about volumetrics before implementing)
	* Coat (GLTF has different specification for coat so not as important)

	*/

	// Transform vectors into local space
	glm::vec3 locWo{ -rD };
	glm::vec3 locLi{ directLightData.lightDir };
	local.toLocal(locWo, locLi);

	if (locWo.z == 0.0f)
	{
		stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
		return;
	}

	glm::vec3 wo{ locWo };
	microfacet::ContextOutgoing ctxo{ microfacet::createContextOutgoing(wo) };
	glm::vec3 wm{};
	glm::vec3 wi{};

	// Initialize GGX roughness data and regularize if necessary
	float roughness{ surface.specular.roughness };
	bool perfectSpecular{ false };
	if (roughness < 0.001f)
	{
		roughness = 0.001f;
		perfectSpecular = true;
	}
	float alpha{ utility::roughnessToAlpha(roughness) };
	microfacet::Alpha alphaMS{ .alphaX = alpha, .alphaY = alpha };
	// if (static_cast<bool>(stateFlags & PathStateBitfield::REGULARIZED))
	// 	alphaMS.regularize();

	// Initialize eta and f0
	float eta{ wo.z > 0.0f ? surface.specular.ior / 1.0f : 1.0f / surface.specular.ior };
	float f0Sr{ (1.0f - eta) / (1.0f + eta) };
	float f0{ f0Sr * f0Sr };
	//Sample LUTs
	float absCosThetaO{ cuda::std::abs(LocalTransform::cosTheta(wo)) };
	float energyCompensationTermDielectric{ 1.0f / tex3D<float>(eta > 1.0f ? parameters.LUTs.dielectricOuterAlbedo : parameters.LUTs.dielectricInnerAlbedo,
			absCosThetaO, roughness, cuda::std::sqrt(cuda::std::abs(f0Sr))) };
	float energyPreservationTermDiffuse{ 1.0f - tex3D<float>(eta > 1.0f ? parameters.LUTs.reflectiveDielectricOuterAlbedo : parameters.LUTs.reflectiveDielectricInnerAlbedo,
			absCosThetaO, roughness, cuda::std::sqrt(cuda::std::abs(f0Sr))) };
	float hemisphericalAlbedoConductor{ tex2D<float>(parameters.LUTs.conductorAlbedo, absCosThetaO, roughness) };
	float hemisphericalAlbedoSheen{ 0.0f };
	microflake::LTC::CoefficientsLTC sheenLTCCoefficients{};
	if (surface.layers & microsurface::Layers::SHEEN)
	{
		microflake::LTC::getLUTData(parameters.LUTs.sheenLTC, absCosThetaO, surface.sheen.alpha, sheenLTCCoefficients, hemisphericalAlbedoSheen);
		if (hemisphericalAlbedoSheen < 0.0001f)
			hemisphericalAlbedoSheen = 0.0f;
	}

	// Calculate direct lighting
	if (!directLightData.occluded)
	{
		// Initialize an incoming direction of a sample and related variables
		wi = locLi;
		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
		const float cosThetaI{ LocalTransform::cosTheta(wi) };
		const float cosThetaO{ LocalTransform::cosTheta(wo) };
		const float cosFactor{ cuda::std::fabs(cosThetaI) };
		const bool reflect{ cosThetaI * cosThetaO > 0.0f };
		float etaR{ 1.0f };
		if (!reflect)
			etaR = eta;

		// Calculate a micronormal of a sample and related variables
		wm = glm::normalize(wi * etaR + wo);
		if (wm.z < 0.0f)
			wm = -wm;
		microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };
		const float dotWmWo{ glm::dot(wm, wo) };
		const float dotWmWi{ glm::dot(wm, wi) };
		const float wowmAbsDot{ cuda::std::fabs(glm::dot(wo, wm)) };

		// Microsfacet BxDF factors initialization
		const float FDielectric{ microfacet::FSchlick(f0, wowmAbsDot) };
		const float Ds{ microfacet::D(ctxm, alphaMS) };
		const float Gs{ microfacet::G(ctxi, ctxo, alphaMS) };

		// Initialize surface material weights
		const float sheen{ hemisphericalAlbedoSheen };
		const float base{ 1.0f - sheen };
		const float metalness{ surface.base.metalness };
		const float translucency{ surface.transmission.weight };
		const float sheenWeight{ sheen };
		const float cSpecWeight{ base * metalness };
		const float dSpecWeight{ base * (1.0f - metalness) * FDielectric };
		const float dTransWeight{ base * (1.0f - metalness) * translucency * (1.0f - FDielectric) };
		const float diffuseWeight{ base * (1.0f - metalness) * (1.0f - translucency) * (1.0f - FDielectric) };

		SampledSpectrum flSheen{ 0.0f };
		float lbxdfSheenPDF{ 1.0f };
		if (sheen != 0.0f)
		{
			float flSheenNoAbsorb{ 0.0f };
			microflake::LTC::evaluate(wo, wi, sheenLTCCoefficients, flSheenNoAbsorb, lbxdfSheenPDF);
			flSheen = color::RGBtoSpectrum(surface.sheen.color,
					wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) * flSheenNoAbsorb;
		}

		if (reflect)
		{
			SampledSpectrum flConductor{ 0.0f };
			float flDielectric{ 0.0f };
			float lbxdfSpecPDF{ 0.0f };
			if (!perfectSpecular)
			{
				SampledSpectrum FConductor{ color::RGBtoSpectrum(microfacet::F82(surface.base.color, wowmAbsDot) *
						(1.0f + microfacet::FAvgIntegralForF82(surface.base.color) * ((1.0f - hemisphericalAlbedoConductor) / hemisphericalAlbedoConductor)),
						wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) };
				flConductor = Ds * Gs * FConductor / cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi));
				flDielectric = Ds * Gs * energyCompensationTermDielectric // FDielectric is included in dSpecWeight
					/ cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi));
				lbxdfSpecPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) / (4.0f * wowmAbsDot);
			}

			SampledSpectrum flDiffuse{ energyPreservationTermDiffuse *
				color::RGBtoSpectrum(surface.base.color, wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) / glm::pi<float>() };
			const float lbxdfDiffPDF{ diffuse::CosWeighted::PDF(LocalTransform::cosTheta(wi)) };

			const float bxdfPDF{
				sheenWeight * lbxdfSheenPDF +
				cSpecWeight * lbxdfSpecPDF +
				dSpecWeight * lbxdfSpecPDF +
				diffuseWeight * lbxdfDiffPDF };
			const float mis{ MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, bxdfPDF) };
			L += directLightData.spectrumSample * throughputWeight * cosFactor * mis * (
					sheenWeight * flSheen
					+
					cSpecWeight * flConductor
					+
					dSpecWeight * flDielectric
					+
					diffuseWeight * flDiffuse
					) / directLightData.lightSamplePDF;
		}
		else if (surface.transmission.weight != 0.0f && !(dotWmWo * cosThetaO <= 0.0f || dotWmWi * cosThetaI <= 0.0f))
		{
			float lbxdfTransPDF{ 0.0f };
			SampledSpectrum flTransmission{ 0.0f };
			if (!perfectSpecular)
			{
				const float t{ dotWmWi + dotWmWo / eta };
				const float denom{ t * t };
				const float dwmdwi{ cuda::std::fabs(dotWmWi) / denom };
				lbxdfTransPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) * dwmdwi;
				flTransmission = color::RGBtoSpectrum(surface.base.color, wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB)
					* Ds * (1.0f - FDielectric) * Gs * energyCompensationTermDielectric
					* cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom));
			}

			const float bxdfPDF{
				sheenWeight * lbxdfSheenPDF +
				dTransWeight * lbxdfTransPDF };
			const float mis{ MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, bxdfPDF) };
			flTransmission /= (eta * eta);
			L += directLightData.spectrumSample * throughputWeight * cosFactor * mis * (
					sheenWeight * flSheen
					+
					dTransWeight * flTransmission
					) / directLightData.lightSamplePDF;
		}
	}

	// BxDF sampling and calculation
	{
		glm::vec3 rand{ QRNG::Sobol::sample3D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF_0) };

		const float sheen{ hemisphericalAlbedoSheen };
		const float base{ 1.0f - sheen };
		const float sheenP{ sheen };
		const float baseP{ base };
		const float baseLayerEnergyCorrection{ 1.0f - hemisphericalAlbedoSheen };

		if (rand.z > baseP)
		{
			SampledSpectrum f{};
			float fNoAbsorb{ 0.0f };
			float pdfP{};
			wi = microflake::LTC::sample(wo, sheenLTCCoefficients, rand);
			microflake::LTC::evaluate(wo, wi, sheenLTCCoefficients, fNoAbsorb, pdfP);
			// TODO: Figure out why "sheen" factor is needed for energy conservation
			f = color::RGBtoSpectrum(surface.sheen.color,
					wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) * fNoAbsorb * sheen;
			bxdfPDF = pdfP * sheenP;
			throughputWeight *= f * cuda::std::abs(LocalTransform::cosTheta(wi)) / bxdfPDF;
		}
		else
		{
			wm = microfacet::VNDF::sample<microfacet::VNDF::SPHERICAL_CAP>(wo, alphaMS, rand);
			microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };
			const float wowmAbsDot{ cuda::std::abs(glm::dot(wo, wm)) };

			float Ds{ microfacet::D(ctxm, alphaMS) };
			float FDielectric{ microfacet::FSchlick(f0, wowmAbsDot) };
			float sin2ThetaI{ cuda::std::fmax(0.0f, 1.0f - wowmAbsDot * wowmAbsDot) };
			float sin2ThetaT{ sin2ThetaI / (eta * eta) };
			if (sin2ThetaT >= 1.0f)
				FDielectric = 1.0f;

			const float metalness{ surface.base.metalness };
			const float translucency{ surface.transmission.weight };
			const float conductorP{ base * metalness };
			const float dielectricSpecP{ base * (1.0f - metalness) * FDielectric };
			const float transmissionP{ base * translucency * (1.0f - metalness) * (1.0f - FDielectric) };
			const float diffuseP{ base * (1.0f - translucency) * (1.0f - metalness) * (1.0f - FDielectric) };

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
							(1.0f + microfacet::FAvgIntegralForF82(surface.base.color) * ((1.0f - hemisphericalAlbedoConductor) / hemisphericalAlbedoConductor)),
							wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB) };
					f = spec * FConductor * metalness;
					pdfP = conductorP;
				}
				else
				{
					f = spec * FDielectric * energyCompensationTermDielectric * (1.0f - metalness);
					pdfP = dielectricSpecP;
				}
				float cosFactor{ cuda::std::fabs(LocalTransform::cosTheta(wi)) };
				bxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) / (4.0f * wowmAbsDot) * pdfP;
				throughputWeight *= f * cosFactor * baseLayerEnergyCorrection / bxdfPDF;
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
					float f{ Ds * (1.0f - FDielectric) * Gs * (1.0f - metalness) * translucency
						* cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom)) };
					f /= (eta * eta);
					bxdfPDF = microfacet::VNDF::PDF<microfacet::VNDF::SPHERICAL_CAP>(wo, wowmAbsDot, ctxo, ctxm, alphaMS) * dwmdwi * transmissionP;
					float cosFactor{ cuda::std::fabs(cosThetaI) };
					throughputWeight *= color::RGBtoSpectrum(surface.base.color, wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB)
						* f * energyCompensationTermDielectric * cosFactor * baseLayerEnergyCorrection / bxdfPDF;
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
					SampledSpectrum f{ color::RGBtoSpectrum(surface.base.color, wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB)
						/ glm::pi<float>() * (1.0f - metalness) * (1.0f - translucency) };
					bxdfPDF = diffuse::CosWeighted::PDF(LocalTransform::cosTheta(wi)) * diffuseP;
					throughputWeight *= f * energyPreservationTermDiffuse * cosFactor * baseLayerEnergyCorrection / bxdfPDF;
				}
			}
		}
	}

	// Check for invalid PDF
	if (bxdfPDF <= 0.0f || isinf(bxdfPDF))
	{
		stateFlags = stateFlags | PathStateBitfield::PATH_TERMINATED;
		return;
	}

	// Set new ray direction
	glm::vec3 locWi{ wi };
	local.fromLocal(locWi);
	rD = glm::normalize(locWi);
}


CU_DEVICE CU_INLINE void unpackInteractionData(const LaunchParameters& params, uint32_t* pl,
		Path& path, Interaction& interaction, glm::mat3& worldFromObjectNormal)
{
	interaction.geometryNormal = Octohedral::decodeU32(pl[3]);
	uint32_t matIndex{ pl[7] };
	interaction.material = params.materials + matIndex;
	
	LightTree::LightPointer lightPointer{ .lptr = pl[8] };
	lightPointer.unpack(interaction.lightType, interaction.lightIndex);

	bool triangularGeometry{ interaction.lightType == LightType::TRIANGLE || interaction.lightType == LightType::NONE };
	if (triangularGeometry)
	{
		path.stateFlags |= PathStateBitfield::TRIANGULAR_GEOMETRY;
		interaction.primitive.triangle.index = pl[4];
		glm::vec2& barycentrics{ interaction.primitive.triangle.barycentrics };
		barycentrics = { __uint_as_float(pl[5]), __uint_as_float(pl[6]) };
		glm::vec3& vert0{ interaction.primitive.triangle.vertices[0] };
		glm::vec3& vert1{ interaction.primitive.triangle.vertices[1] };
		glm::vec3& vert2{ interaction.primitive.triangle.vertices[2] };
		vert0 = glm::vec3{ __uint_as_float(pl[0]), __uint_as_float(pl[1]), __uint_as_float(pl[2]) };
		vert1 = glm::vec3{ __uint_as_float(pl[18]), __uint_as_float(pl[19]), __uint_as_float(pl[20]) };
		vert2 = glm::vec3{ __uint_as_float(pl[21]), __uint_as_float(pl[22]), __uint_as_float(pl[23]) };
		glm::vec3& hitPos{ interaction.hitPos };
		hitPos.x = vert0.x * (1.0f - barycentrics.x - barycentrics.y) + vert1.x * barycentrics.x + vert2.x * barycentrics.y;
		hitPos.y = vert0.y * (1.0f - barycentrics.x - barycentrics.y) + vert1.y * barycentrics.x + vert2.y * barycentrics.y;
		hitPos.z = vert0.z * (1.0f - barycentrics.x - barycentrics.y) + vert1.z * barycentrics.x + vert2.z * barycentrics.y;
	}
	else
	{
		interaction.hitPos = glm::vec3{ __uint_as_float(pl[0]), __uint_as_float(pl[1]), __uint_as_float(pl[2]) };
	}

	interaction.primitive.triangle.hitPosInterp = interaction.hitPos;
	interaction.backHit = glm::dot(interaction.geometryNormal, path.ray.d) > 0.0f;

	worldFromObjectNormal[0][0] = __uint_as_float(pl[9]);
	worldFromObjectNormal[1][0] = __uint_as_float(pl[10]);
	worldFromObjectNormal[2][0] = __uint_as_float(pl[11]);
	worldFromObjectNormal[0][1] = __uint_as_float(pl[12]);
	worldFromObjectNormal[1][1] = __uint_as_float(pl[13]);
	worldFromObjectNormal[2][1] = __uint_as_float(pl[14]);
	worldFromObjectNormal[0][2] = __uint_as_float(pl[15]);
	worldFromObjectNormal[1][2] = __uint_as_float(pl[16]);
	worldFromObjectNormal[2][2] = __uint_as_float(pl[17]);
}
CU_DEVICE CU_INLINE void updateStateFlags(PathStateBitfield& stateFlags)
{
	PathStateBitfield excludeFlags{
		PathStateBitfield::CURRENT_HIT_SPECULAR |
		PathStateBitfield::PREVIOUS_HIT_SPECULAR |
		PathStateBitfield::TRIANGULAR_GEOMETRY |
		PathStateBitfield::RIGHT_HANDED_FRAME |
		PathStateBitfield::RAY_REFRACTED };

	PathStateBitfield includeFlags{ PathStateBitfield::NO_FLAGS };
	if (static_cast<bool>(stateFlags & PathStateBitfield::CURRENT_HIT_SPECULAR))
		includeFlags |= PathStateBitfield::PREVIOUS_HIT_SPECULAR;
	else
		includeFlags |= PathStateBitfield::REGULARIZED;
	if (static_cast<bool>(stateFlags & PathStateBitfield::RAY_REFRACTED))
		includeFlags |= PathStateBitfield::REFRACTION_HAPPENED;

	stateFlags = (stateFlags & (~excludeFlags)) | includeFlags;
}

CU_DEVICE CU_INLINE float evaluateNodeImportance(const LightTree::UnpackedNode& node, const glm::vec3& position, const glm::vec3& normal)
{
	namespace SG = SphericalGaussian;

	// Create Spherical Gaussian Light Lobe
	const glm::vec3 spatialMean{
		node.attributes.spatialMean[0],
		node.attributes.spatialMean[1],
		node.attributes.spatialMean[2], };
	const glm::vec3 averageDirection{
		node.attributes.averageDirection[0],
		node.attributes.averageDirection[1],
		node.attributes.averageDirection[2], };
	glm::vec3 lightDir{ spatialMean - position };
	const float squaredDistance{ glm::dot(lightDir, lightDir) };
	lightDir *= rsqrt(squaredDistance);
	const float lightVariance{ cuda::std::fmax(node.attributes.spatialVariance, (0x1.0p-31f) * squaredDistance) };
	const float lightSharpness{ squaredDistance / lightVariance };
	const SG::Lobe lightLobe{ SG::Product(averageDirection, node.attributes.sharpness, lightDir, lightSharpness) };

	// Calculate importance (estimate for diffuse distribution only because we evaluate the entire BxDF when light sampling and it is cheaper)
	const float amplitude{ cuda::std::exp(lightLobe.logAmplitude) };
	const float cosine{ glm::clamp(glm::dot(lightLobe.axis, normal), -1.0f, 1.0f) };
	const float importance{ (node.attributes.flux / (lightVariance * SG::Integral(node.attributes.sharpness))) *
		amplitude * SG::ClampedCosineProductIntegralOverPi2024(cosine, lightLobe.sharpness) };

	return importance;
}
CU_DEVICE CU_INLINE float traverseLightTree(const LaunchParameters::LightTree& tree, const LightType type, const uint32_t index,
		const glm::vec3& position, const glm::vec3& normal)
{
	float PDF{};
	float envMapImportance{ tree.envMap.enabled ? LightTree::KEnvironmentMapImportance : 0.0f };
	if (type == LightType::SKY)
	{
		PDF = envMapImportance;
		return PDF;
	}
	else
		PDF = 1.0f - envMapImportance;

	// Get bitmask of the given light
	uint64_t bitmask{ tree.bitmasks[static_cast<int>(type)][index] };

	// Initialize the first node
	uint32_t currentNodeIndex{ 0 };
	bool isLeaf{};
	LightTree::UnpackedNode current{ LightTree::unpackNode(tree.nodes[currentNodeIndex], isLeaf) };
	for (int depth{ 0 }; depth < sizeof(uint64_t) * 8; ++depth)
	{
		// Break if current node is a leaf node
		if (isLeaf)
			break;

		// Access child nodes
		bool leftIsLeaf{};
		LightTree::UnpackedNode leftNode{ LightTree::unpackNode(tree.nodes[currentNodeIndex + 1], leftIsLeaf) };
		float leftImportance{ evaluateNodeImportance(leftNode, position, normal) };
		bool rightIsLeaf{};
		LightTree::UnpackedNode rightNode{ LightTree::unpackNode(tree.nodes[current.core.branch.rightChildIndex], rightIsLeaf) };
		float rightImportance{ evaluateNodeImportance(rightNode, position, normal) };
		
		// Compute PDF of the bitmask path
		bool rightChosen{ static_cast<bool>(bitmask & (1ull << depth)) };
		currentNodeIndex = rightChosen ? current.core.branch.rightChildIndex : currentNodeIndex + 1;
		current = rightChosen ? rightNode : leftNode;
		isLeaf = rightChosen ? rightIsLeaf : leftIsLeaf;
		PDF *= rightChosen ?
			rightImportance / (leftImportance + rightImportance)
			:
			leftImportance / (leftImportance + rightImportance);
	}
	PDF *= 1.0f / current.core.leaf.lightCount;

	return PDF;
}
CU_DEVICE CU_INLINE void sampleLightTree(const LaunchParameters::LightTree& tree, float rand, const glm::vec3& position, const glm::vec3& normal,
		LightType& type, uint32_t& index, float& PDF)
{
	// Helper function to normalize "rand" after choosing a node
	auto normRand{ [](float rand, float sepVal, bool rightChosen) -> float
		{
			float res{};
			if (rightChosen)
				res = (rand - sepVal) / (1.0f - sepVal);
			else
				res = rand / sepVal;
			return glm::clamp(res, 0.0f, 1.0f);
		} };

	// Choose between environment map and light tree sampling
	bool envMapIsPresent{ parameters.lightTree.envMap.enabled };
	bool lightTreeIsPresent{ parameters.lightTree.nodes != nullptr };
	float envMapImportance{ envMapIsPresent ? (lightTreeIsPresent ? LightTree::KEnvironmentMapImportance : 1.0f) : 0.0f };
	if (rand < envMapImportance)
	{
		type = LightType::SKY;
		index = 0;
		PDF = envMapImportance;
		return;
	}
	rand = normRand(rand, envMapImportance, true);
	PDF = 1.0f - envMapImportance;

	// Traversal and importance evaluation
	uint32_t currentNodeIndex{ 0 };
	bool isLeaf{};
	LightTree::UnpackedNode current{ LightTree::unpackNode(tree.nodes[currentNodeIndex], isLeaf) };
	while (!isLeaf)
	{
		bool leftIsLeaf{};
		LightTree::UnpackedNode left{ LightTree::unpackNode(tree.nodes[currentNodeIndex + 1], leftIsLeaf) };
		bool rightIsLeaf{};
		LightTree::UnpackedNode right{ LightTree::unpackNode(tree.nodes[current.core.branch.rightChildIndex], rightIsLeaf) };
		float leftImportance{ evaluateNodeImportance(left, position, normal) };
		float rightImportance{ evaluateNodeImportance(right, position, normal) };
		float importanceSepVal{ leftImportance / (leftImportance + rightImportance) };
		bool rightChosen{ rand > importanceSepVal };
		currentNodeIndex = rightChosen ? current.core.branch.rightChildIndex : currentNodeIndex + 1;
		current = rightChosen ? right : left;
		isLeaf = rightChosen ? rightIsLeaf : leftIsLeaf;
		PDF *= rightChosen ? 1.0f - importanceSepVal : importanceSepVal;
		rand = normRand(rand, importanceSepVal, rightChosen);
	};

	// Extracting a light pointer from the leaf
	uint32_t indexIntoLeafLights{ static_cast<uint32_t>(current.core.leaf.lightCount * rand) };
	uint32_t offsetPlusIndex{ current.core.leaf.lightOffset +
		(indexIntoLeafLights == current.core.leaf.lightCount ? current.core.leaf.lightCount - 1 : indexIntoLeafLights) };
	PDF *= 1.0f / current.core.leaf.lightCount;

	// Unpacking the light pointer
	tree.lightPointers[offsetPlusIndex].unpack(type, index);

	if (isnan(PDF) || isinf(PDF) || PDF == 0.0f)
		type = LightType::NONE;

	return;
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
			uint32_t pl[24];
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
					   pl[17], pl[18], pl[19], pl[20], pl[21], pl[22], pl[23]);

			// Unpack path interaction data
			Interaction interaction{};
			glm::mat3 worldFromObjectNormal{};
			unpackInteractionData(parameters, pl,
					path, interaction, worldFromObjectNormal);

			// Hit emission estimation
			if (interaction.lightType != LightType::NONE)
			{
				const bool prevNEEHappened{ path.depth != 0 && !static_cast<bool>(path.stateFlags & PathStateBitfield::PREVIOUS_HIT_SPECULAR) };
				float lightTreePDF{ prevNEEHappened ?
					traverseLightTree(parameters.lightTree, interaction.lightType, interaction.lightIndex, path.ray.o, path.lastShadingNormal)
					:
					1.0f };
				float lightPDF{};
				SampledSpectrum Le{};
				bool noEmission{ false };

				glm::vec3 toHit{ interaction.hitPos - path.ray.o };
				float sqrdDistToLight{ toHit.x * toHit.x + toHit.y * toHit.y + toHit.z * toHit.z };
				switch (interaction.lightType)
				{
					case LightType::TRIANGLE:
						{
							const EmissiveTriangleLightData& tri{ parameters.lightTree.triangles[interaction.lightIndex] };
							glm::vec3 a{ tri.vertices[0], tri.vertices[1], tri.vertices[2] };
							glm::vec3 b{ tri.vertices[3], tri.vertices[4], tri.vertices[5] };
							glm::vec3 c{ tri.vertices[6], tri.vertices[7], tri.vertices[8] };
							glm::vec3 n{ glm::cross(b - a, c - a) };
							float nL{ glm::length(n) };
							n /= nL;
							float area{ nL / 2.0f };
							float lCos{ -glm::dot(path.ray.d, n) };
							noEmission = lCos <= 0.0f;
							float surfacePDF{ 1.0f / area };
							lightPDF = surfacePDF * sqrdDistToLight / lCos;
							const MaterialData& material{ parameters.materials[tri.materialIndex] };
							float emission[3]{ material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2] };
							if (static_cast<bool>(material.textures & MaterialData::TextureTypeBitfield::EMISSION))
							{
								glm::vec2 bary{ interaction.primitive.triangle.barycentrics.x, interaction.primitive.triangle.barycentrics.y };
								float baryWeights[3]{ 1.0f - bary.x - bary.y, bary.x, bary.y };
								glm::vec2 uv{ baryWeights[0] * glm::vec2{tri.uvs[0], tri.uvs[1]} +
									baryWeights[1] * glm::vec2{tri.uvs[2], tri.uvs[3]} +
									baryWeights[2] * glm::vec2{tri.uvs[4], tri.uvs[5]} };
								float4 texEm{ tex2D<float4>(interaction.material->emissiveTexture, uv.x, uv.y) };
								emission[0] *= texEm.x;
								emission[1] *= texEm.y;
								emission[2] *= texEm.z;
							}
							Le = color::RGBtoSpectrum(
								glm::vec3{emission[0], emission[1], emission[2]},
								path.wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB);
						}
						break;
					case LightType::SKY:
						{
							const LaunchParameters::LightTree::EnvironmentMap& envMap{ parameters.lightTree.envMap };
							if (envMap.enabled)
							{
								float phi{ cuda::std::atan2(path.ray.d.y, path.ray.d.x) };
								float theta{ cuda::std::acos(path.ray.d.z) };
								float4 skyMap{ tex2D<float4>(envMap.environmentTexture,
										0.5f - phi / (2.0f * glm::pi<float>()), theta / glm::pi<float>()) };
								glm::vec3 skyColor{ skyMap.x, skyMap.y, skyMap.z };
								float surfacePDF{ 1.0f / (4.0f * glm::pi<float>()) };
								lightPDF = surfacePDF * ((skyColor.x + skyColor.y + skyColor.z) / 3.0f)
									* cuda::std::sin(theta) // Applying Cartesian to spherical Jacobian
									/ envMap.integral;
								Le = color::RGBtoSpectrum(skyColor, path.wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB);
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
					lightPDF *= lightTreePDF;

					float emissionWeight{};
					if (prevNEEHappened)
						emissionWeight = MIS::powerHeuristic(1, path.bxdfPDF, 1, lightPDF);
					else
						emissionWeight = 1.0f;

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
							normals[0] = frameMats[0][2];
							normals[1] = frameMats[1][2];
							normals[2] = frameMats[2][2];
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

							normals[0] = Octohedral::decode(*reinterpret_cast<glm::vec2*>(attr + attributesStride * indices[0]));
							normals[1] = Octohedral::decode(*reinterpret_cast<glm::vec2*>(attr + attributesStride * indices[1]));
							normals[2] = Octohedral::decode(*reinterpret_cast<glm::vec2*>(attr + attributesStride * indices[2]));
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
					glm::vec3 bitangent{ glm::cross(normal, tangent) * (flipTangent ? -1.0f : 1.0f) };
					glm::mat3 frame{ tangent, bitangent, normal };
					glm::vec3 n{};
					if (static_cast<bool>(interaction.material->textures & MaterialData::TextureTypeBitfield::NORMAL))
					{
						float2 uv{ interaction.material->nmTexCoordSetIndex ? float2{texC2.x, texC2.y} : float2{texC1.x, texC1.y} };
						float4 nm{ tex2D<float4>(interaction.material->normalTexture, uv.x, uv.y) };
						n = glm::normalize(glm::vec3{nm.x * 2.0f - 1.0f, nm.y * 2.0f - 1.0f, nm.z * 2.0f - 1.0f});
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

					glm::vec3 shadingBitangent{ glm::normalize(glm::cross(n, tangent)) };
					glm::vec3 shadingTangent{ glm::normalize(glm::cross(shadingBitangent, n)) };
					shadingBitangent *= (flipTangent ? -1.0f : 1.0f);
					frame = glm::mat3{shadingTangent, shadingBitangent, n};
					localTransform = LocalTransform{frame};
					path.lastShadingNormal = n;

					bool bcTexture{ static_cast<bool>(interaction.material->textures & MaterialData::TextureTypeBitfield::BASE_COLOR) };
					bool bcFactor{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::BASE_COLOR) };
					float alpha{};
					if (bcTexture)
					{
						float2 uv{ interaction.material->nmTexCoordSetIndex ? float2{texC2.x, texC2.y} : float2{texC1.x, texC1.y} };
						float4 bcTexData{ tex2D<float4>(interaction.material->baseColorTexture, uv.x, uv.y) };
						surface.base.color = glm::vec3{bcTexData.x, bcTexData.y, bcTexData.z};
						alpha = bcTexData.w;
						if (bcFactor)
						{
							surface.base.color
								*= glm::vec3{interaction.material->baseColorFactor[0], interaction.material->baseColorFactor[1], interaction.material->baseColorFactor[2]};
							alpha *= interaction.material->baseColorFactor[3];
						}
					}
					else if (bcFactor)
					{
						surface.base.color
							= glm::vec3{interaction.material->baseColorFactor[0], interaction.material->baseColorFactor[1], interaction.material->baseColorFactor[2]};
						alpha = interaction.material->baseColorFactor[3];
					}
					const bool cutoff{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::CUTOFF) };
					const bool blend{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::BLEND) };
					if (cutoff)
					{
						if (alpha < interaction.material->alphaCutoff)
							interaction.skipped = true;
					}
					else if (blend)
					{
						if (QRNG::Sobol::sample1D(qrngState, QRNG::DimensionOffset::ALPHA_BLEND) > alpha)
							interaction.skipped = true;
					}

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
						surface.transmission.weight = trTexData * (trFactor ? interaction.material->transmissionFactor : 1.0f);
					}
					else
					{
						surface.transmission.weight = trFactor ? interaction.material->transmissionFactor : surface.transmission.weight;
					}
					surface.specular.ior = interaction.material->ior;

					if (interaction.material->sheenPresent)
					{
						surface.layers |= microsurface::Layers::SHEEN;
						bool shcTexture{ static_cast<bool>(interaction.material->textures & MaterialData::TextureTypeBitfield::SHEEN_COLOR) };
						bool shcFactor{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::SHEEN_COLOR) };
						if (shcTexture)
						{
							float2 uv{ interaction.material->shcTexCoordSetIndex ? float2{texC2.x, texC2.y} : float2{texC1.x, texC1.y} };
							float4 shcTexData{ tex2D<float4>(interaction.material->sheenColorTexture, uv.x, uv.y) };
							surface.sheen.color.x = shcTexData.x * (shcFactor ? interaction.material->sheenColorFactor[0] : 1.0f);
							surface.sheen.color.y = shcTexData.y * (shcFactor ? interaction.material->sheenColorFactor[1] : 1.0f);
							surface.sheen.color.z = shcTexData.z * (shcFactor ? interaction.material->sheenColorFactor[2] : 1.0f);
						}
						else if (shcFactor)
						{
							surface.sheen.color.x = interaction.material->sheenColorFactor[0];
							surface.sheen.color.y = interaction.material->sheenColorFactor[1];
							surface.sheen.color.z = interaction.material->sheenColorFactor[2];
						}
						bool shrTexture{ static_cast<bool>(interaction.material->textures & MaterialData::TextureTypeBitfield::SHEEN_ROUGH) };
						bool shrFactor{ static_cast<bool>(interaction.material->factors & MaterialData::FactorTypeBitfield::SHEEN_ROUGH) };
						if (shrTexture)
						{
							float2 uv{ interaction.material->shrTexCoordSetIndex ? float2{texC2.x, texC2.y} : float2{texC1.x, texC1.y} };
							float shrTexData{ tex2D<float4>(interaction.material->sheenRoughTexture, uv.x, uv.y).w };
							float shRoughness{ shrTexData * (shrFactor ? interaction.material->sheenRoughnessFactor : 1.0f) };
							surface.sheen.alpha = shRoughness * shRoughness;
						}
						else if (shrFactor)
						{
							surface.sheen.alpha = interaction.material->sheenRoughnessFactor;
						}
					}
				}
			}
			else
			{
				localTransform = LocalTransform{interaction.geometryNormal};
				path.lastShadingNormal = interaction.geometryNormal;
			}

			// Light sampling and BxDF evaluation
			if (!interaction.skipped)
			{
				bool envMapIsPresent{ parameters.lightTree.envMap.enabled };
				bool lightTreeIsPresent{ parameters.lightTree.nodes != nullptr };
				// Next event estimation data
				DirectLightSampleData directLightData{};
				if (!envMapIsPresent && !lightTreeIsPresent)
				{
					directLightData.occluded = true;
				}
				else
				{
					glm::vec3 rand{ QRNG::Sobol::sample3D(qrngState, QRNG::DimensionOffset::LIGHT) };

					const bool triangularGeometry{ static_cast<bool>(path.stateFlags & PathStateBitfield::TRIANGULAR_GEOMETRY) };
					const bool surfaceCanTransmit{ surface.transmission.weight != 0.0f };
					const glm::vec3& gn{ interaction.geometryNormal };
					const glm::vec3& sn{ path.lastShadingNormal };

					// Sample light tree
					LightType type{};
					uint32_t index{};
					float lightTreePDF{};
					sampleLightTree(parameters.lightTree, rand.z,
							interaction.hitPos, sn,
							type, index, lightTreePDF);

					// Take the light sample
					glm::vec3 lightRayOrigin{};
					float dToL{};
					float lightPDF{};
					switch (type)
					{
						case LightType::TRIANGLE:
							{
								const EmissiveTriangleLightData& tri{ parameters.lightTree.triangles[index] };
								glm::vec3 a{ tri.vertices[0], tri.vertices[1], tri.vertices[2] };
								glm::vec3 b{ tri.vertices[3], tri.vertices[4], tri.vertices[5] };
								glm::vec3 c{ tri.vertices[6], tri.vertices[7], tri.vertices[8] };
								glm::vec3 n{ glm::cross(b - a, c - a) };
								float nL{ glm::length(n) };
								n /= nL;
								float area{ nL / 2.0f };

								float baryWeights[3]{};
								glm::vec3 lSmplPos{ sampling::triangle::sampleUniform(glm::vec2{rand.x, rand.y},
										a, b, c,
										baryWeights[0], baryWeights[1], baryWeights[2]) };

								const glm::vec3 sr{ lSmplPos - interaction.hitPos };
								const bool inSample{ (glm::dot(sn, sr) < 0.0f) };
								if ((inSample && !surfaceCanTransmit) || (inSample && (glm::dot(gn, sr) > 0.0f)))
									directLightData.occluded = true;
								lightRayOrigin = utility::offsetPoint(
									inSample || !triangularGeometry ? interaction.hitPos : interaction.primitive.triangle.hitPosInterp,
									inSample ? -gn : gn);

								glm::vec3 rToLight{ lSmplPos - lightRayOrigin };
								float sqrdToLight{ rToLight.x * rToLight.x + rToLight.y * rToLight.y + rToLight.z * rToLight.z };
								dToL = cuda::std::sqrtf(sqrdToLight);
								directLightData.lightDir = rToLight / dToL;
								float lCos{ -glm::dot(n, directLightData.lightDir) };

								directLightData.occluded = lCos <= 0.0f;

								float surfacePDF{ 1.0f / area };
								lightPDF = surfacePDF * sqrdToLight / lCos;

								const MaterialData& material{ parameters.materials[tri.materialIndex] };
								float emission[3]{ material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2] };
								if (static_cast<bool>(material.textures & MaterialData::TextureTypeBitfield::EMISSION))
								{
									glm::vec2 uv{ baryWeights[0] * glm::vec2{tri.uvs[0], tri.uvs[1]} +
										baryWeights[1] * glm::vec2{tri.uvs[2], tri.uvs[3]} +
										baryWeights[2] * glm::vec2{tri.uvs[4], tri.uvs[5]} };
									float4 texEm{ tex2D<float4>(material.emissiveTexture, uv.x, uv.y) };
									emission[0] *= texEm.x;
									emission[1] *= texEm.y;
									emission[2] *= texEm.z;
								}
								directLightData.spectrumSample = color::RGBtoSpectrum(
									glm::vec3{emission[0], emission[1], emission[2]},
									path.wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB);
							}
							break;
						case LightType::SKY:
							{
								// Importance sampling environment map with interpolated inverted CDF indices
								const LaunchParameters::LightTree::EnvironmentMap& envMap{ parameters.lightTree.envMap };
								glm::vec2 fullC{
									cuda::std::fmin(rand.x * (envMap.width - 1.0f), envMap.width - 1.0001f),
									cuda::std::fmin(rand.y * (envMap.height - 1.0f), envMap.height - 1.0001f) };
								glm::vec2 floorC{ glm::floor(fullC) };
								glm::vec2 fractC{ fullC - floorC };
								glm::uvec2 floorCI{ floorC };

								glm::vec2 cdfIndicesM{
									envMap.marginalCDFIndices[floorCI.y],
									envMap.marginalCDFIndices[floorCI.y + 1] };
								glm::vec2 cdfIndicesC{
									envMap.conditionalCDFIndices[static_cast<uint32_t>(cdfIndicesM.x * envMap.width) + floorCI.x],
									envMap.conditionalCDFIndices[static_cast<uint32_t>(cdfIndicesM.x * envMap.width) + floorCI.x + 1] };

								glm::vec2 impSample{
									glm::mix(cdfIndicesC.x, cdfIndicesC.y, fractC.x) * (1.0f / envMap.width),
										glm::mix(cdfIndicesM.x, cdfIndicesM.y, fractC.y) * (1.0f / envMap.height) };

								float phi{ 2.0f * glm::pi<float>() * impSample.x };
								float theta{ glm::pi<float>() * impSample.y };
								directLightData.lightDir = glm::vec3{
									-cuda::std::sin(theta) * cuda::std::cos(phi),
									cuda::std::sin(theta) * cuda::std::sin(phi),
									cuda::std::cos(theta), };
								float4 rgbSample{ tex2D<float4>(envMap.environmentTexture, impSample.x, impSample.y) };
								directLightData.spectrumSample = color::RGBtoSpectrum(glm::vec3{rgbSample.x, rgbSample.y, rgbSample.z},
										path.wavelengths, *parameters.spectralBasisR, *parameters.spectralBasisG, *parameters.spectralBasisB);
								float surfacePDF{ 1.0f / (4.0f * glm::pi<float>()) };
								lightPDF = surfacePDF * ((rgbSample.x + rgbSample.y + rgbSample.z) / 3.0f)
									* cuda::std::sin(theta) // Applying Cartesian to spherical Jacobian
									/ envMap.integral;
								dToL = FLT_MAX;
								lightRayOrigin = utility::offsetPoint(
										!triangularGeometry ? interaction.hitPos : interaction.primitive.triangle.hitPosInterp,
										interaction.backHit ? -interaction.geometryNormal : interaction.geometryNormal);
							}
							break;
						default:
							{
								directLightData.occluded = true;
							}
							break;
					}
					directLightData.lightSamplePDF = lightPDF * lightTreePDF;

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

			// Offset new ray origin and correct the direction
			{
				const bool refracted{ static_cast<bool>(path.stateFlags & PathStateBitfield::RAY_REFRACTED) };

				const bool triangularGeometry{ static_cast<bool>(path.stateFlags & PathStateBitfield::TRIANGULAR_GEOMETRY) };
				const bool in{ refracted || interaction.skipped };
				const glm::vec3& chosenOrigin{ (in == interaction.backHit) && triangularGeometry
					? interaction.primitive.triangle.hitPosInterp : interaction.hitPos };
				const glm::vec3 chosenOffset{ in == interaction.backHit
					? interaction.geometryNormal : -interaction.geometryNormal };
				path.ray.o = utility::offsetPoint(chosenOrigin, chosenOffset);

				const float dotNgRo{ glm::dot(interaction.geometryNormal, path.ray.d) };
				constexpr float correctionBias{ 0.001f };
				if (dotNgRo > 0.0f
					&&
					(interaction.backHit != refracted)
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
