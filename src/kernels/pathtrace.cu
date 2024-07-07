#include <cuda_runtime.h>
#include <optix_device.h>
#include <cuda/std/cstdint>
#include <cuda/std/cmath>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/launch_parameters.h"
#include "../core/util_macros.h"
#include "../core/material.h"
#include "../device/util.h"
#include "../device/local_transform.h"
#include "../device/filter.h"
#include "../device/color.h"
#include "../device/quasi_random.h"
#include "../device/spectral.h"
#include "../device/sampling.h"
#include "../device/microfacet.h"
#include "../device/mis.h"

typedef uint32_t PathStateFlags;
enum class PathStateFlagBit : uint32_t
{
	NO_FLAGS = 0u,
	PREVIOUS_HIT_SPECULAR = 1u,
	CURRENT_HIT_SPECULAR = 2u,
	MISS = 4u,
	TRANSMISSION = 8u,
	EMISSIVE_OBJECT_HIT = 16u,
	REGULARIZED = 32u,
	SECONDARY_SPECTRAL_SAMPLES_TERMINATED = 64u,
	PATH_TERMINATED = 128u,
};
STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_WITH_PREFIX(PathStateFlags, PathStateFlagBit, CU_DEVICE CU_INLINE)

extern "C"
{
	__constant__ LaunchParameters parameters{};
}

struct DirectLightData
{
	SampledSpectrum spectrumSample{};
	glm::vec3 lightDir{};
	float lightSamplePDF{};
	bool occluded{};
};

CU_DEVICE CU_INLINE void unpackTraceData(const LaunchParameters& params, glm::vec3& hP, glm::vec3& hN, PathStateFlags& stateFlags, MaterialData** materialDataPtr,
	uint32_t pl0, uint32_t pl1, uint32_t pl2, uint32_t pl3, uint32_t pl4, uint32_t pl5, uint32_t pl6)
{
	hP = glm::vec3{ __uint_as_float(pl0), __uint_as_float(pl1), __uint_as_float(pl2) };
	hN = glm::vec3{ __uint_as_float(pl3), __uint_as_float(pl4), __uint_as_float(pl5) };
	stateFlags |= pl6 >> 16;
	uint32_t matIndex = pl6 & 0xFFFF;
	*materialDataPtr = params.materials + matIndex;
}
CU_DEVICE CU_INLINE void updateStateFlags(uint32_t& stateFlags)
{
	PathStateFlags excludeFlags{ PathStateFlagBit::EMISSIVE_OBJECT_HIT | PathStateFlagBit::CURRENT_HIT_SPECULAR };
	PathStateFlags includeFlags{ (stateFlags & PathStateFlagBit::CURRENT_HIT_SPECULAR) ?
		static_cast<PathStateFlags>(PathStateFlagBit::PREVIOUS_HIT_SPECULAR) : static_cast<PathStateFlags>(PathStateFlagBit::REGULARIZED) };

	stateFlags = (stateFlags & (~excludeFlags)) | includeFlags;
}
CU_DEVICE CU_INLINE void resolveSample(SampledSpectrum& L, const SampledSpectrum& pdf)
{
	for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
	{
		L[i] = pdf[i] != 0.0f ? L[i] / pdf[i] : 0.0f;
	}
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

	float3 u{ vertexObjectData[2].x - vertexObjectData[0].x, vertexObjectData[2].y - vertexObjectData[0].y, vertexObjectData[2].z - vertexObjectData[0].z };
	float3 v{ vertexObjectData[1].x - vertexObjectData[0].x, vertexObjectData[1].y - vertexObjectData[0].y, vertexObjectData[1].z - vertexObjectData[0].z };

	float WFO[12]{};
	optixGetObjectToWorldTransformMatrix(WFO);

	float3 normal{u.y * v.z - u.z * v.y,
				  u.z * v.x - u.x * v.z,
				  u.x * v.y - u.y * v.x};
	normal = {normal.x * WFO[0] + normal.y * WFO[1] + normal.z * WFO[2],
			  normal.x * WFO[4] + normal.y * WFO[5] + normal.z * WFO[6],
			  normal.x * WFO[8] + normal.z * WFO[9] + normal.z * WFO[10] };
	float normalizeDiv{ __frsqrt_rn(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z) };
	normal = {normal.x * normalizeDiv, normal.y * normalizeDiv, normal.z * normalizeDiv};

	float3 hpos{ hposO.x * WFO[0] + hposO.y * WFO[1] + hposO.z * WFO[2]  + WFO[3], 
				 hposO.x * WFO[4] + hposO.y * WFO[5] + hposO.z * WFO[6]  + WFO[7], 
				 hposO.x * WFO[8] + hposO.y * WFO[9] + hposO.z * WFO[10] + WFO[11] };

	uint32_t materialIndex{ *reinterpret_cast<uint32_t*>(optixGetSbtDataPointer()) };

	optixSetPayload_0(__float_as_uint(hpos.x));
	optixSetPayload_1(__float_as_uint(hpos.y));
	optixSetPayload_2(__float_as_uint(hpos.z));
	optixSetPayload_3(__float_as_uint(normal.x));
	optixSetPayload_4(__float_as_uint(normal.y));
	optixSetPayload_5(__float_as_uint(normal.z));
	optixSetPayload_6(materialIndex);
}
extern "C" __global__ void __intersection__disk()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

	float3 rOT{ optixGetWorldRayOrigin() };
	float3 rDT{ optixGetWorldRayDirection() };
	glm::vec3 rO{ rOT.x, rOT.y, rOT.z };
	glm::vec3 rD{ rDT.x, rDT.y, rDT.z };

	glm::mat3 matFrame{ glm::mat3_cast(parameters.diskFrame) };
	glm::vec3 dC{ parameters.diskLightPosition };
	//glm::vec3 dT{ matFrame[0] };
	//glm::vec3 dB{ matFrame[1] };
	glm::vec3 dN{ matFrame[2] };
	dN = glm::dot(rD, dN) < 0.0f ? dN : -dN;
	float dR{ parameters.diskLightRadius };

	glm::vec3 o{ rO - dC };
	float t{ -glm::dot(dN, o) / glm::dot(rD, dN) };
	glm::vec3 rhP{ o + rD * t };

	bool intersect{ glm::dot(rhP, rhP) < dR * dR };
	if (intersect)
	{
		glm::vec3 hP{ rhP + dC };
		optixSetPayload_0(__float_as_uint(hP.x));
		optixSetPayload_1(__float_as_uint(hP.y));
		optixSetPayload_2(__float_as_uint(hP.z));
		optixSetPayload_3(__float_as_uint(dN.x));
		optixSetPayload_4(__float_as_uint(dN.y));
		optixSetPayload_5(__float_as_uint(dN.z));
		optixReportIntersection(t, 0);
	}
}
extern "C" __global__ void __closesthit__disk() 
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

	uint32_t payload{ *reinterpret_cast<uint32_t*>(optixGetSbtDataPointer()) };
	payload = static_cast<uint32_t>(PathStateFlagBit::EMISSIVE_OBJECT_HIT) << 16;
	optixSetPayload_6(payload);
}

extern "C" __global__ void __miss__miss()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0); 
	optixSetPayload_6(PathStateFlags(PathStateFlagBit::MISS) << 16);
}

extern "C" __device__ void __direct_callable__DielectricBxDF(const MaterialData& materialData, const DirectLightData& directLightData, const QRNG::State& qrngState, const glm::vec3& normal,
	SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathStateFlags& stateFlags)
{
	LocalTransform local{ normal } ;

	glm::vec3 locWo{ -rD };
	glm::vec3 locLi{ directLightData.lightDir };
	local.toLocal(locWo, locLi);

	glm::vec3 rand{ QRNG::Sobol::sample3D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF) };
	
	wavelengths.terminateSecondary();

	float eta{ parameters.spectrums[materialData.indexOfRefractSpectrumDataIndex].sample(wavelengths[0]) };

	float alpha{ utility::roughnessToAlpha(materialData.mfRoughnessValue) };
	microfacet::Microsurface ms{ .alphaX = alpha, .alphaY = alpha };
	if (stateFlags & PathStateFlagBit::REGULARIZED)
		ms.regularize();
	glm::vec3 wo{ locWo };
	const float cosThetaO{ LocalTransform::cosTheta(wo) };
	glm::vec3 wi{};
	glm::vec3 wm{};
	microfacet::ContextOutgoing ctxo{ microfacet::createContextOutgoing(wo) };

	if (alpha < 0.001f || eta == 1.0f)
	{
		float R{ microfacet::FReal(cosThetaO, eta) };
		float T{ glm::clamp(1.0f - R, 0.0f, 1.0f) };
		float p{ R / (R + T) };
		if (rand.z < p)
		{
			wi = {-wo.x, -wo.y, wo.z};
			bxdfPDF = p;
			throughputWeight *= R / bxdfPDF;
		}
		else
		{
			bool valid;
			float etaRel;
			wi = utility::refract(wo, glm::vec3{0.0f, 0.0f, 1.0f}, eta, valid, &etaRel);
			if (!valid)
			{
				stateFlags = stateFlags | PathStateFlagBit::PATH_TERMINATED;
				return;
			}
			stateFlags = (stateFlags & PathStateFlagBit::TRANSMISSION) ? stateFlags & (~static_cast<PathStateFlags>(PathStateFlagBit::TRANSMISSION)) : stateFlags | PathStateFlagBit::TRANSMISSION;
			bxdfPDF = cuda::std::fmax(0.0f, 1.0f - p);
			throughputWeight *= T / (etaRel * etaRel) / bxdfPDF;
		}
		stateFlags = stateFlags | PathStateFlagBit::CURRENT_HIT_SPECULAR;
		local.fromLocal(wi);
		rD = wi;
		return;
	}

	if (!directLightData.occluded)
	{
		wi = locLi;
		float cosThetaI{ LocalTransform::cosTheta(wi) };
		float t{ cosThetaO * cosThetaI };
		bool reflect{ t > 0.0f };
		float etaRel{ 1.0f };
		if (!reflect)
			etaRel = cosThetaO > 0.0f ? eta : 1.0f / eta;
		wm = wi * etaRel + wo;
		wm = glm::normalize(wm);
		wm = wm.z > 0.0f ? wm : -wm;
		float dotWmWo{ glm::dot(wm, wo) };
		float dotWmWi{ glm::dot(wm, wi) };
		if (t != 0.0f && !(dotWmWo * cosThetaO < 0.0f || dotWmWi * cosThetaI < 0.0f))
		{
			microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
			microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };
			const float R{ microfacet::FReal(glm::dot(wo, wm), eta) };
			const float T{ glm::clamp(1.0f - R, 0.0f, 1.0f) };
			const float pR{ R / (R + T) };
			const float pT{ cuda::std::fmax(0.0f, 1.0f - pR) };
			const float cosFactor{ cuda::std::fabs(cosThetaI) };
			const float wowmAbsDot{ cuda::std::fabs(dotWmWo) };

			SampledSpectrum f;
			float lbxdfPDF;
			if (reflect)
			{
				f = microfacet::D(wm, ctxm, ms) * R * microfacet::G(wi, wo, ctxi, ctxo, ms)
				    / (4.0f * cosThetaO * cosThetaI);
				lbxdfPDF = microfacet::VNDF::PDF(wo, wm, ctxo, ctxm, ms) / (4.0f * wowmAbsDot) * pR;
			}
			else
			{
				float t{ dotWmWi + dotWmWo / etaRel };
				float denom{ t * t };
				float dwmdwi{ cuda::std::fabs(dotWmWi) / denom };
				f = microfacet::D(wm, ctxm, ms) * T * microfacet::G(wi, wo, ctxi, ctxo, ms)
					* cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom))
					/ (etaRel * etaRel);
				lbxdfPDF = microfacet::VNDF::PDF(wo, wm, ctxo, ctxm, ms) * dwmdwi * pT;
			}
			L += directLightData.spectrumSample * f * cosFactor * throughputWeight
				 * MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, lbxdfPDF)
				 / directLightData.lightSamplePDF;
		}
	}

	wm = microfacet::VNDF::sample(wo, ctxo, ms, glm::vec2{rand.x, rand.y});
	microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };

	const float dotWmWo{ glm::dot(wo, wm) };
	const float R{ microfacet::FReal(dotWmWo, eta) };
	const float T{ glm::clamp(1.0f - R, 0.0f, 1.0f) };
	const float pR{ R / (R + T) };

	SampledSpectrum f;
	float cosFactor;
	if (rand.z < pR)
	{
		wi = utility::reflect(wo, wm);
		if (wo.z * wi.z <= 0.0f)
		{
			stateFlags = stateFlags | PathStateFlagBit::PATH_TERMINATED;
			return;
		}
		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
		const float cosThetaI{ LocalTransform::cosTheta(wi) };
		f = microfacet::D(wm, ctxm, ms) * R * microfacet::G(wi, wo, ctxi, ctxo, ms) 
			/ (4.0f * cosThetaO * cosThetaI);
		bxdfPDF = microfacet::VNDF::PDF(wo, wm, ctxo, ctxm, ms) / (4.0f * cuda::std::fabs(dotWmWo)) * pR;
		cosFactor = cuda::std::fabs(cosThetaI);
	}
	else
	{
		bool valid;
		float etaRel;
		wi = utility::refract(wo, wm, eta, valid, &etaRel);
		if (wo.z * wi.z >= 0.0f || !valid)
		{
			stateFlags = stateFlags | PathStateFlagBit::PATH_TERMINATED;
			return;
		}
		microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };
		const float cosThetaI{ LocalTransform::cosTheta(wi) };
		const float dotWmWi{ glm::dot(wo, wi) };
		const float t{ dotWmWi + dotWmWo / etaRel };
		const float denom{ t * t };
		const float dwmdwi{ cuda::std::fabs(dotWmWi) / denom };
		f = microfacet::D(wm, ctxm, ms) * T * microfacet::G(wi, wo, ctxi, ctxo, ms)
			* cuda::std::fabs(dotWmWo * dotWmWi / (cosThetaO * cosThetaI * denom));
		const float pT{ cuda::std::fmax(0.0f, 1.0f - pR) };
		bxdfPDF = microfacet::VNDF::PDF(wo, wm, ctxo, ctxm, ms) * dwmdwi * pT;
		stateFlags = (stateFlags & PathStateFlagBit::TRANSMISSION) ? stateFlags & (~static_cast<PathStateFlags>(PathStateFlagBit::TRANSMISSION)) : stateFlags | PathStateFlagBit::TRANSMISSION;
		cosFactor = cuda::std::fabs(cosThetaI);
	}
	throughputWeight *= f * cosFactor / bxdfPDF;

	glm::vec3 locWi{ wi };
	local.fromLocal(locWi);
	rD = locWi;
}
extern "C" __device__ void __direct_callable__ConductorBxDF(const MaterialData& materialData, const DirectLightData& directLightData, const QRNG::State& qrngState, const glm::vec3& normal,
	SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathStateFlags& stateFlags)
{
	LocalTransform local{ normal } ;

	glm::vec3 locWo{ -rD };
	glm::vec3 locLi{ directLightData.lightDir };
	local.toLocal(locWo, locLi);

	if (locWo.z == 0.0f)
	{
		stateFlags = stateFlags | PathStateFlagBit::PATH_TERMINATED;
		return;
	}

	SampledSpectrum eta{ parameters.spectrums[materialData.indexOfRefractSpectrumDataIndex].sample(wavelengths) };
	SampledSpectrum k{ parameters.spectrums[materialData.absorpCoefSpectrumDataIndex].sample(wavelengths) };

	float alpha{ utility::roughnessToAlpha(materialData.mfRoughnessValue) };
	microfacet::Microsurface ms{ .alphaX = alpha, .alphaY = alpha };
	if (stateFlags & PathStateFlagBit::REGULARIZED)
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
		stateFlags = stateFlags | PathStateFlagBit::CURRENT_HIT_SPECULAR;
		local.fromLocal(wi);
		rD = wi;
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
			float lbxdfPDF{ microfacet::VNDF::PDF(wo, wm, ctxo, ctxm, ms) / (4.0f * wowmAbsDot) };
			L += directLightData.spectrumSample * f * cosFactor * throughputWeight
				* MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, lbxdfPDF)
				/ directLightData.lightSamplePDF;
		}
	}

	glm::vec2 rand{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF) };
	wm = microfacet::VNDF::sample(wo, ctxo, ms, rand);
	microfacet::ContextMicronormal ctxm{ microfacet::createContextMicronormal(wm) };

	wi = utility::reflect(wo, wm);
	microfacet::ContextIncident ctxi{ microfacet::createContextIncident(wi) };

	if (wo.z * wi.z <= 0.0f)
	{
		stateFlags = stateFlags | PathStateFlagBit::PATH_TERMINATED;
		return;
	}

	const float wowmAbsDot{ cuda::std::fabs(glm::dot(wo, wm)) };

	SampledSpectrum f{ microfacet::D(wm, ctxm, ms) * microfacet::FComplex(wowmAbsDot, eta, k) * microfacet::G(wi, wo, ctxi, ctxo, ms) 
					   / cuda::std::fabs(4.0f * LocalTransform::cosTheta(wo) * LocalTransform::cosTheta(wi)) };
	float cosFactor{ cuda::std::fabs(LocalTransform::cosTheta(wi)) };
	bxdfPDF = microfacet::VNDF::PDF(wo, wm, ctxo, ctxm, ms) / (4.0f * wowmAbsDot);
	throughputWeight *= f * cosFactor / bxdfPDF;

	glm::vec3 locWi{ wi };
	local.fromLocal(locWi);
	rD = locWi;
}

extern "C" __global__ void __raygen__main()
{
	uint3 li{ optixGetLaunchIndex() };
	glm::vec2 pixelCoordinate{ static_cast<float>(li.x), static_cast<float>(li.y) };

	//
	glm::vec3 deb{};
	//

	QRNG::State qrngState{ parameters.samplingState.offset, QRNG::getPixelHash(li.x, li.y) };

	glm::dvec4 result{ parameters.renderingData[li.y * parameters.filmWidth + li.x] };
	uint32_t sample{ 0 };
	bool terminated{ false };
	do
	{
		const glm::vec2 subsampleOffset{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::FILTER) };
		const float xScale{ 2.0f * ((pixelCoordinate.x + subsampleOffset.x) * parameters.invFilmWidth) - 1.0f };
		const float yScale{ 2.0f * ((pixelCoordinate.y + subsampleOffset.y) * parameters.invFilmHeight) - 1.0f };
		glm::vec3 rD{ glm::normalize(parameters.camW
		+ parameters.camU * xScale * parameters.camPerspectiveScaleW
		+ parameters.camV * yScale * parameters.camPerspectiveScaleH) };
		glm::vec3 rO{ 0.0 };

		SampledWavelengths wavelengths{ SampledWavelengths::sampleVisible(QRNG::Sobol::sample1D(qrngState, QRNG::DimensionOffset::WAVELENGTH)) };
		SampledSpectrum L{ 0.0f };
		SampledSpectrum throughputWeight{ 1.0f };
		float bxdfPDF{ 1.0f };
		float refractionScale{ 1.0f };
		PathStateFlags stateFlags{ 0 };
		uint32_t depth{ 0 };
		do
		{
			uint32_t pl0, pl1, pl2, pl3, pl4, pl5, pl6;
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
					   pl0, pl1, pl2, pl3, pl4, pl5, pl6); //Payload

			glm::vec3 hP; //Hit position
			glm::vec3 hN; //Hit normal
			MaterialData* material;
			unpackTraceData(parameters, hP, hN, stateFlags, &material,
				pl0, pl1, pl2, pl3, pl4, pl5, pl6);
			glm::vec3 toHit{ hP - rO };
			float dToHSqr{ toHit.x * toHit.x + toHit.y * toHit.y + toHit.z * toHit.z };
			rO = utility::offsetRay(hP, hN);

			if (stateFlags & PathStateFlagBit::MISS)
			{
				/*SampledSpectrum Lo{ sampleSpectrum(wavelengths) };
				float emissionWeight{ 1.0f };
				if (depth != 0 && !(stateFlags & PathStateFlagBit::PREVIOUS_HIT_SPECULAR))
					emissionWeight = MIS::powerHeuristic(1, bxdfPDF, 1, lightPDF);
				L += throughputWeight * Lo * emissionWeight;*/
				
				goto breakPath;
			}


			if (stateFlags & PathStateFlagBit::EMISSIVE_OBJECT_HIT)
			{
				SampledSpectrum Le{ parameters.spectrums[material->emissionSpectrumDataIndex].sample(wavelengths) * parameters.lightScale };
				float emissionWeight{};
				const float lightCos{ -glm::dot(rD, parameters.diskNormal) };

				if (lightCos <= 0.0f)
					emissionWeight = 0.0f;
				else if (depth == 0 || (stateFlags & PathStateFlagBit::PREVIOUS_HIT_SPECULAR))
					emissionWeight = 1.0f;
				else
				{
					float lPDF{ parameters.diskSurfacePDF * dToHSqr / lightCos };
					emissionWeight = MIS::powerHeuristic(1, bxdfPDF, 1, lPDF);
				}
				L += throughputWeight * Le * emissionWeight;
			}

			// Fill DirectLightData
			DirectLightData directLightData{};
			directLightData.spectrumSample =
				parameters.spectrums[parameters.illuminantSpectralDistributionIndex].sample(wavelengths) * parameters.lightScale;
			glm::vec3 rToLight{
				(parameters.diskLightPosition + sampling::disk::sampleUniform3D(glm::vec2{QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::LIGHT)}, parameters.diskFrame) * parameters.diskLightRadius) - rO };
			float sqrdToLight{ rToLight.x * rToLight.x + rToLight.y * rToLight.y + rToLight.z * rToLight.z };
			const float dToL{ cuda::std::sqrtf(sqrdToLight) };
			directLightData.lightDir = rToLight / dToL;
			float lCos{ -glm::dot(parameters.diskNormal, directLightData.lightDir) };

			if (lCos > 0.0f)
			{
				directLightData.lightSamplePDF = parameters.diskSurfacePDF * sqrdToLight / lCos;
				const glm::vec3& lD{ directLightData.lightDir };
				optixTraverse(parameters.traversable,
						{ rO.x, rO.y, rO.z },
						{ lD.x, lD.y, lD.z },
						0.0f,
						dToL - 0.01f,
						0.0f,
						0xFF,
						OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
						0,
						0,
						0);
				directLightData.occluded = optixHitObjectIsHit();
			}
			else
				directLightData.occluded = true;

			// Launch BxDF evaluation
			optixDirectCall<void, 
				const MaterialData&, const DirectLightData&, const QRNG::State&,
				const glm::vec3&,
				SampledSpectrum&, SampledWavelengths&, SampledSpectrum&,
				float&, glm::vec3&, PathStateFlags&>
				(material->bxdfIndexSBT,
				 *material, directLightData, qrngState,
				 hN,
				 L, wavelengths, throughputWeight,
				 bxdfPDF, rD, stateFlags);

			if (stateFlags & PathStateFlagBit::TRANSMISSION)
				rO = utility::offsetRay(hP, -hN);

			qrngState.advanceBounce();
			updateStateFlags(stateFlags);
			terminated = stateFlags & PathStateFlagBit::PATH_TERMINATED;
		} while (++depth < parameters.maxPathDepth && !terminated);
	breakPath:
		qrngState.advanceSample();

		if (!terminated)
		{
			resolveSample(L, wavelengths.getPDF());
			result += glm::dvec4{color::toRGB(*parameters.sensorSpectralCurveA, *parameters.sensorSpectralCurveB, *parameters.sensorSpectralCurveC,
					wavelengths, L), filter::computeFilterWeight(subsampleOffset)};
		}
	} while (++sample < parameters.samplingState.count);
	//result = glm::dvec4{ deb, 1.0f };
	parameters.renderingData[li.y * parameters.filmWidth + li.x] = result;
}
