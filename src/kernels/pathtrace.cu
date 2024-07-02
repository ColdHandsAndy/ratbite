#include <cuda_runtime.h>
#include <optix_device.h>
#include <cuda/std/cstdint>
#include <cuda/std/cmath>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../core/launch_parameters.h"
#include "../core/util_macros.h"
#include "../core/material.h"
#include "../device/geometry.h"
#include "../device/filter.h"
#include "../device/color.h"
#include "../device/quasi_random.h"
#include "../device/spectral.h"
#include "../device/sampling.h"
#include "../device/bxdf.h"
#include "../device/mis.h"

typedef uint32_t PathTracingStateFlags;
enum class PathTracingStateFlagBit : uint32_t
{
	NO_FLAGS = 0u,
	PREVIOUS_BOUNCE_SPECULAR = 1u,
	CURRENT_BOUNCE_SPECULAR = 2u,
	MISS = 4u,
	TRANSMISSION = 8u,
	EMISSIVE_OBJECT_HIT = 16u,
	SECONDARY_SPECTRAL_SAMPLES_TERMINATED = 32u
};
STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_WITH_PREFIX(PathTracingStateFlags, PathTracingStateFlagBit, CU_DEVICE CU_INLINE)

extern "C"
{
	__constant__ LaunchParameters parameters{};
}

struct DirectLightData
{
	SampledSpectrum spectrumSample{};
	glm::vec3 lightDir{};
	float distToLight{};
	float lightSamplePDF{};
};

CU_DEVICE CU_INLINE void unpackTraceData(const LaunchParameters& params, glm::vec3& hP, glm::vec3& hN, PathTracingStateFlags& stateFlags, MaterialData** materialDataPtr,
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
	PathTracingStateFlags excludeFlags{ PathTracingStateFlagBit::EMISSIVE_OBJECT_HIT | PathTracingStateFlagBit::CURRENT_BOUNCE_SPECULAR };
	PathTracingStateFlags includeFlags{ stateFlags & PathTracingStateFlagBit::CURRENT_BOUNCE_SPECULAR ?
		static_cast<PathTracingStateFlags>(PathTracingStateFlagBit::PREVIOUS_BOUNCE_SPECULAR) : static_cast<PathTracingStateFlags>(PathTracingStateFlagBit::NO_FLAGS) };

	stateFlags = ((stateFlags | excludeFlags) ^ excludeFlags) | includeFlags;
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
	payload = static_cast<uint32_t>(PathTracingStateFlagBit::EMISSIVE_OBJECT_HIT) << 16;
	optixSetPayload_6(payload);
}

extern "C" __global__ void __miss__miss()
{
	optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0); 
	optixSetPayload_6(PathTracingStateFlags(PathTracingStateFlagBit::MISS) << 16);
}

//extern "C" __device__ void __direct_callable__DielectricBxDF(const DirectLightData& directLightData, const glm::vec3& normal,
//	SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathTracingStateFlags& stateFlags)
//{
//
//}
//extern "C" __device__ void __direct_callable__ConductorBxDF(const DirectLightData& directLightData, const glm::vec3& normal,
//	SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathTracingStateFlags& stateFlags)
//{
//
//}
extern "C" __device__ void __direct_callable__temp(const MaterialData& materialData, const DirectLightData& directLightData, const glm::vec3& rO, const glm::vec3& normal,
	const glm::vec2& rP, SampledSpectrum& L, SampledWavelengths& wavelengths, SampledSpectrum& throughputWeight, float& bxdfPDF, glm::vec3& rD, PathTracingStateFlags& stateFlags, glm::vec3& deb, uint32_t depth)
{
	const glm::vec3& lD{ directLightData.lightDir };
	const float& dToL{ directLightData.distToLight };

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
		0
	);
	const bool lOccluded{ optixHitObjectIsHit() || directLightData.lightSamplePDF == 0.0f };

	geometry::LocalTransform local{ normal } ;

	glm::vec3 locN{ normal };
	glm::vec3 locWo{ -rD };
	glm::vec3 locLi{ lD };
	// Compare between templated and non-templated versions of the function
	local.toLocal(locN, locWo, locLi);

	glm::vec2 uvD{ sampling::disk::sampleUniform2D(rP) };
	glm::vec3 locWi{ uvD.x, uvD.y, cuda::std::sqrtf(glm::clamp(1.0f - uvD.x * uvD.x - uvD.y * uvD.y, 0.0f, 1.0f)) };

	SampledSpectrum eta{ parameters.spectrums[materialData.indexOfRefractSpectrumDataIndex].sample(wavelengths) };
	SampledSpectrum k{ parameters.spectrums[materialData.absorpCoefSpectrumDataIndex].sample(wavelengths) };

	auto OrenNayar{ [](const glm::vec3& ld, const glm::vec3& vd, const glm::vec3& sn, float r, float a) -> float
		{
			float LdotV = glm::dot(ld, vd);
			float NdotL = glm::dot(ld, sn);
			float NdotV = glm::dot(sn, vd);

			float s = LdotV - NdotL * NdotV;

			float sigma2 = r * r;
			float A = 1.0f - 0.5f * (sigma2 / (((sigma2 + 0.33f) + 0.000001f)));
			float B = 0.45f * sigma2 / ((sigma2 + 0.09f) + 0.00001f);

			float ga = glm::dot(vd - sn * NdotV,sn - sn * NdotL);

			return glm::max(0.0f, NdotL) * (A + B * glm::max(0.0f, ga) * cuda::std::sqrtf(glm::max((1.0f - NdotV * NdotV) * (1.0f - NdotL * NdotL), 0.0f)) / glm::max(NdotL, NdotV));
		} };

	if (!lOccluded)
	{
		float lbxdfPDF{ cuda::std::abs(locLi.z) / (1.0f / glm::pi<float>()) };
		L += OrenNayar(locLi, locWo, locN, 1.0f, 1.0f) * directLightData.spectrumSample
			* throughputWeight * microfacet::FComplex(locLi.z, eta, k)
			* MIS::powerHeuristic(1, directLightData.lightSamplePDF, 1, lbxdfPDF)
			/ directLightData.lightSamplePDF;
	}

	bxdfPDF = cuda::std::abs(locWi.z) * (1.0f / glm::pi<float>());
	throughputWeight *= microfacet::FComplex(locWi.z, eta, k) * OrenNayar(locWi, locWo, locN, 1.0f, 1.0f) / bxdfPDF;

	//deb = glm::vec3(cuda::std::abs(rP.x), cuda::std::abs(rP.y), cuda::std::abs(0.0f));

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
		PathTracingStateFlags stateFlags{ 0 };
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
			float dToLSqr{ toHit.x * toHit.x + toHit.y * toHit.y + toHit.z * toHit.z };
			rO = geometry::offsetRay(hP, hN);

			if (stateFlags & PathTracingStateFlagBit::MISS)
			{
				/*SampledSpectrum Lo{ sampleSpectrum(wavelengths) };
				float emissionWeight{ 1.0f };
				if (depth != 0 && !(stateFlags & PathTracingStateFlagBit::PREVIOUS_BOUNCE_SPECULAR))
					emissionWeight = MIS::powerHeuristic(1, bxdfPDF, 1, lightPDF);
				L += throughputWeight * Lo * emissionWeight;*/
				
				goto breakPath;
			}

			if (stateFlags & PathTracingStateFlagBit::EMISSIVE_OBJECT_HIT)
			{
				SampledSpectrum Le{ parameters.spectrums[material->emissionSpectrumDataIndex].sample(wavelengths) };
				float emissionWeight{ 1.0f };
				const float lightCos{ glm::dot(rD, parameters.diskNormal) };
				// Hit emission check
				if (lightCos > 0.0f)
					emissionWeight = 0.0f;
				else if (depth != 0 && !(stateFlags & PathTracingStateFlagBit::PREVIOUS_BOUNCE_SPECULAR))
				{
					float lPDF{ parameters.diskSurfacePDF * dToLSqr / cuda::std::abs(lightCos) };
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
			directLightData.distToLight = cuda::std::sqrtf(sqrdToLight);
			rToLight /= directLightData.distToLight;
			directLightData.lightDir = rToLight;
			float lCos{ -glm::dot(parameters.diskNormal, directLightData.lightDir) };
			directLightData.lightSamplePDF = lCos > 0.0f ? parameters.diskSurfacePDF * sqrdToLight / lCos : 0.0f;

			// Generate quasi-random values
			glm::vec2 u{ QRNG::Sobol::sample2D(qrngState, QRNG::DimensionOffset::SURFACE_BXDF) };
			// Launch BxDF evaluation
			optixDirectCall<void, 
				const MaterialData&, const DirectLightData&,
				const glm::vec3&, const glm::vec3&, const glm::vec2&,
				SampledSpectrum&, SampledWavelengths&, SampledSpectrum&,
				float&, glm::vec3&, PathTracingStateFlags&,
				glm::vec3&, uint32_t>
				(material->bxdfIndexSBT,
				 *material,directLightData,
				 rO, hN, u,
				 L, wavelengths, throughputWeight,
				 bxdfPDF, rD, stateFlags,
				 deb, depth);

			qrngState.advanceBounce();
			updateStateFlags(stateFlags);
		} while (++depth < parameters.maxPathDepth);
	breakPath:
		qrngState.advanceSample();
		resolveSample(L, wavelengths.getPDF());

		result += glm::dvec4{color::toRGB(*parameters.sensorSpectralCurveA, *parameters.sensorSpectralCurveB, *parameters.sensorSpectralCurveC,
										  wavelengths, L), filter::computeFilterWeight(subsampleOffset)};
	} while (++sample < parameters.samplingState.count);
	//result = glm::dvec4{ deb, 1.0f };
	parameters.renderingData[li.y * parameters.filmWidth + li.x] = result;
}
