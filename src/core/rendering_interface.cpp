#pragma once

#define NOMINMAX

#include "rendering_interface.h"

#include <fstream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <functional>
#include <tuple>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

#include "../core/scene.h"
#include "../core/camera.h"
#include "../core/window.h"
#include "../core/render_context.h"
#include "../core/launch_parameters.h"
#include "../core/light_tree_types.h"
#include "../core/light_tree.h"
#include "../core/util.h"
#include "../core/debug_macros.h"
#include "../core/callbacks.h"
#include "../kernels/optix_programs_desc.h"


void RenderingInterface::createOptixContext()
{
	OptixDeviceContextOptions options{ .logCallbackFunction = optixLogCallback, .logCallbackLevel = 4 };

#ifdef _DEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

	OPTIX_CHECK(optixDeviceContextCreate(0, &options, &m_context));
}
void RenderingInterface::fillModelMaterials(RenderingInterface::ModelResource& modelRes, SceneData::Model& model)
{
	// Uploading images and textures
	for(auto& imData : model.imageData)
	{
		TextureType texType{};
		if (imData.channelCount == 1)
			texType = TextureType::R8_UNORM;
		else if (imData.channelCount == 4)
			texType = TextureType::R8G8B8A8_UNORM;
		else
			R_ERR_LOG("Invalid image channel count\n");
		modelRes.images.emplace_back(imData.width, imData.height, 1, texType)
			.fill(imData.data, 0, 0, 0, imData.width, imData.height, 1, cudaMemcpyHostToDevice);
	}
	for(auto& texData : model.textureData)
	{
		modelRes.textures.emplace_back(modelRes.images[texData.imageIndex], texData.addressX, texData.addressY, TextureAddress::CLAMP,
				texData.filter, texData.sRGB);
	}
	model.clearImageData();

	// Uploading spectra and converting "MaterialDescriptor" to "MaterialData". Uploading vertex attributes
	int cur{ 0 };
	for(auto& mesh : model.meshes)
		for(auto& submesh : mesh.submeshes)
		{
			const SceneData::MaterialDescriptor desc{ model.materialDescriptors[cur++] };
			MaterialData mat{};

			mat.bxdfIndexSBT = bxdfTypeToIndex(desc.bxdf);

			mat.doubleSided = desc.doubleSided;

			// Spectral Material
			mat.mfRoughnessValue = desc.roughness;
			mat.indexOfRefractSpectrumDataIndex = changeSpectrum(desc.baseIOR, SpectralData::SpectralDataType::NONE);
			mat.absorpCoefSpectrumDataIndex = changeSpectrum(desc.baseAC, SpectralData::SpectralDataType::NONE);
			mat.emissionSpectrumDataIndex = changeSpectrum(desc.baseEmission, SpectralData::SpectralDataType::NONE);
			// Triplet Material
			if (desc.baseColorTextureIndex != -1)
			{
				mat.textures |= MaterialData::TextureTypeBitfield::BASE_COLOR;
				mat.baseColorTexture = modelRes.textures[desc.baseColorTextureIndex].getTextureObject();
				mat.bcTexCoordSetIndex = desc.bcTexCoordIndex == 1; // Only two texture coordinate sets supported
			}
			if (desc.bcFactorPresent)
			{
				mat.factors |= MaterialData::FactorTypeBitfield::BASE_COLOR;
				mat.baseColorFactor[0] = desc.baseColorFactor[0];
				mat.baseColorFactor[1] = desc.baseColorFactor[1];
				mat.baseColorFactor[2] = desc.baseColorFactor[2];
				mat.baseColorFactor[3] = desc.baseColorFactor[3];
			}
			if (desc.alphaInterpretation == SceneData::MaterialDescriptor::AlphaInterpretation::CUTOFF)
			{
				mat.factors |= MaterialData::FactorTypeBitfield::CUTOFF;
				mat.alphaCutoff = desc.alphaCutoff;
			}
			else if (desc.alphaInterpretation == SceneData::MaterialDescriptor::AlphaInterpretation::BLEND)
			{
				mat.factors |= MaterialData::FactorTypeBitfield::BLEND;
				mat.alphaCutoff = desc.alphaCutoff;
			}
			if (desc.metalRoughnessTextureIndex != -1)
			{
				mat.textures |= MaterialData::TextureTypeBitfield::MET_ROUGH;
				mat.pbrMetalRoughnessTexture = modelRes.textures[desc.metalRoughnessTextureIndex].getTextureObject();
				mat.mrTexCoordSetIndex = desc.mrTexCoordIndex == 1;
			}
			if (desc.metFactorPresent)
			{
				mat.factors |= MaterialData::FactorTypeBitfield::METALNESS;
				mat.metalnessFactor = desc.metalnessFactor;
			}
			if (desc.roughFactorPresent)
			{
				mat.factors |= MaterialData::FactorTypeBitfield::ROUGHNESS;
				mat.roughnessFactor = desc.roughnessFactor;
			}
			if (desc.normalTextureIndex != -1)
			{
				mat.textures |= MaterialData::TextureTypeBitfield::NORMAL;
				mat.normalTexture = modelRes.textures[desc.normalTextureIndex].getTextureObject();
				mat.nmTexCoordSetIndex = desc.nmTexCoordIndex == 1;
			}
			if (desc.transmissionTextureIndex != -1)
			{
				mat.textures |= MaterialData::TextureTypeBitfield::TRANSMISSION;
				mat.transmissionTexture = modelRes.textures[desc.transmissionTextureIndex].getTextureObject();
				mat.trTexCoordSetIndex = desc.trTexCoordIndex == 1;
			}
			if (desc.transmitFactorPresent)
			{
				mat.factors |= MaterialData::FactorTypeBitfield::TRANSMISSION;
				mat.transmissionFactor = desc.transmitFactor;
			}
			if (desc.sheenPresent)
			{
				mat.sheenPresent = true;
				if (desc.sheenColorFactorPresent)
				{
					mat.factors |= MaterialData::FactorTypeBitfield::SHEEN_COLOR;
					mat.sheenColorFactor[0] = desc.sheenColorFactor[0];
					mat.sheenColorFactor[1] = desc.sheenColorFactor[1];
					mat.sheenColorFactor[2] = desc.sheenColorFactor[2];
				}
				if (desc.sheenColorTextureIndex != -1)
				{
					mat.textures |= MaterialData::TextureTypeBitfield::SHEEN_COLOR;
					mat.sheenColorTexture = modelRes.textures[desc.sheenColorTextureIndex].getTextureObject();
					mat.shcTexCoordSetIndex = desc.shcTexCoordIndex == 1;
				}
				if (desc.sheenRoughnessFactorPresent)
				{
					mat.factors |= MaterialData::FactorTypeBitfield::SHEEN_ROUGH;
					mat.sheenRoughnessFactor = desc.sheenRoughnessFactor;
				}
				if (desc.sheenRoughTextureIndex != -1)
				{
					mat.textures |= MaterialData::TextureTypeBitfield::SHEEN_ROUGH;
					mat.sheenRoughTexture = modelRes.textures[desc.sheenRoughTextureIndex].getTextureObject();
					mat.shrTexCoordSetIndex = desc.shrTexCoordIndex == 1;
				}
			}
			if (desc.emissiveFactorPresent)
			{
				mat.factors |= MaterialData::FactorTypeBitfield::EMISSION;
				mat.emissiveFactor[0] = desc.emissiveFactor[0];
				mat.emissiveFactor[1] = desc.emissiveFactor[1];
				mat.emissiveFactor[2] = desc.emissiveFactor[2];
			}
			mat.ior = desc.ior;


			mat.indexType = submesh.indexType;
			CUdeviceptr& iBuf{ modelRes.indexBuffers.emplace_back() };
			size_t iBufferByteSize{ (submesh.indexType == IndexType::UINT_32 ? sizeof(uint32_t) * 3 : sizeof(uint16_t) * 3) * submesh.primitiveCount };
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(iBuf)),
						iBufferByteSize));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(iBuf), submesh.indices,
						iBufferByteSize,
						cudaMemcpyHostToDevice));
			mat.indices = iBuf;


			uint8_t attributeOffset{ 0 };
			CUdeviceptr& aBuf{ modelRes.attributeBuffers.emplace_back() };
			void* packedAttribBuf{};
			bool normalAttr{ false };
			bool frameAttr{ false };
			bool colorAttr{ false };
			bool texCoordsAttr1{ false };
			bool texCoordsAttr2{ false };
			if (submesh.normals.size() != 0 && submesh.tangents.size() != 0)
			{
				frameAttr = true;
				mat.attributes |= MaterialData::AttributeTypeBitfield::FRAME;
				mat.frameOffset = attributeOffset;
				attributeOffset += sizeof(float) * 4;
			}
			else if (submesh.normals.size() != 0)
			{
				normalAttr = true;
				mat.attributes |= MaterialData::AttributeTypeBitfield::NORMAL;
				mat.normalOffset = attributeOffset;
				attributeOffset += sizeof(float) * 2;
			}
			if (submesh.texCoordsSets.size() >= 1)
			{
				texCoordsAttr1 = true;
				mat.attributes |= MaterialData::AttributeTypeBitfield::TEX_COORD_1;
				mat.texCoord1Offset = attributeOffset;
				attributeOffset += sizeof(float) * 2;
			}
			if (submesh.texCoordsSets.size() >= 2)
			{
				texCoordsAttr2 = true;
				mat.attributes |= MaterialData::AttributeTypeBitfield::TEX_COORD_2;
				mat.texCoord2Offset = attributeOffset;
				attributeOffset += sizeof(float) * 2;
			}
			mat.attributeStride = attributeOffset;
			packedAttribBuf = malloc(mat.attributeStride * submesh.vertexCount);
			uint8_t* attr{ reinterpret_cast<uint8_t*>(packedAttribBuf) };
			auto encodeNormal{ [](const glm::vec3& vec) {
				const float& x{ vec.x };
				const float& y{ vec.y };
				const float& z{ vec.z };

				glm::vec2 p{ glm::vec2{x, y} * (1.0f / (std::fabs(x) + std::fabs(y) + std::fabs(z))) };
				glm::vec2 res{
					z <= 0.0f
						?
						(glm::vec2{1.0f} - glm::vec2{std::fabs(p.y), std::fabs(p.x)}) * glm::vec2{(p.x >= 0.0f) ? +1.0f : -1.0f, (p.y >= 0.0f) ? +1.0f : -1.0f}
					:
						p
				};
				return res;
			} };
			for (uint32_t i{ 0 }; i < submesh.vertexCount; ++i)
			{
				if (normalAttr)
					*reinterpret_cast<glm::vec2*>(attr + mat.normalOffset) = encodeNormal(submesh.normals[i]);
				if (frameAttr)
				{
					glm::quat frame{ glm::quat_cast(glm::mat3{
							submesh.tangents[i],
							glm::cross(glm::vec3{submesh.normals[i]}, glm::vec3{submesh.tangents[i]}),
							submesh.normals[i],
							}) };
					if (frame.w == 0.0f)
					{
						union
						{
							float f;
							uint32_t u;
						} signAdjustedFloat{};
						signAdjustedFloat.f = frame.w;
						if (submesh.tangents[i].w > 0.0f)
							signAdjustedFloat.u = signAdjustedFloat.u & 0x7FFFFFFF;
						else
							signAdjustedFloat.u = signAdjustedFloat.u | 0x80000000;
						frame.w = signAdjustedFloat.f;
					}
					else if (frame.w * submesh.tangents[i].w < 0.0f)
						frame = -frame;
					*reinterpret_cast<glm::quat*>(attr + mat.frameOffset) = frame;
				}
				// if (colorAttr)
				// 	*reinterpret_cast<*>() = ;
				if (texCoordsAttr1)
					*reinterpret_cast<glm::vec2*>(attr + mat.texCoord1Offset) = submesh.texCoordsSets[0][i];
				if (texCoordsAttr2)
					*reinterpret_cast<glm::vec2*>(attr + mat.texCoord2Offset) = submesh.texCoordsSets[1][i];

				attr = attr + mat.attributeStride;
			}

			size_t aBufferByteSize{ mat.attributeStride * submesh.vertexCount };
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(aBuf)),
						aBufferByteSize));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(aBuf), packedAttribBuf,
						aBufferByteSize,
						cudaMemcpyHostToDevice));
			mat.attributeData = aBuf;
			free(packedAttribBuf);

			modelRes.materialIndices.push_back(addMaterial(mat));
		}
	model.clearVertexData();
}
uint32_t RenderingInterface::fillLightMaterial(const SceneData::MaterialDescriptor& desc)
{
	MaterialData mat{};

	mat.bxdfIndexSBT = bxdfTypeToIndex(desc.bxdf);

	mat.doubleSided = desc.doubleSided;

	// Spectral Material
	mat.mfRoughnessValue = desc.roughness;
	mat.indexOfRefractSpectrumDataIndex = changeSpectrum(desc.baseIOR, SpectralData::SpectralDataType::NONE);
	mat.absorpCoefSpectrumDataIndex = changeSpectrum(desc.baseAC, SpectralData::SpectralDataType::NONE);
	mat.emissionSpectrumDataIndex = changeSpectrum(desc.baseEmission, SpectralData::SpectralDataType::NONE);

	return addMaterial(mat);
}
void RenderingInterface::uploadLightData(const SceneData& scene, const glm::vec3& cameraPosition, bool resizeBuffers)
{
	if (scene.diskLights.size() != 0)
	{
		if (resizeBuffers)
		{
			if (m_diskLights != CUdeviceptr{})
				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_diskLights)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_diskLights), scene.diskLights.size() * sizeof(DiskLightData)));
			m_launchParameters.lights.disks = m_diskLights;
		}
		DiskLightData* diskLightData{ new DiskLightData[scene.diskLights.size()] };
		for (int i{ 0 }; i < scene.diskLights.size(); ++i)
		{
			const SceneData::DiskLight& dl{ scene.diskLights[i] };
			uint32_t matIndex{ m_lightResources[dl.getID()].materialIndex };
			diskLightData[i] = {
				.position = dl.getPosition() - cameraPosition,
				.powerScale = dl.getPowerScale(),
				.frame = dl.getFrame(),
				.radius = dl.getRadius(),
				.materialIndex = matIndex};
		}
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_diskLights), diskLightData, scene.diskLights.size() * sizeof(DiskLightData), cudaMemcpyHostToDevice));
		delete[] diskLightData;
	}

	if (scene.sphereLights.size() != 0)
	{
		if (resizeBuffers)
		{
			if (m_sphereLights != CUdeviceptr{})
				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sphereLights)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sphereLights), scene.sphereLights.size() * sizeof(SphereLightData)));
			m_launchParameters.lights.spheres = m_sphereLights;
		}
		SphereLightData* sphereLightData{ new SphereLightData[scene.sphereLights.size()] };
		for (int i{ 0 }; i < scene.sphereLights.size(); ++i)
		{
			const SceneData::SphereLight& sl{ scene.sphereLights[i] };
			uint32_t matIndex{ m_lightResources[sl.getID()].materialIndex };
			sphereLightData[i] = {
				.position = sl.getPosition() - cameraPosition,
				.powerScale = sl.getPowerScale(),
				.frame = sl.getFrame(),
				.radius = sl.getRadius(),
				.materialIndex = matIndex};
		}
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sphereLights), sphereLightData, scene.sphereLights.size() * sizeof(SphereLightData), cudaMemcpyHostToDevice));
		delete[] sphereLightData;
	}

	if (resizeBuffers)
	{
		uint16_t orderedSizes[KSampleableLightCount]{};
		orderedSizes[KSphereLightIndex] = scene.sphereLights.size();
		orderedSizes[KDiskLightIndex] = scene.diskLights.size();
		if (m_launchParameters.lights.orderedCount == CUdeviceptr{})
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.lights.orderedCount), sizeof(uint16_t) * ARRAYSIZE(orderedSizes)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_launchParameters.lights.orderedCount), orderedSizes, sizeof(uint16_t) * ARRAYSIZE(orderedSizes), cudaMemcpyHostToDevice));

		m_launchParameters.lights.lightCount = static_cast<float>(scene.getLightCount());
	}
}
void RenderingInterface::createModulesProgramGroupsPipeline()
{
	OptixPayloadType payloadTypes[1]{};
	payloadTypes[0].numPayloadValues = Program::payloadValueCount;
	payloadTypes[0].payloadSemantics = Program::payloadSemantics;
	OptixModuleCompileOptions moduleCompileOptions{ .optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT, 
		.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE, .numBoundValues = 0, 
		.numPayloadTypes = ARRAYSIZE(payloadTypes), .payloadTypes = payloadTypes};
#ifdef _DEBUG
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

	OptixPipelineCompileOptions pipelineCompileOptions{ 
		.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
#ifdef _DEBUG
			.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
#endif
			.pipelineLaunchParamsVariableName = "parameters",
			.usesPrimitiveTypeFlags = static_cast<uint32_t>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE) };

	std::ifstream ifstr{ getExeDir() / "pathtrace.ptx", std::ios_base::binary | std::ios_base::ate };
	size_t inputSize{ static_cast<size_t>(ifstr.tellg()) };
	char* input{ new char[inputSize] };
	ifstr.seekg(0);
	ifstr.read(input, inputSize);

	OptixModule& optixModule{ m_ptModule };
	OPTIX_CHECK_LOG(optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, input, inputSize, OPTIX_LOG, &OPTIX_LOG_SIZE, &optixModule));
	delete[] input;

	OptixBuiltinISOptions builtInSphereModuleCompileOptions{ .builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE, .usesMotionBlur = false };
	OPTIX_CHECK_LOG(optixBuiltinISModuleGet(m_context, &moduleCompileOptions, &pipelineCompileOptions, &builtInSphereModuleCompileOptions, &m_builtInSphereModule));


	OptixProgramGroupOptions programGroupOptions{ .payloadType = payloadTypes + 0 };
	{
		OptixProgramGroupDesc descs[m_ptProgramGroupCount]{};
		descs[RenderingInterface::RAYGEN] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
			.raygen = {.module = optixModule, .entryFunctionName = Program::raygenName} };
		descs[RenderingInterface::MISS] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
			.raygen = {.module = optixModule, .entryFunctionName = Program::missName} };
		descs[RenderingInterface::TRIANGLE] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
			.hitgroup = {.moduleCH = optixModule, .entryFunctionNameCH = Program::closehitTriangleName} };
		descs[RenderingInterface::DISK] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
			.hitgroup = {.moduleCH = optixModule, .entryFunctionNameCH = Program::closehitDiskName, .moduleIS = optixModule,.entryFunctionNameIS = Program::intersectionDiskName} };
		descs[RenderingInterface::SPHERE] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
			.hitgroup = {.moduleCH = optixModule, .entryFunctionNameCH = Program::closehitSphereName, .moduleIS = m_builtInSphereModule} };
		descs[RenderingInterface::PURE_CONDUCTOR_BXDF] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
			.callables = {.moduleDC = optixModule, .entryFunctionNameDC = Program::pureConductorBxDFName} };
		descs[RenderingInterface::PURE_DIELECTRIC_BXDF] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
			.callables = {.moduleDC = optixModule, .entryFunctionNameDC = Program::pureDielectricBxDFName} };
		descs[RenderingInterface::COMPLEX_SURFACE_BXDF] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
			.callables = {.moduleDC = optixModule, .entryFunctionNameDC = Program::complexSurfaceBxDFName} };
		OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, descs, ARRAYSIZE(descs),
					&programGroupOptions,
					OPTIX_LOG, &OPTIX_LOG_SIZE,
					m_ptProgramGroups));
	}
	OptixPipelineLinkOptions pipelineLinkOptions{ .maxTraceDepth = Program::maxTraceDepth };

	OPTIX_CHECK_LOG(optixPipelineCreate(m_context,
				&pipelineCompileOptions,
				&pipelineLinkOptions,
				m_ptProgramGroups,
				m_ptProgramGroupCount,
				OPTIX_LOG, &OPTIX_LOG_SIZE,
				&m_pipeline));

	OptixStackSizes stackSizes{};
	for (auto pg : m_ptProgramGroups)
		OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, m_pipeline));

	uint32_t dcStackSizeTraversal{};
	uint32_t dcStackSizeState{};
	uint32_t ccStackSize{};
	OPTIX_CHECK(optixUtilComputeStackSizes(
				&stackSizes,
				Program::maxTraceDepth,
				Program::maxCCDepth,
				Program::maxDCDepth,
				&dcStackSizeTraversal,
				&dcStackSizeState,
				&ccStackSize));

	const uint32_t maxTraversalDepth{ 2 };
	OPTIX_CHECK(optixPipelineSetStackSize(
				m_pipeline,
				dcStackSizeTraversal,
				dcStackSizeState,
				ccStackSize,
				maxTraversalDepth));
}
void RenderingInterface::createRenderResolveProgram()
{
	std::ifstream ifstr{ getExeDir() / "image.ptx", std::ios_base::binary | std::ios_base::ate };
	size_t inputSize{ static_cast<size_t>(ifstr.tellg()) };
	char* input{ new char[inputSize + 1] };
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	input[inputSize] = '\0';
	CUDA_CHECK(cuModuleLoadData(&m_imageModule, input));
	CUDA_CHECK(cuModuleGetFunction(&m_resolveRenderDataFunc, m_imageModule, m_renderResolveFunctionName.c_str()));
	delete[] input;
}
void RenderingInterface::createCDFBuildProgram()
{
	std::ifstream ifstr{ getExeDir() / "cdf.ptx", std::ios_base::binary | std::ios_base::ate };
	size_t inputSize{ static_cast<size_t>(ifstr.tellg()) };
	char* input{ new char[inputSize + 1] };
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	input[inputSize] = '\0';
	CUDA_CHECK(cuModuleLoadData(&m_CDFModule, input));
	R_ASSERT(m_buildCDFFunctions.size() == m_buildCDFFuncNames.size());
	for (int i{ 0 }; i < m_buildCDFFunctions.size(); ++i)
		CUDA_CHECK(cuModuleGetFunction(&m_buildCDFFunctions[i], m_CDFModule, m_buildCDFFuncNames[i]));
	CUDA_CHECK(cuModuleGetFunction(&m_invertCDFToIndicesFunction, m_CDFModule, m_invertCDFToIndicesFuncName));
	delete[] input;
}
void RenderingInterface::createConstantSBTRecords()
{
	//Fill raygen and miss records
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.raygenRecord), sizeof(OptixRecordRaygen)));
	OptixRecordRaygen raygenRecord{};
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::RAYGEN], &raygenRecord));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.raygenRecord), &raygenRecord, sizeof(OptixRecordRaygen), cudaMemcpyHostToDevice));

	m_sbt.missRecordCount = 1;
	m_sbt.missRecordStrideInBytes = sizeof(OptixRecordMiss);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.missRecordBase), m_sbt.missRecordStrideInBytes * m_sbt.missRecordCount));
	OptixRecordMiss missRecord{};
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::MISS], &missRecord));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.missRecordBase), &missRecord, m_sbt.missRecordStrideInBytes * m_sbt.missRecordCount, cudaMemcpyHostToDevice));

	//Fill callable records (bxdfs)
	constexpr int callableCount{ 3 };
	m_sbt.callablesRecordCount = callableCount;
	m_sbt.callablesRecordStrideInBytes = sizeof(OptixRecordCallable);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.callablesRecordBase), m_sbt.callablesRecordStrideInBytes * m_sbt.callablesRecordCount));
	OptixRecordCallable callableRecords[callableCount]{};
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::PURE_CONDUCTOR_BXDF], callableRecords + bxdfTypeToIndex(SceneData::BxDF::PURE_CONDUCTOR)));
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::PURE_DIELECTRIC_BXDF], callableRecords + bxdfTypeToIndex(SceneData::BxDF::PURE_DIELECTRIC)));
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::COMPLEX_SURFACE_BXDF], callableRecords + bxdfTypeToIndex(SceneData::BxDF::COMPLEX_SURFACE)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.callablesRecordBase), callableRecords, m_sbt.callablesRecordStrideInBytes * m_sbt.callablesRecordCount, cudaMemcpyHostToDevice));
}
void RenderingInterface::updateHitgroupSBTRecords(const SceneData& scene)
{
	uint32_t hitgroupCount{ 0 };
	//Fill hitgroup records for lights (light data)               | Trace stride, trace offset and instance offset affects these
	OptixRecordHitgroup lightHitgroupRecords[KSampleableLightCount]{};
	hitgroupCount += ARRAYSIZE(lightHitgroupRecords);
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::SPHERE], lightHitgroupRecords[0].header));
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::DISK], lightHitgroupRecords[1].header));

	//Fill hitgroup records for ordinary geometry (material data) | Trace stride, trace offset and instance offset affects these
	std::vector<OptixRecordHitgroup> matHitgroupRecords{};
	for (auto& model : scene.models)
	{
		const RenderingInterface::ModelResource& modelRes{ m_modelResources[model.id] };
		for (auto& instance : model.instances)
		{
			for (auto& submesh : model.meshes[instance.meshIndex].submeshes)
			{
				OptixRecordHitgroup& record{ matHitgroupRecords.emplace_back() };
				OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::TRIANGLE], record.header));
				record.data = modelRes.materialIndices[submesh.materialIndex];
			}
		}
	}
	hitgroupCount += matHitgroupRecords.size();

	OptixRecordHitgroup* hitgroups{ new OptixRecordHitgroup[hitgroupCount] };
	int j{ 0 };
	for (auto& hg : lightHitgroupRecords)
		hitgroups[j++] = hg;
	for (auto& hg : matHitgroupRecords)
		hitgroups[j++] = hg;
	m_sbt.hitgroupRecordCount = hitgroupCount;
	m_sbt.hitgroupRecordStrideInBytes = sizeof(OptixRecordHitgroup);
	if (m_sbt.hitgroupRecordBase != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase), m_sbt.hitgroupRecordStrideInBytes * m_sbt.hitgroupRecordCount));
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase), hitgroups, m_sbt.hitgroupRecordStrideInBytes * m_sbt.hitgroupRecordCount, cudaMemcpyHostToDevice, m_streams[1]));
}
void RenderingInterface::fillSpectralCurvesData()
{
	DenselySampledSpectrum sensorSpectralCurves[]{ SpectralData::CIE::X(), SpectralData::CIE::Y(), SpectralData::CIE::Z() };
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sensorSpectralCurvesData), sizeof(DenselySampledSpectrum) * ARRAYSIZE(sensorSpectralCurves)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sensorSpectralCurvesData), sensorSpectralCurves, sizeof(DenselySampledSpectrum) * ARRAYSIZE(sensorSpectralCurves), cudaMemcpyHostToDevice));

	DenselySampledSpectrum spectralBasis[]{ SpectralData::CIE::BasisR(), SpectralData::CIE::BasisG(), SpectralData::CIE::BasisB() };
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_spectralBasisData), sizeof(DenselySampledSpectrum) * ARRAYSIZE(spectralBasis)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_spectralBasisData), spectralBasis, sizeof(DenselySampledSpectrum) * ARRAYSIZE(spectralBasis), cudaMemcpyHostToDevice));
}
void RenderingInterface::loadLookUpTables()
{
	constexpr uint32_t lutWidth{ 32 };
	constexpr uint32_t lutHeight{ 32 };
	constexpr uint32_t lutDepth2D{ 1 };
	constexpr uint32_t lutDepth3D{ 32 };
	std::ifstream ifstr{};
	CudaCombinedTexture* lut{};
	size_t inputSize{};
	char* input{};

	lut = &m_lookUpTables[LookUpTable::CONDUCTOR_ALBEDO];
	*lut = CudaCombinedTexture{lutWidth, lutHeight, lutDepth2D, TextureType::R32_FLOAT,
			TextureAddress::CLAMP, TextureAddress::CLAMP, TextureAddress::CLAMP,
			TextureFilter::LINEAR, false};
	ifstr.open(getExeDir() / "bin" / "albedoConductorLUT.bin", std::ios_base::binary | std::ios_base::ate);
	inputSize = static_cast<size_t>(ifstr.tellg());
	input = new char[inputSize];
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	ifstr.close();
	lut->image.fill(input, 0, 0, 0, lutWidth, lutHeight, lutDepth2D, cudaMemcpyHostToDevice);
	delete[] input;
	m_launchParameters.LUTs.conductorAlbedo = m_lookUpTables[LookUpTable::CONDUCTOR_ALBEDO].texture.getTextureObject();

	lut = &m_lookUpTables[LookUpTable::DIELECTRIC_OUTER_ALBEDO];
	*lut = CudaCombinedTexture{lutWidth, lutHeight, lutDepth3D, TextureType::R32_FLOAT,
			TextureAddress::CLAMP, TextureAddress::CLAMP, TextureAddress::CLAMP,
			TextureFilter::LINEAR, false};
	ifstr.open(getExeDir() / "bin" / "albedoDielectricOuterLUT.bin", std::ios_base::binary | std::ios_base::ate);
	inputSize = static_cast<size_t>(ifstr.tellg());
	input = new char[inputSize];
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	ifstr.close();
	lut->image.fill(input, 0, 0, 0, lutWidth, lutHeight, lutDepth3D, cudaMemcpyHostToDevice);
	delete[] input;
	m_launchParameters.LUTs.dielectricOuterAlbedo = m_lookUpTables[LookUpTable::DIELECTRIC_OUTER_ALBEDO].texture.getTextureObject();

	lut = &m_lookUpTables[LookUpTable::DIELECTRIC_INNER_ALBEDO];
	*lut = CudaCombinedTexture{lutWidth, lutHeight, lutDepth3D, TextureType::R32_FLOAT,
			TextureAddress::CLAMP, TextureAddress::CLAMP, TextureAddress::CLAMP,
			TextureFilter::LINEAR, false};
	ifstr.open(getExeDir() / "bin" / "albedoDielectricInnerLUT.bin", std::ios_base::binary | std::ios_base::ate);
	inputSize = static_cast<size_t>(ifstr.tellg());
	input = new char[inputSize];
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	ifstr.close();
	lut->image.fill(input, 0, 0, 0, lutWidth, lutHeight, lutDepth3D, cudaMemcpyHostToDevice);
	delete[] input;
	m_launchParameters.LUTs.dielectricInnerAlbedo = m_lookUpTables[LookUpTable::DIELECTRIC_INNER_ALBEDO].texture.getTextureObject();

	lut = &m_lookUpTables[LookUpTable::REFLECTIVE_DIELECTRIC_OUTER_ALBEDO];
	*lut = CudaCombinedTexture{lutWidth, lutHeight, lutDepth3D, TextureType::R32_FLOAT,
			TextureAddress::CLAMP, TextureAddress::CLAMP, TextureAddress::CLAMP,
			TextureFilter::LINEAR, false};
	ifstr.open(getExeDir() / "bin" / "albedoDielectricReflectiveOuterLUT.bin", std::ios_base::binary | std::ios_base::ate);
	inputSize = static_cast<size_t>(ifstr.tellg());
	input = new char[inputSize];
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	ifstr.close();
	lut->image.fill(input, 0, 0, 0, lutWidth, lutHeight, lutDepth3D, cudaMemcpyHostToDevice);
	delete[] input;
	m_launchParameters.LUTs.reflectiveDielectricOuterAlbedo = m_lookUpTables[LookUpTable::REFLECTIVE_DIELECTRIC_OUTER_ALBEDO].texture.getTextureObject();

	lut = &m_lookUpTables[LookUpTable::REFLECTIVE_DIELECTRIC_INNER_ALBEDO];
	*lut = CudaCombinedTexture{lutWidth, lutHeight, lutDepth3D, TextureType::R32_FLOAT,
			TextureAddress::CLAMP, TextureAddress::CLAMP, TextureAddress::CLAMP,
			TextureFilter::LINEAR, false};
	ifstr.open(getExeDir() / "bin" / "albedoDielectricReflectiveInnerLUT.bin", std::ios_base::binary | std::ios_base::ate);
	inputSize = static_cast<size_t>(ifstr.tellg());
	input = new char[inputSize];
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	ifstr.close();
	lut->image.fill(input, 0, 0, 0, lutWidth, lutHeight, lutDepth3D, cudaMemcpyHostToDevice);
	delete[] input;
	m_launchParameters.LUTs.reflectiveDielectricInnerAlbedo = m_lookUpTables[LookUpTable::REFLECTIVE_DIELECTRIC_INNER_ALBEDO].texture.getTextureObject();

	lut = &m_lookUpTables[LookUpTable::SHEEN_LTC];
	*lut = CudaCombinedTexture{lutWidth, lutHeight, lutDepth2D, TextureType::R32G32B32A32_FLOAT,
			TextureAddress::CLAMP, TextureAddress::CLAMP, TextureAddress::CLAMP,
			TextureFilter::LINEAR, false};
	ifstr.open(getExeDir() / "bin" / "sheenLUT.bin", std::ios_base::binary | std::ios_base::ate);
	inputSize = static_cast<size_t>(ifstr.tellg());
	input = new char[inputSize];
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	ifstr.close();
	lut->image.fill(input, 0, 0, 0, lutWidth, lutHeight, lutDepth2D, cudaMemcpyHostToDevice);
	delete[] input;
	m_launchParameters.LUTs.sheenLTC = m_lookUpTables[LookUpTable::SHEEN_LTC].texture.getTextureObject();
}
void RenderingInterface::prepareDataForRendering(const Camera& camera, const RenderContext& renderContext)
{
	m_mode = renderContext.getRenderMode();
	m_sampleCount = renderContext.getSampleCount();
	m_launchWidth = renderContext.getRenderWidth();
	m_launchHeight = renderContext.getRenderHeight();
	m_imageExposure = renderContext.getImageExposure();
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_renderData),
				m_rDataComponentSize * m_rDataComponentCount * renderContext.getRenderWidth() * renderContext.getRenderHeight()));

	m_launchParameters.resolutionState = {
		.filmWidth = static_cast<uint32_t>(renderContext.getRenderWidth()),
		.filmHeight = static_cast<uint32_t>(renderContext.getRenderHeight()),
		.invFilmWidth = renderContext.getRenderInvWidth(),
		.invFilmHeight = renderContext.getRenderInvHeight(),
		.perspectiveScaleW = static_cast<float>(glm::tan(camera.getFieldOfView() * 0.5)) * (static_cast<float>(renderContext.getRenderWidth()) / static_cast<float>(renderContext.getRenderHeight())),
		.perspectiveScaleH = static_cast<float>(glm::tan(camera.getFieldOfView() * 0.5)) };
	int maxPathDepth{ std::max(1, renderContext.getMaxPathDepth()) };
	m_launchParameters.pathState.maxPathDepth = maxPathDepth;
	m_launchParameters.pathState.maxReflectedPathDepth = std::min(maxPathDepth, renderContext.getMaxReflectedPathDepth());
	m_launchParameters.pathState.maxTransmittedPathDepth = std::min(maxPathDepth, renderContext.getMaxTransmittedPathDepth());
	m_launchParameters.samplingState = {
		.offset = static_cast<uint32_t>(m_currentSampleOffset),
		.count = static_cast<uint32_t>(m_currentSampleCount) };
	m_launchParameters.renderData = m_renderData;
	m_launchParameters.cameraState = {
		.camU = camera.getU(),
		.camV = camera.getV(),
		.camW = camera.getW(),
		.depthOfFieldEnabled = camera.depthOfFieldEnabled(),
		.appertureSize = static_cast<float>(camera.getAperture()),
		.focusDistance = static_cast<float>(camera.getFocusDistance()),
	};
	m_launchParameters.materials = m_materialData;
	m_launchParameters.spectra = m_spectralData;
	m_launchParameters.sensorSpectralCurveA = m_sensorSpectralCurvesData + sizeof(DenselySampledSpectrum) * 0;
	m_launchParameters.sensorSpectralCurveB = m_sensorSpectralCurvesData + sizeof(DenselySampledSpectrum) * 1;
	m_launchParameters.sensorSpectralCurveC = m_sensorSpectralCurvesData + sizeof(DenselySampledSpectrum) * 2;
	m_launchParameters.spectralBasisR = m_spectralBasisData + sizeof(DenselySampledSpectrum) * 0;
	m_launchParameters.spectralBasisG = m_spectralBasisData + sizeof(DenselySampledSpectrum) * 1;
	m_launchParameters.spectralBasisB = m_spectralBasisData + sizeof(DenselySampledSpectrum) * 2;
	m_launchParameters.traversable = m_iasBuffer;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_lpBuffer), sizeof(LaunchParameters)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_lpBuffer), &m_launchParameters, sizeof(LaunchParameters), cudaMemcpyHostToDevice));
}
void RenderingInterface::prepareDataForPreviewDrawing()
{
	glGenTextures(1, &m_glTexture);
	glBindTexture(GL_TEXTURE_2D, m_glTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_launchWidth, m_launchHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_graphicsResource, m_glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	glBindTexture(GL_TEXTURE_2D, 0);

	// glGenVertexArrays(1, &m_VAO);
	//
	// const char* vShaderCode{
	// 	"#version 430 core\n"
	//
	// 	"out vec2 uv;\n"
	//
	// 	"void main()\n"
	// 	"{\n"
	// 		"uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
	// 		"gl_Position = vec4(uv * 2.0f - 1.0f, 0.0f, 1.0f);\n"
	// 	"}\0"
	// };
	// const char* fShaderCode{
	// 	"#version 430 core\n"
	// 	"uniform sampler2D tex;\n"
	// 	"uniform vec2 uvScale;\n"
	// 	"uniform vec2 uvOffset;\n"
	//
	// 	"in vec2 uv;\n"
	//
	// 	"out vec4 FragColor;\n"
	//
	// 	"void main()\n"
	// 	"{\n"
	// 		"vec2 fitUV = uv * uvScale + uvOffset;\n"
	// 		"vec3 color = texture(tex, fitUV).xyz;\n"
	// 		"if (fitUV.x < 0.0f || fitUV.x > 1.0f || fitUV.y < 0.0f || fitUV.y > 1.0f) color.xyz = vec3(0.0f);\n"
	// 		"FragColor = vec4(color, 1.0f);\n"
	// 	"}\0"
	// };
	//
	// uint32_t vertexShader{ glCreateShader(GL_VERTEX_SHADER) };
	// glShaderSource(vertexShader, 1, &vShaderCode, NULL);
	// glCompileShader(vertexShader);
	// checkGLShaderCompileErrors(vertexShader);
	//
	// uint32_t fragmentShader{ glCreateShader(GL_FRAGMENT_SHADER) };
	// glShaderSource(fragmentShader, 1, &fShaderCode, NULL);
	// glCompileShader(fragmentShader);
	// checkGLShaderCompileErrors(fragmentShader);
	//
	// m_drawProgram = glCreateProgram();
	// glAttachShader(m_drawProgram, vertexShader);
	// glAttachShader(m_drawProgram, fragmentShader);
	// glLinkProgram(m_drawProgram);
	// checkGLProgramLinkingErrors(m_drawProgram);
	//
	// glDeleteShader(vertexShader);
	// glDeleteShader(fragmentShader);
}

void RenderingInterface::buildGeometryAccelerationStructures(RenderingInterface::ModelResource& modelRes, SceneData::Model& model)
{
	CUdeviceptr tempBuffer{};

	for (int j{ 0 }; j < model.meshes.size(); ++j)
	{
		const SceneData::Mesh& mesh{ model.meshes[j] };
		OptixTraversableHandle gasHandle{};
		CUdeviceptr gasBuffer{};

		std::vector<OptixBuildInput> gasBuildInputs(mesh.submeshes.size());
		std::vector<CUdeviceptr> vertexBuffers(mesh.submeshes.size());
		std::vector<CUdeviceptr> indexBuffers(mesh.submeshes.size());

		for (int k{ 0 }; k < mesh.submeshes.size(); ++k)
		{
			const SceneData::Submesh& submesh{ mesh.submeshes[k] };
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(vertexBuffers[k])), sizeof(decltype(submesh.vertices)::value_type) * submesh.vertices.size()));
			CUDA_CHECK(cudaMemcpy(
						reinterpret_cast<void*>(vertexBuffers[k]), submesh.vertices.data(),
						sizeof(decltype(submesh.vertices)::value_type) * submesh.vertices.size(),
						cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(indexBuffers[k])), (submesh.indexType == IndexType::UINT_32 ? sizeof(uint32_t) * 3 : sizeof(uint16_t) * 3) * submesh.primitiveCount));
			CUDA_CHECK(cudaMemcpy(
						reinterpret_cast<void*>(indexBuffers[k]), submesh.indices,
						(submesh.indexType == IndexType::UINT_32 ? sizeof(uint32_t) * 3 : sizeof(uint16_t) * 3) * submesh.primitiveCount,
						cudaMemcpyHostToDevice));

			OptixIndicesFormat indexFormat{ submesh.indexType == IndexType::UINT_32 ? OPTIX_INDICES_FORMAT_UNSIGNED_INT3 : OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 };
			uint32_t indexStride{ static_cast<uint32_t>(submesh.indexType == IndexType::UINT_32 ? sizeof(uint32_t) * 3 : sizeof(uint16_t) * 3) };
			uint32_t flags{ OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };

			gasBuildInputs[k] = OptixBuildInput{.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
				.triangleArray = OptixBuildInputTriangleArray{
				.vertexBuffers = &(vertexBuffers[k]),
				.numVertices = static_cast<uint32_t>(submesh.vertices.size()),
				.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
				.vertexStrideInBytes = sizeof(decltype(submesh.vertices)::value_type),
				.indexBuffer = indexBuffers[k],
				.numIndexTriplets = submesh.primitiveCount,
				.indexFormat = indexFormat,
				.indexStrideInBytes = indexStride,
				.flags = &flags,
				.numSbtRecords = 1,
				.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE}};
		}

		OptixAccelBuildOptions accelBuildOptions{
			.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
				.operation = OPTIX_BUILD_OPERATION_BUILD };
		OptixAccelBufferSizes computedBufferSizes{};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accelBuildOptions, gasBuildInputs.data(), gasBuildInputs.size(), &computedBufferSizes));
		size_t compactedSizeOffset{ ALIGNED_SIZE(computedBufferSizes.tempSizeInBytes, 8ull) };
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), compactedSizeOffset + sizeof(size_t)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gasBuffer), computedBufferSizes.outputSizeInBytes));
		OptixAccelEmitDesc emittedProperty{};
		emittedProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emittedProperty.result = reinterpret_cast<CUdeviceptr>(reinterpret_cast<uint8_t*>(tempBuffer) + compactedSizeOffset);
		OPTIX_CHECK(optixAccelBuild(m_context, 0, &accelBuildOptions,
					gasBuildInputs.data(), gasBuildInputs.size(),
					tempBuffer, computedBufferSizes.tempSizeInBytes,
					gasBuffer, computedBufferSizes.outputSizeInBytes, &gasHandle, &emittedProperty, 1));
		size_t compactedGasSize{};
		CUDA_CHECK(cudaMemcpy(&compactedGasSize, reinterpret_cast<void*>(emittedProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));
		if (compactedGasSize < computedBufferSizes.outputSizeInBytes)
		{
			CUdeviceptr noncompactedGasBuffer{ gasBuffer };
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gasBuffer), compactedGasSize));
			OPTIX_CHECK(optixAccelCompact(m_context, 0, gasHandle, gasBuffer, compactedGasSize, &gasHandle));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(noncompactedGasBuffer)));
		}
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));

		for(auto buf : vertexBuffers)
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
		for(auto buf : indexBuffers)
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));

		modelRes.gasHandles.push_back(gasHandle);
		modelRes.gasBuffers.push_back(gasBuffer);
	}
}
void RenderingInterface::buildLightAccelerationStructure(const SceneData& scene, LightType type)
{
	CUdeviceptr aabbBuffer{};
	CUdeviceptr spherePosBuffer{};
	CUdeviceptr sphereRadiusBuffer{};
	CUdeviceptr tempBuffer{};


	if (type == LightType::DISK)
	{
		if (m_customPrimBuffer != CUdeviceptr{})
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_customPrimBuffer)));

		OptixBuildInput customPrimBuildInputs[1]{};
		constexpr uint32_t diskLightSBTRecordCount{ 1 };

		size_t aabbCount{ scene.diskLights.size() };
		std::vector<OptixAabb> aabbs(aabbCount);
		uint32_t diskGeometryFlags[diskLightSBTRecordCount]{};
		for (int i{ 0 }; i < diskLightSBTRecordCount; ++i)
			diskGeometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
		if (aabbCount != 0)
		{
			for (int i{ 0 }; i < aabbCount; ++i)
				aabbs[i] = scene.diskLights[i].getOptixAABB();
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&aabbBuffer), sizeof(OptixAabb) * aabbCount));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(aabbBuffer), aabbs.data(), sizeof(OptixAabb) * aabbCount, cudaMemcpyHostToDevice));
		}

		OptixAccelBuildOptions customPrimAccelBuildOptions{
			.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
				.operation = OPTIX_BUILD_OPERATION_BUILD };

		customPrimBuildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		customPrimBuildInputs[0].customPrimitiveArray.numSbtRecords = diskLightSBTRecordCount;
		customPrimBuildInputs[0].customPrimitiveArray.flags = diskGeometryFlags;
		if (aabbCount != 0)
		{
			customPrimBuildInputs[0].customPrimitiveArray.aabbBuffers = &aabbBuffer;
			customPrimBuildInputs[0].customPrimitiveArray.numPrimitives = static_cast<uint32_t>(aabbCount);
			customPrimBuildInputs[0].customPrimitiveArray.flags = diskGeometryFlags;
			customPrimBuildInputs[0].customPrimitiveArray.primitiveIndexOffset = 0;
		}

		OptixAccelBufferSizes computedBufferSizes{};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &customPrimAccelBuildOptions, customPrimBuildInputs, ARRAYSIZE(customPrimBuildInputs), &computedBufferSizes));
		size_t compactedSizeOffset{ ALIGNED_SIZE(computedBufferSizes.tempSizeInBytes, 8ull) };
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), compactedSizeOffset + sizeof(size_t)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_customPrimBuffer), computedBufferSizes.outputSizeInBytes));
		OptixAccelEmitDesc emittedProperty{};
		emittedProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emittedProperty.result = reinterpret_cast<CUdeviceptr>(reinterpret_cast<uint8_t*>(tempBuffer) + compactedSizeOffset);
		OPTIX_CHECK(optixAccelBuild(m_context, 0, &customPrimAccelBuildOptions,
					customPrimBuildInputs, ARRAYSIZE(customPrimBuildInputs),
					tempBuffer, computedBufferSizes.tempSizeInBytes,
					m_customPrimBuffer, computedBufferSizes.outputSizeInBytes, &m_customPrimHandle, &emittedProperty, 1));
		size_t compactedSize{};
		CUDA_CHECK(cudaMemcpy(&compactedSize, reinterpret_cast<void*>(emittedProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
		if (compactedSize < computedBufferSizes.outputSizeInBytes)
		{
			CUdeviceptr noncompactedBuffer{ m_customPrimBuffer };
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_customPrimBuffer), compactedSize));
			OPTIX_CHECK(optixAccelCompact(m_context, 0, m_customPrimHandle, m_customPrimBuffer, compactedSize, &m_customPrimHandle));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(noncompactedBuffer)));
		}

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(aabbBuffer)));
	}
	else if (type == LightType::SPHERE)
	{
		if (m_spherePrimBuffer != CUdeviceptr{})
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_spherePrimBuffer)));

		OptixBuildInput spherePrimitiveBuildInput[1]{};

		size_t sphereCount{ scene.sphereLights.size() };
		std::vector<glm::vec4> posBuffer(sphereCount);
		std::vector<float> radiusBuffer(sphereCount);
		constexpr uint32_t sphereLightSBTRecordCount{ 1 };
		uint32_t sphereGeometryFlags[sphereLightSBTRecordCount]{};
		for (int i{ 0 }; i < sphereLightSBTRecordCount; ++i)
			sphereGeometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
		if (sphereCount != 0)
		{
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&spherePosBuffer), sphereCount * sizeof(glm::vec4)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sphereRadiusBuffer), sphereCount * sizeof(float)));
			for (int i{ 0 }; i < sphereCount; ++i)
			{
				posBuffer[i] = glm::vec4{scene.sphereLights[i].getPosition(), 0.0f};
				radiusBuffer[i] = scene.sphereLights[i].getRadius();
			}
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(spherePosBuffer), posBuffer.data(), sphereCount * sizeof(glm::vec4), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sphereRadiusBuffer), radiusBuffer.data(), sphereCount * sizeof(float), cudaMemcpyHostToDevice));
		}

		OptixAccelBuildOptions spherePrimAccelBuildOptions{
			.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
				.operation = OPTIX_BUILD_OPERATION_BUILD };

		spherePrimitiveBuildInput[0].type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
		spherePrimitiveBuildInput[0].sphereArray.numSbtRecords = 1;
		spherePrimitiveBuildInput[0].sphereArray.flags = sphereGeometryFlags;
		if (sphereCount != 0)
		{
			spherePrimitiveBuildInput[0].sphereArray.vertexBuffers = &spherePosBuffer;
			spherePrimitiveBuildInput[0].sphereArray.vertexStrideInBytes = sizeof(glm::vec4);
			spherePrimitiveBuildInput[0].sphereArray.numVertices = static_cast<uint32_t>(sphereCount);
			spherePrimitiveBuildInput[0].sphereArray.radiusBuffers = &sphereRadiusBuffer;
			spherePrimitiveBuildInput[0].sphereArray.radiusStrideInBytes = sizeof(float);
			spherePrimitiveBuildInput[0].sphereArray.primitiveIndexOffset = 0;
		}

		OptixAccelBufferSizes computedBufferSizes{};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &spherePrimAccelBuildOptions, spherePrimitiveBuildInput, ARRAYSIZE(spherePrimitiveBuildInput), &computedBufferSizes));
		size_t compactedSizeOffset{ ALIGNED_SIZE(computedBufferSizes.tempSizeInBytes, 8ull) };
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), compactedSizeOffset + sizeof(size_t)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_spherePrimBuffer), computedBufferSizes.outputSizeInBytes));
		OptixAccelEmitDesc emittedProperty{};
		emittedProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emittedProperty.result = reinterpret_cast<CUdeviceptr>(reinterpret_cast<uint8_t*>(tempBuffer) + compactedSizeOffset);
		OPTIX_CHECK(optixAccelBuild(m_context, 0, &spherePrimAccelBuildOptions,
					spherePrimitiveBuildInput, ARRAYSIZE(spherePrimitiveBuildInput),
					tempBuffer, computedBufferSizes.tempSizeInBytes,
					m_spherePrimBuffer, computedBufferSizes.outputSizeInBytes, &m_spherePrimitiveHandle, &emittedProperty, 1));
		size_t compactedSize{};
		CUDA_CHECK(cudaMemcpy(&compactedSize, reinterpret_cast<void*>(emittedProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
		if (compactedSize < computedBufferSizes.outputSizeInBytes)
		{
			CUdeviceptr noncompactedBuffer{ m_spherePrimBuffer };
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_spherePrimBuffer), compactedSize));
			OPTIX_CHECK(optixAccelCompact(m_context, 0, m_spherePrimitiveHandle, m_spherePrimBuffer, compactedSize, &m_spherePrimitiveHandle));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(noncompactedBuffer)));
		}

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(spherePosBuffer)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sphereRadiusBuffer)));
	}
}
void RenderingInterface::updateInstanceAccelerationStructure(const SceneData& scene, const Camera& camera)
{
	if (m_iasBuffer != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_iasBuffer)));

	const glm::vec3& cameraPosition{ camera.getPosition() };

	CUdeviceptr tempBuffer{};

	uint32_t instanceCount{ 2 }; // Non-triangle primitives instances
	for(auto& model : scene.models)
		instanceCount += model.instances.size();

	OptixInstance* instances{};
	CUdeviceptr instanceBuffer{};
	CUDA_CHECK(cudaHostAlloc(&instances, sizeof(OptixInstance) * instanceCount, cudaHostAllocMapped));
	CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&instanceBuffer), instances, 0));
	instances[0].instanceId = 0;
	instances[0].sbtOffset = 0;
	instances[0].traversableHandle = m_spherePrimitiveHandle;
	instances[0].visibilityMask = 0xFF;
	instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
	instances[0].transform[0]  = 1.0f;
	instances[0].transform[1]  = 0.0f;
	instances[0].transform[2]  = 0.0f;
	instances[0].transform[3]  = -cameraPosition.x;
	instances[0].transform[4]  = 0.0f;
	instances[0].transform[5]  = 1.0f;
	instances[0].transform[6]  = 0.0f;
	instances[0].transform[7]  = -cameraPosition.y;
	instances[0].transform[8]  = 0.0f;
	instances[0].transform[9]  = 0.0f;
	instances[0].transform[10] = 1.0f;
	instances[0].transform[11] = -cameraPosition.z;
	instances[1].instanceId = 1;
	instances[1].sbtOffset = m_spherePrimitiveSBTRecordCount;
	instances[1].traversableHandle = m_customPrimHandle;
	instances[1].visibilityMask = 0xFF;
	instances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
	instances[1].transform[0]  = 1.0f;
	instances[1].transform[1]  = 0.0f;
	instances[1].transform[2]  = 0.0f;
	instances[1].transform[3]  = -cameraPosition.x;
	instances[1].transform[4]  = 0.0f;
	instances[1].transform[5]  = 1.0f;
	instances[1].transform[6]  = 0.0f;
	instances[1].transform[7]  = -cameraPosition.y;
	instances[1].transform[8]  = 0.0f;
	instances[1].transform[9]  = 0.0f;
	instances[1].transform[10] = 1.0f;
	instances[1].transform[11] = -cameraPosition.z;
	int inst{ 2 };
	int sbtOffset{ m_spherePrimitiveSBTRecordCount + m_customPrimitiveSBTRecordCount };
	for(auto& model : scene.models)
	{
		const RenderingInterface::ModelResource& modelRes{ m_modelResources[model.id] };
		glm::mat4 modelTransform{ model.transform };
		modelTransform[3][3] = 1.0f;
		for(auto& instance : model.instances)
		{
			glm::mat4 transform{ instance.transform };
			transform[3][3] = 1.0f;
			transform = modelTransform * transform;

			instances[inst].instanceId = inst;
			instances[inst].sbtOffset = sbtOffset;
			instances[inst].traversableHandle = modelRes.gasHandles[instance.meshIndex];
			instances[inst].visibilityMask = 0xFF;
			instances[inst].flags = OPTIX_INSTANCE_FLAG_NONE;
			instances[inst].transform[0]  = transform[0][0];
			instances[inst].transform[1]  = transform[1][0];
			instances[inst].transform[2]  = transform[2][0];
			instances[inst].transform[3]  = -cameraPosition.x + transform[3][0];
			instances[inst].transform[4]  = transform[0][1];
			instances[inst].transform[5]  = transform[1][1];
			instances[inst].transform[6]  = transform[2][1];
			instances[inst].transform[7]  = -cameraPosition.y + transform[3][1];
			instances[inst].transform[8]  = transform[0][2];
			instances[inst].transform[9]  = transform[1][2];
			instances[inst].transform[10] = transform[2][2];
			instances[inst].transform[11] = -cameraPosition.z + transform[3][2];
			sbtOffset += model.meshes[instance.meshIndex].submeshes.size();
			++inst;
		}
	}
	OptixBuildInput iasBuildInputs[]{
		OptixBuildInput{ .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
			.instanceArray = OptixBuildInputInstanceArray{ .instances = instanceBuffer, .numInstances = instanceCount } } };

	OptixAccelBuildOptions accelBuildOptions{
		.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
		.operation = OPTIX_BUILD_OPERATION_BUILD };
	OptixAccelBufferSizes computedBufferSizes{};
	OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accelBuildOptions, iasBuildInputs, ARRAYSIZE(iasBuildInputs), &computedBufferSizes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), computedBufferSizes.tempSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_iasBuffer), computedBufferSizes.outputSizeInBytes));
	OPTIX_CHECK(optixAccelBuild(m_context, 0, &accelBuildOptions,
				iasBuildInputs, ARRAYSIZE(iasBuildInputs), 
				tempBuffer, computedBufferSizes.tempSizeInBytes, m_iasBuffer,
				computedBufferSizes.outputSizeInBytes, &m_iasHandle, nullptr, 0));

	CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(instances)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));

	m_launchParameters.traversable = m_iasHandle;
}
uint32_t RenderingInterface::addMaterial(const MaterialData& matData)
{
	if (m_freeMaterialsIndices.empty())
	{
		constexpr int addMatAllocNum{ 3 };
		int newMatDataCount{ m_matDataCount + addMatAllocNum };
		CUdeviceptr newMatDataAlloc{};
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&newMatDataAlloc), sizeof(MaterialData) * newMatDataCount));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(newMatDataAlloc), reinterpret_cast<void*>(m_materialData), sizeof(MaterialData) * m_matDataCount, cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_materialData)));
		for (int i{ m_matDataCount }; i < newMatDataCount; ++i)
			m_freeMaterialsIndices.push(i);
		m_matDataCount = newMatDataCount;
		m_materialData = newMatDataAlloc;
		m_launchParameters.materials = m_materialData;
	}
	uint32_t index{ m_freeMaterialsIndices.top() }; m_freeMaterialsIndices.pop();
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<MaterialData*>(m_materialData) + index, &matData, sizeof(MaterialData), cudaMemcpyHostToDevice));
	return index;
}
int RenderingInterface::changeSpectrum(SpectralData::SpectralDataType newSpecType, SpectralData::SpectralDataType oldSpecType)
{
	std::function remSpecRef{ [this](const SpectralData::SpectralDataType sdt)
		{
			if (sdt != SpectralData::SpectralDataType::NONE)
			{
				if (m_loadedSpectra.contains(sdt))
				{
					SpectrumRecord& rec{ m_loadedSpectra.at(sdt) };
					R_ASSERT_LOG(rec.refcount > 0, "Spectrum refcount is zero or negative.");
					if (--rec.refcount == 0)
					{
						m_freeSpectra.push(rec.index);
						m_loadedSpectra.erase(sdt);
					}
				}
				else
					R_ERR_LOG("Attempted to remove a reference to a not loaded spectrum.")
			}
		} };
	int spectrumIndex{};
	if (oldSpecType != newSpecType)
	{
		remSpecRef(oldSpecType);
		spectrumIndex = setNewSpectrum(newSpecType);
	}
	else
	{
		spectrumIndex = getSpectrum(oldSpecType);
	}

	return spectrumIndex;
}
uint32_t RenderingInterface::setNewSpectrum(SpectralData::SpectralDataType type)
{
	if (type != SpectralData::SpectralDataType::NONE)
	{
		int index{};
		if (m_loadedSpectra.contains(type))
		{
			SpectrumRecord& rec{ m_loadedSpectra.at(type) };
			rec.refcount += 1;
			index = rec.index;
		}
		else
		{
			if (m_freeSpectra.empty())
			{
				index = static_cast<int>(m_loadedSpectra.size());
				int oldCount{ index };
				m_loadedSpectra.insert({type, SpectrumRecord{index, 1}});
				CUdeviceptr oldSpectraBuffer{ m_spectralData };
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_spectralData), sizeof(DenselySampledSpectrum) * m_loadedSpectra.size()));
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_spectralData), reinterpret_cast<void*>(oldSpectraBuffer), sizeof(DenselySampledSpectrum) * oldCount, cudaMemcpyDeviceToDevice));
				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(oldSpectraBuffer)));
				DenselySampledSpectrum spec{ SpectralData::loadSpectrum(type) };
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<DenselySampledSpectrum*>(m_spectralData) + index, &spec, sizeof(DenselySampledSpectrum), cudaMemcpyHostToDevice));
				m_launchParameters.spectra = m_spectralData;
			}
			else
			{
				index = m_freeSpectra.top();
				m_freeSpectra.pop();
				m_loadedSpectra.insert({type, {index, 1}});
				DenselySampledSpectrum spec{ SpectralData::loadSpectrum(type) };
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<DenselySampledSpectrum*>(m_spectralData) + index,
							&spec,
							sizeof(DenselySampledSpectrum),
							cudaMemcpyHostToDevice));
			}
		}
		return index;
	}
	return 0;
}
uint32_t RenderingInterface::getSpectrum(SpectralData::SpectralDataType type)
{
	if (type != SpectralData::SpectralDataType::NONE)
	{
		R_ASSERT_LOG(m_loadedSpectra.contains(type), "Spectral data not loaded");
		return m_loadedSpectra.at(type).index;
	}
	return 0;
}
int RenderingInterface::bxdfTypeToIndex(SceneData::BxDF type)
{
	switch (type)
	{
		case SceneData::BxDF::PURE_CONDUCTOR:
			return 0;
			break;
		case SceneData::BxDF::PURE_DIELECTRIC:
			return 1;
			break;
		case SceneData::BxDF::COMPLEX_SURFACE:
			return 2;
			break;
		default:
			R_ERR_LOG("Unknown BxDF type.")
			break;
	}
	return -1;
}

void RenderingInterface::loadModel(SceneData::Model& model)
{
	RenderingInterface::ModelResource& modelData{ m_modelResources[model.id] };
	buildGeometryAccelerationStructures(modelData, model);
	fillModelMaterials(modelData, model);
}
void RenderingInterface::loadLights(SceneData& scene, const Camera& camera)
{
	for (int i{ 0 }; i < scene.sphereLights.size(); ++i)
		m_lightResources[scene.sphereLights[i].getID()].materialIndex = fillLightMaterial(scene.sphereLights[i].getMaterialDescriptor());
	for (int i{ 0 }; i < scene.diskLights.size(); ++i)
		m_lightResources[scene.diskLights[i].getID()].materialIndex = fillLightMaterial(scene.diskLights[i].getMaterialDescriptor());
	uploadLightData(scene, camera.getPosition(), true);
	buildLightAccelerationStructure(scene, LightType::SPHERE);
	buildLightAccelerationStructure(scene, LightType::DISK);
}
void RenderingInterface::removeModel(uint32_t modelID)
{
	auto& res{ m_modelResources[modelID] };
	for(auto ind : res.materialIndices)
		m_freeMaterialsIndices.push(ind);
	for (auto buf : res.gasBuffers)
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
	for (auto buf : res.indexBuffers)
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
	for (auto buf : res.attributeBuffers)
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
	m_modelResources.erase(modelID);
}
void RenderingInterface::removeLight(uint32_t lightID)
{
	m_freeMaterialsIndices.push(m_lightResources[lightID].materialIndex);
	m_lightResources.erase(lightID);
}
void RenderingInterface::loadEnvironmentMap(const char* path)
{
	LaunchParameters::EnvironmentMap envMap{};

	// Free previous environment map data if needed
	if (m_launchParameters.envMap.conditionalCDFIndices != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.envMap.conditionalCDFIndices)));
	if (m_launchParameters.envMap.marginalCDFIndices != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.envMap.marginalCDFIndices)));
	// Load HDRI map into memory
	int w{};
	int h{};
	int n{};
	float* data{ stbi_loadf(path, &w, &h, &n, 4) };
	R_ASSERT_LOG(data != nullptr, "Loading environment image failed");
	envMap.width = static_cast<float>(w);
	envMap.height = static_cast<float>(h);
	R_ASSERT_LOG((w % 4 == 0) && (h % 4 == 0), "Image dimensions are not divisible by 4");
	m_envMap = CudaCombinedTexture{ static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1, TextureType::R32G32B32A32_FLOAT,
		TextureAddress::CLAMP, TextureAddress::CLAMP, TextureAddress::CLAMP,
		TextureFilter::LINEAR, false /*HDRI is already in linear colorspace*/ };
	envMap.environmentTexture = m_envMap.texture.getTextureObject();
	m_envMap.image.fill(data, 0, 0, 0, w, h, 1, cudaMemcpyHostToDevice);
	stbi_image_free(data);
	// Compute CDFs and integrals
	CUdeviceptr cdfConditional{};
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cdfConditional), w * h * sizeof(float)));
	CUdeviceptr cdfMarginal{};
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cdfMarginal), h * sizeof(float)));
	CUdeviceptr integral{};
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&integral), sizeof(float)));
	constexpr int numAtomics{ 1 };
	CUdeviceptr counter{};
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&counter), numAtomics * sizeof(int)));
	CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(counter), 0, numAtomics * sizeof(int)));
	cudaResourceDesc resDesc{};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = m_envMap.image.getData();
	cudaSurfaceObject_t surface{ 0 };
	CUDA_CHECK(cudaCreateSurfaceObject(&surface, &resDesc));
	void* buildCdfParams[]{ &w, &h, &surface, &cdfConditional, &cdfMarginal, &integral, &counter };
	int numElements{ DISPATCH_SIZE(w, 4) };
	const int kBlockSize{ (numElements <= 256 ? 256 :
			(numElements <= 512 ? 512 : 1024)) };
	dim3 numBlocks{ 1, static_cast<unsigned int>(h) };
	dim3 blockSize{ static_cast<unsigned int>(kBlockSize), 1 };
	int blocksPerRaw{ DISPATCH_SIZE(numElements, kBlockSize) };
	int buildCDFFuncIndex{};
	if (blocksPerRaw == 1)
		buildCDFFuncIndex = 0;
	else if (blocksPerRaw == 2)
		buildCDFFuncIndex = 1;
	else if (blocksPerRaw <= 4)
		buildCDFFuncIndex = 2;
	else if (blocksPerRaw <= 8)
		buildCDFFuncIndex = 3;
	else
		R_ERR_LOG("Light map resolution exceeds limit of 32k.");
	CUDA_SYNC_DEVICE();
	CUDA_CHECK(cuLaunchKernel(m_buildCDFFunctions[buildCDFFuncIndex],
				numBlocks.x, numBlocks.y, numBlocks.z,
				blockSize.x, blockSize.y, blockSize.z,
				kBlockSize * sizeof(float),
				m_streams[0],
				buildCdfParams, nullptr));
	CUDA_CHECK(cudaMemcpyAsync(&envMap.integral, reinterpret_cast<void*>(integral), sizeof(float), cudaMemcpyDeviceToHost, m_streams[0]));
	envMap.integral /= static_cast<float>(w) * static_cast<float>(h);
	// Invert CDF to indices
	CUdeviceptr conditionalCDFIndices{};
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&conditionalCDFIndices), w * h * sizeof(uint16_t)));
	envMap.conditionalCDFIndices = conditionalCDFIndices;
	CUdeviceptr marginalCDFIndices{};
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&marginalCDFIndices), h * sizeof(uint16_t)));
	envMap.marginalCDFIndices = marginalCDFIndices;
	void* invertCDFParams[]{ &w, &h, &cdfConditional, &cdfMarginal, &conditionalCDFIndices, &marginalCDFIndices };
	CUDA_CHECK(cuLaunchKernel(m_invertCDFToIndicesFunction,
				h + 1, DISPATCH_SIZE(std::max(w, h), 1024), 1,
				1024, 1, 1,
				0,
				m_streams[0],
				invertCDFParams, nullptr));
	// Free temporary resources
	CUDA_SYNC_DEVICE();
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(counter)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(cdfMarginal)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(cdfConditional)));
	CUDA_CHECK(cudaDestroySurfaceObject(surface));
	// Copy image data to image texture
	// Free temporary resources
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(integral)));

	envMap.enabled = true;
	m_launchParameters.envMap = envMap;
}

void RenderingInterface::resolveRender(const glm::mat3& colorspaceTransform)
{
	if (m_imageCudaSurface != cudaSurfaceObject_t{})
		CUDA_CHECK(cudaDestroySurfaceObject(m_imageCudaSurface));

	CUDA_CHECK(cudaGraphicsMapResources(1, &m_graphicsResource, m_streams[0]));
	CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&m_imageCudaArray, m_graphicsResource, 0, 0));
	cudaResourceDesc resDesc{ .resType = cudaResourceTypeArray, .res = { m_imageCudaArray } };
	CUDA_CHECK(cudaCreateSurfaceObject(&m_imageCudaSurface, &resDesc));

	glm::mat3 colspTransform{ colorspaceTransform };
	void* params[]{ &m_launchWidth, &m_launchHeight, &colspTransform, &m_imageExposure, &m_renderData, &m_imageCudaSurface };
	CUDA_CHECK(cuLaunchKernel(m_resolveRenderDataFunc,
				DISPATCH_SIZE(m_launchWidth, 16), DISPATCH_SIZE(m_launchHeight, 16), 1, 
				16, 16, 1,
				0, 
				m_streams[0], 
				params, nullptr));

	CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_graphicsResource, m_streams[0]));
}
void RenderingInterface::processCommands(CommandBuffer& commands, RenderContext& renderContext, Camera& camera, SceneData& scene)
{
	bool restartRender{ false };
	bool reresolveRender{ false };
	bool rebuildSphereLightsAS{ false };
	bool rebuildDiskLightsAS{ false };
	bool rebuildIAS{ false };
	bool updateSBTRecords{ false };
	while (!commands.empty())
	{
		Command cmd{ commands.pullCommand() };
		CommandType type{ cmd.type };
		const void* payload{ cmd.payload };

		switch (type)
		{
			case CommandType::CHANGE_RENDER_MODE:
				{
					m_mode = renderContext.getRenderMode();
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_SAMPLE_COUNT:
				{
					m_sampleCount = renderContext.getSampleCount();
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_PATH_DEPTH:
				{
					int maxPathDepth{ std::max(1, renderContext.getMaxPathDepth()) };
					m_launchParameters.pathState.maxPathDepth = maxPathDepth;
					m_launchParameters.pathState.maxReflectedPathDepth = std::min(maxPathDepth, renderContext.getMaxReflectedPathDepth());
					m_launchParameters.pathState.maxTransmittedPathDepth = std::min(maxPathDepth, renderContext.getMaxTransmittedPathDepth());
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_IMAGE_EXPOSURE:
				{
					m_imageExposure = renderContext.getImageExposure();
					reresolveRender = true;
					break;
				}
			case CommandType::CHANGE_DEPTH_OF_FIELD_SETTINGS:
				{
					m_launchParameters.cameraState.depthOfFieldEnabled = camera.depthOfFieldEnabled();
					m_launchParameters.cameraState.appertureSize = camera.getAperture();
					m_launchParameters.cameraState.focusDistance = camera.getFocusDistance();
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_RENDER_RESOLUTION:
				{
					m_launchWidth = renderContext.getRenderWidth();
					m_launchHeight = renderContext.getRenderHeight();
					LaunchParameters::ResolutionState newResolutionState{
							.filmWidth = static_cast<uint32_t>(renderContext.getRenderWidth()),
							.filmHeight = static_cast<uint32_t>(renderContext.getRenderHeight()),
							.invFilmWidth = renderContext.getRenderInvWidth(),
							.invFilmHeight = renderContext.getRenderInvHeight(),
							.perspectiveScaleW = static_cast<float>(glm::tan(camera.getFieldOfView() * 0.5)) * (static_cast<float>(renderContext.getRenderWidth()) / static_cast<float>(renderContext.getRenderHeight())),
							.perspectiveScaleH = static_cast<float>(glm::tan(camera.getFieldOfView() * 0.5)) };

					m_launchParameters.resolutionState = newResolutionState;

					CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_renderData)));
					CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_renderData),
								m_rDataComponentSize * m_rDataComponentCount * renderContext.getRenderWidth() * renderContext.getRenderHeight()));

					m_launchParameters.renderData = m_renderData;

					CUDA_CHECK(cudaGraphicsUnregisterResource(m_graphicsResource));
					glBindTexture(GL_TEXTURE_2D, m_glTexture);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_launchWidth, m_launchHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
					CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_graphicsResource, m_glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
					glBindTexture(GL_TEXTURE_2D, 0);

					restartRender = true;
					break;
				}
			case CommandType::CHANGE_CAMERA_POS:
				{
					uploadLightData(scene, camera.getPosition(), false);
					rebuildIAS = true;
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_CAMERA_ORIENTATION:
				{
					m_launchParameters.cameraState.camU = camera.getU();
					m_launchParameters.cameraState.camV = camera.getV();
					m_launchParameters.cameraState.camW = camera.getW();
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_CAMERA_FIELD_OF_VIEW:
				{
					LaunchParameters::ResolutionState newResolutionState{
							.filmWidth = static_cast<uint32_t>(renderContext.getRenderWidth()),
							.filmHeight = static_cast<uint32_t>(renderContext.getRenderHeight()),
							.invFilmWidth = renderContext.getRenderInvWidth(),
							.invFilmHeight = renderContext.getRenderInvHeight(),
							.perspectiveScaleW = static_cast<float>(glm::tan(camera.getFieldOfView() * 0.5)) * (static_cast<float>(renderContext.getRenderWidth()) / static_cast<float>(renderContext.getRenderHeight())),
							.perspectiveScaleH = static_cast<float>(glm::tan(camera.getFieldOfView() * 0.5)) };

					m_launchParameters.resolutionState = newResolutionState;

					restartRender = true;
					break;
				}
			case CommandType::ADD_ENVIRONMENT_MAP:
				{
					const CommandPayloads::EnvironmentMap* envMapPayload{ reinterpret_cast<const CommandPayloads::EnvironmentMap*>(payload) };
					loadEnvironmentMap(envMapPayload->path.c_str());
					restartRender = true;
					break;
				}
			case CommandType::REMOVE_ENVIRONMENT_MAP:
				{
					if (m_launchParameters.envMap.conditionalCDFIndices != CUdeviceptr{})
						CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.envMap.conditionalCDFIndices)));
					m_launchParameters.envMap.conditionalCDFIndices = CUdeviceptr{};
					if (m_launchParameters.envMap.marginalCDFIndices != CUdeviceptr{})
						CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.envMap.marginalCDFIndices)));
					m_launchParameters.envMap.marginalCDFIndices = CUdeviceptr{};
					m_envMap = CudaCombinedTexture{};
					m_launchParameters.envMap.enabled = false;
					restartRender = true;
					break;
				}
			case CommandType::ADD_LIGHT:
				{
					const CommandPayloads::Light* lightPayload{ reinterpret_cast<const CommandPayloads::Light*>(payload) };
					uint32_t index{ lightPayload->index };
					if (lightPayload->type == LightType::SPHERE)
					{
						m_lightResources[scene.sphereLights[index].getID()].materialIndex = fillLightMaterial(scene.sphereLights[index].getMaterialDescriptor());
						rebuildSphereLightsAS = true;
					}
					else if (lightPayload->type == LightType::DISK)
					{
						m_lightResources[scene.diskLights[index].getID()].materialIndex = fillLightMaterial(scene.diskLights[index].getMaterialDescriptor());
						rebuildDiskLightsAS = true;
					}
					uploadLightData(scene, camera.getPosition(), true);
					rebuildIAS = true;
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_LIGHT_POSITION:
				{
					const CommandPayloads::Light* lightPayload{ reinterpret_cast<const CommandPayloads::Light*>(payload) };
					if (lightPayload->type == LightType::SPHERE)
					{
						const SceneData::SphereLight& sl{ scene.sphereLights[lightPayload->index] };
						glm::vec3 newPos{ sl.getPosition() - glm::vec3{camera.getPosition()} };
						CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(m_sphereLights) + sizeof(SphereLightData) * lightPayload->index + offsetof(SphereLightData, position),
									&newPos,
									sizeof(newPos), cudaMemcpyHostToDevice));
						rebuildSphereLightsAS = true;
					}
					else if (lightPayload->type == LightType::DISK)
					{
						const SceneData::DiskLight& dl{ scene.diskLights[lightPayload->index] };
						glm::vec3 newPos{ dl.getPosition() - glm::vec3{camera.getPosition()} };
						CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(m_diskLights) + sizeof(DiskLightData) * lightPayload->index + offsetof(DiskLightData, position),
									&newPos,
									sizeof(newPos), cudaMemcpyHostToDevice));
						rebuildDiskLightsAS = true;
					}
					rebuildIAS = true;
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_LIGHT_SIZE:
				{
					const CommandPayloads::Light* lightPayload{ reinterpret_cast<const CommandPayloads::Light*>(payload) };

					if (lightPayload->type == LightType::SPHERE)
					{
						const SceneData::SphereLight& sl{ scene.sphereLights[lightPayload->index] };
						float newRadius{ sl.getRadius() };
						CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(m_sphereLights) + sizeof(SphereLightData) * lightPayload->index + offsetof(SphereLightData, radius),
									&newRadius,
									sizeof(newRadius), cudaMemcpyHostToDevice));
						rebuildSphereLightsAS = true;
					}
					else if (lightPayload->type == LightType::DISK)
					{
						const SceneData::DiskLight& dl{ scene.diskLights[lightPayload->index] };
						float newRadius{ dl.getRadius() };
						CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(m_diskLights) + sizeof(DiskLightData) * lightPayload->index + offsetof(DiskLightData, radius),
									&newRadius,
									sizeof(newRadius), cudaMemcpyHostToDevice));
						rebuildDiskLightsAS = true;
					}

					rebuildIAS = true;
					restartRender = true;
					break;
					break;
				}
			case CommandType::CHANGE_LIGHT_ORIENTATION:
				{
					const CommandPayloads::Light* lightPayload{ reinterpret_cast<const CommandPayloads::Light*>(payload) };
					if (lightPayload->type == LightType::SPHERE)
					{
						R_ERR_LOG("Changing sphere orientation. Isn't supposed to happen.'");
					}
					else if (lightPayload->type == LightType::DISK)
					{
						rebuildDiskLightsAS = true;
					}

					const SceneData::DiskLight& dl{ scene.diskLights[lightPayload->index] };
					auto newFrame{ dl.getFrame() };
					CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(m_diskLights) + sizeof(DiskLightData) * lightPayload->index + offsetof(DiskLightData, frame),
								&newFrame,
								sizeof(newFrame), cudaMemcpyHostToDevice));

					rebuildIAS = true;
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_LIGHT_POWER:
				{
					const CommandPayloads::Light* lightPayload{ reinterpret_cast<const CommandPayloads::Light*>(payload) };

					if (lightPayload->type == LightType::SPHERE)
					{
						const SceneData::SphereLight& sl{ scene.sphereLights[lightPayload->index] };
						float newPowScale{ sl.getPowerScale() };
						CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(m_sphereLights) + sizeof(SphereLightData) * lightPayload->index + offsetof(SphereLightData, powerScale),
									&newPowScale,
									sizeof(newPowScale), cudaMemcpyHostToDevice));
					}
					else if (lightPayload->type == LightType::DISK)
					{
						const SceneData::DiskLight& dl{ scene.diskLights[lightPayload->index] };
						float newPowScale{ dl.getPowerScale() };
						CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(m_diskLights) + sizeof(DiskLightData) * lightPayload->index + offsetof(DiskLightData, powerScale),
									&newPowScale,
									sizeof(newPowScale), cudaMemcpyHostToDevice));
					}

					restartRender = true;
					break;
				}
			case CommandType::CHANGE_LIGHT_EMISSION_SPECTRUM:
				{
					const CommandPayloads::Light* lightPayload{ reinterpret_cast<const CommandPayloads::Light*>(payload) };
					uint32_t matIndex{};
					SpectralData::SpectralDataType newType{};
					if (lightPayload->type == LightType::SPHERE)
					{
						uint32_t index{ lightPayload->index };
						matIndex = m_lightResources[scene.sphereLights[index].getID()].materialIndex;
						newType = scene.sphereLights[index].getMaterialDescriptor().baseEmission;
					}
					else if (lightPayload->type == LightType::DISK)
					{
						uint32_t index{ lightPayload->index };
						matIndex = m_lightResources[scene.diskLights[index].getID()].materialIndex;
						newType = scene.diskLights[index].getMaterialDescriptor().baseEmission;
					}
					uint16_t newSpectrumIndex{ static_cast<uint16_t>(changeSpectrum(newType, lightPayload->oldEmissionType)) };
					CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(m_materialData) + sizeof(MaterialData) * matIndex + offsetof(MaterialData, emissionSpectrumDataIndex),
								&newSpectrumIndex,
								sizeof(newSpectrumIndex), cudaMemcpyHostToDevice));
					restartRender = true;
					break;
				}
			case CommandType::REMOVE_LIGHT:
				{
					const CommandPayloads::Light* lightPayload{ reinterpret_cast<const CommandPayloads::Light*>(payload) };
					removeLight(lightPayload->id);
					if (lightPayload->type == LightType::SPHERE)
					{
						rebuildSphereLightsAS = true;
					}
					else if (lightPayload->type == LightType::DISK)
					{
						rebuildDiskLightsAS = true;
					}
					uploadLightData(scene, camera.getPosition(), true);
					rebuildIAS = true;
					restartRender = true;
					break;
				}
			case CommandType::ADD_MODEL:
				{
					const CommandPayloads::Model* modelPayload{ reinterpret_cast<const CommandPayloads::Model*>(payload) };
					loadModel(scene.models[modelPayload->index]);
					rebuildIAS = true;
					updateSBTRecords = true;
					restartRender = true;

					//
					uint32_t lightCount{ 0 };
					for(auto& model : scene.models)
						for(auto& subset : model.instancedEmissiveMeshSubsets)
							lightCount += subset.triangles.size();
					LightTree::Builder::SortData* sortData{ new LightTree::Builder::SortData[lightCount] };
					uint32_t sdIndex{ 0 };
					for (int i{ 0 }; i < scene.models.size(); ++i)
					{
						auto& model{ scene.models[i] };
						for (int j{ 0 }; j < model.instancedEmissiveMeshSubsets.size(); ++j)
						{
							auto& subset{ model.instancedEmissiveMeshSubsets[j] };
							for (int k{ 0 }; k < subset.triangles.size(); ++k)
							{
								auto& triangle{ subset.triangles[k] };
								glm::vec3 normal{ glm::normalize(glm::cross(triangle.v1WS - triangle.v0WS, triangle.v2WS - triangle.v0WS)) };
								sortData[sdIndex].bounds.min[0] = std::min(triangle.v0WS.x, std::min(triangle.v1WS.x, triangle.v2WS.x));
								sortData[sdIndex].bounds.min[1] = std::min(triangle.v0WS.y, std::min(triangle.v1WS.y, triangle.v2WS.y));
								sortData[sdIndex].bounds.min[2] = std::min(triangle.v0WS.z, std::min(triangle.v1WS.z, triangle.v2WS.z));
								sortData[sdIndex].bounds.max[0] = std::max(triangle.v0WS.x, std::max(triangle.v1WS.x, triangle.v2WS.x));
								sortData[sdIndex].bounds.max[1] = std::max(triangle.v0WS.y, std::max(triangle.v1WS.y, triangle.v2WS.y));
								sortData[sdIndex].bounds.max[2] = std::max(triangle.v0WS.z, std::max(triangle.v1WS.z, triangle.v2WS.z));
								for (int i{ 0 }; i < 3; ++i)
								{
									if (sortData[sdIndex].bounds.max[i] <= sortData[sdIndex].bounds.min[i])
									{
										sortData[sdIndex].bounds.min[i] = sortData[sdIndex].bounds.max[i] - std::numeric_limits<float>::epsilon();
										sortData[sdIndex].bounds.max[i] = sortData[sdIndex].bounds.max[i] + std::numeric_limits<float>::epsilon();
									}
								}
								sortData[sdIndex].coneDirection[0] = normal.x;
								sortData[sdIndex].coneDirection[1] = normal.y;
								sortData[sdIndex].coneDirection[2] = normal.z;
								sortData[sdIndex].cosConeAngle = 1.0f;
								sortData[sdIndex].flux = triangle.flux * subset.transformFluxCorrection;
								sortData[sdIndex].lightDataRef.triangleRef.modelIndex = i;
								sortData[sdIndex].lightDataRef.triangleRef.subsetIndex = j;
								sortData[sdIndex].lightDataRef.triangleRef.triangleIndex = k;
								++sdIndex;
							}
						}
					}
					LightTree::Builder builder{};
					LightTree::Tree tree{ builder.build(scene, sortData, lightCount) };
					delete[] sortData;
					//

					break;
				}
			case CommandType::CHANGE_MODEL_MATERIAL:
				{
					const CommandPayloads::Model* modelPayload{ reinterpret_cast<const CommandPayloads::Model*>(payload) };
					// for(auto ind : m_modelResources[modelPayload->id].materialIndices)
					// {
					// 	m_materialData;
					// }
					restartRender = true;
					break;
				}
			case CommandType::CHANGE_MODEL_TRANSFORM:
				{
					const CommandPayloads::Model* modelPayload{ reinterpret_cast<const CommandPayloads::Model*>(payload) };
					rebuildIAS = true;
					restartRender = true;
					break;
				}
			case CommandType::REMOVE_MODEL:
				{
					const CommandPayloads::Model* modelPayload{ reinterpret_cast<const CommandPayloads::Model*>(payload) };
					removeModel(modelPayload->id);
					rebuildIAS = true;
					updateSBTRecords = true;
					restartRender = true;
					break;
				}
			default:
				{
					R_ERR_LOG("Unknown command sent.");
					break;
				}
		}
	}

	if (rebuildSphereLightsAS)
	{
		buildLightAccelerationStructure(scene, LightType::SPHERE);
	}
	if (rebuildDiskLightsAS)
	{
		buildLightAccelerationStructure(scene, LightType::DISK);
	}
	if (rebuildIAS)
	{
		updateInstanceAccelerationStructure(scene, camera);
	}
	if (updateSBTRecords)
	{
		updateHitgroupSBTRecords(scene);
	}
	if (restartRender)
	{
		CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_lpBuffer), &m_launchParameters, sizeof(m_launchParameters), cudaMemcpyHostToDevice, m_streams[1]));

		m_currentSampleCount = 1;
		m_currentSampleOffset = 0;
		m_processedSampleCount = 0;

		m_renderingIsFinished = false;
	}
	else if (reresolveRender)
	{
		resolveRender(renderContext.getColorspaceTransform());
		CUDA_SYNC_STREAM(m_streams[0]);
	}
}
void RenderingInterface::updateSubLaunchData()
{
	LaunchParameters::SamplingState currentSamplingState{
		.offset = static_cast<uint32_t>(m_currentSampleOffset),
		.count = static_cast<uint32_t>(m_currentSampleCount) };

	CUDA_CHECK(cudaMemcpyAsync(
				reinterpret_cast<void*>(m_lpBuffer + offsetof(LaunchParameters, samplingState)),
				reinterpret_cast<void*>(&currentSamplingState),
				sizeof(currentSamplingState),
				cudaMemcpyHostToDevice,
				m_streams[1]));
}
void RenderingInterface::updateSamplingState()
{
	m_processedSampleCount = m_currentSampleOffset;
	m_currentSampleOffset += m_currentSampleCount;

	int sublaunchSize{};
	constexpr int maxSublaunchSize{ 64 };
	if (m_mode == RenderContext::Mode::GRADUAL)
		sublaunchSize = std::min(static_cast<int>(std::pow(std::max(3, m_currentSampleCount), 1.5)), maxSublaunchSize);
	else
		sublaunchSize = 1;

	m_currentSampleCount = std::min(std::max(0, m_sampleCount - static_cast<int>(m_currentSampleOffset)), sublaunchSize);
}
void RenderingInterface::launch()
{
	OPTIX_CHECK(optixLaunch(m_pipeline, m_streams[1], m_lpBuffer, sizeof(LaunchParameters), &m_sbt, m_launchWidth, m_launchHeight, 1));
	CUDA_CHECK(cudaEventRecord(m_execEvent, m_streams[1]));
	if (m_mode == RenderContext::Mode::GRADUAL)
		CUDA_SYNC_DEVICE();
}
void RenderingInterface::cleanup()
{
	OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[RAYGEN]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[MISS]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[TRIANGLE]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[DISK]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[PURE_CONDUCTOR_BXDF]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[PURE_DIELECTRIC_BXDF]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[COMPLEX_SURFACE_BXDF]));
	OPTIX_CHECK(optixModuleDestroy(m_ptModule));
	OPTIX_CHECK(optixDeviceContextDestroy(m_context));

	glDeleteTextures(1, &m_glTexture);
	glDeleteVertexArrays(1, &m_VAO);
	glDeleteProgram(m_drawProgram);
	CUDA_CHECK(cudaGraphicsUnregisterResource(m_graphicsResource));
	CUDA_CHECK(cuModuleUnload(m_imageModule));
	CUDA_CHECK(cuModuleUnload(m_CDFModule));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.callablesRecordBase)));
	for (auto& mdr : m_modelResources)
	{
		for(auto buf : mdr.second.gasBuffers)
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
		for(auto buf : mdr.second.indexBuffers)
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
		for(auto buf : mdr.second.attributeBuffers)
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
	}
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_customPrimBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_spherePrimBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_iasBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sensorSpectralCurvesData)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_spectralBasisData)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_lpBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_renderData)));

	if (m_launchParameters.envMap.environmentTexture != cudaTextureObject_t{})
		CUDA_CHECK(cudaDestroyTextureObject(m_launchParameters.envMap.environmentTexture));
	if (m_launchParameters.envMap.marginalCDFIndices != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.envMap.marginalCDFIndices)));
	if (m_launchParameters.envMap.conditionalCDFIndices != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.envMap.conditionalCDFIndices)));

	if (m_launchParameters.lights.orderedCount != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.lights.orderedCount)));
	if (m_materialData != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_materialData)));
	if (m_spectralData != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_spectralData)));
	if (m_diskLights != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_diskLights)));
	if (m_sphereLights != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sphereLights)));
}

RenderingInterface::RenderingInterface(const Camera& camera, const RenderContext& renderContext, SceneData& scene)
{
	int lowP;
	int highP;
	CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowP, &highP));
	CUDA_CHECK(cudaStreamCreateWithPriority(m_streams + 0, cudaStreamNonBlocking, highP));
	CUDA_CHECK(cudaStreamCreateWithPriority(m_streams + 1, cudaStreamNonBlocking, lowP));

	createOptixContext();
	createRenderResolveProgram();
	createCDFBuildProgram();
	createModulesProgramGroupsPipeline();
	createConstantSBTRecords();
	fillSpectralCurvesData();
	loadLookUpTables();

	loadLights(scene, camera);
	for (auto& model : scene.models)
		loadModel(model);
	updateHitgroupSBTRecords(scene);
	updateInstanceAccelerationStructure(scene, camera);

	prepareDataForRendering(camera, renderContext);
	prepareDataForPreviewDrawing();

	CUDA_CHECK(cudaEventCreateWithFlags(&m_execEvent, cudaEventDisableTiming));
}

void RenderingInterface::render(CommandBuffer& commands, RenderContext& renderContext, Camera& camera, SceneData& scene)
{
	if (commands.empty())
	{
		if (cudaEventQuery(m_execEvent) != cudaSuccess)
			return;
	}
	else
	{
		CUDA_SYNC_STREAM(m_streams[1]);
	}

	static bool first{ true };
	if (first) [[unlikely]]
		first = false;
	else
	{
		resolveRender(renderContext.getColorspaceTransform());
		CUDA_SYNC_STREAM(m_streams[0]);
	}

	if (!commands.empty())
		processCommands(commands, renderContext, camera, scene);

	if (m_currentSampleCount == 0)
	{
		m_processedSampleCount = m_currentSampleOffset;
		m_renderingIsFinished = true;
		return;
	}
	updateSubLaunchData();
	updateSamplingState();
	launch();
}
void RenderingInterface::drawPreview(int winWidth, int winHeight) const
{
	glClearColor(0.4f, 1.0f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_glTexture);
	glBindVertexArray(m_VAO);
	glUseProgram(m_drawProgram);

	// Identify render image and window size relationship and scale and offset appropriately
	static GLint uvScaleLoc{ glGetUniformLocation(m_drawProgram, "uvScale") };
	static GLint uvOffsetLoc{ glGetUniformLocation(m_drawProgram, "uvOffset") };
	bool relResCheck{ (static_cast<float>(m_launchWidth) / static_cast<float>(m_launchHeight)) * (static_cast<float>(winHeight) / static_cast<float>(winWidth)) > 1.0f };
	if (relResCheck)
	{
		float aspect{ static_cast<float>(m_launchWidth) / static_cast<float>(m_launchHeight) };
		aspect *= static_cast<float>(winHeight) / static_cast<float>(winWidth);
		float scale{ aspect };
		float offset{ 0.5f * (1.0f - aspect) };
		glUniform2f(uvScaleLoc, 1.0f, scale);
		glUniform2f(uvOffsetLoc, 0.0f, offset);
	}
	else
	{
		float aspect{ static_cast<float>(m_launchHeight) / static_cast<float>(m_launchWidth) };
		aspect *= static_cast<float>(winWidth) / static_cast<float>(winHeight);
		float scale{ aspect };
		float offset{ 0.5f * (1.0f - aspect) };
		glUniform2f(uvScaleLoc, scale, 1.0f);
		glUniform2f(uvOffsetLoc, offset, 0.0f);
	}

	glDrawArrays(GL_TRIANGLES, 0, 3);
}
