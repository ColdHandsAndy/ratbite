#pragma once

#include "rendering_interface.h"

#include <fstream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <functional>
#define NOMINMAX
#include <windows.h>

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

#include "scene.h"
#include "camera.h"
#include "render_context.h"
#include "launch_parameters.h"
#include "util_macros.h"
#include "debug_macros.h"
#include "callbacks.h"
#include "../kernels/optix_programs_desc.h"

void RenderingInterface::createOptixContext()
{
	OptixDeviceContextOptions options{ .logCallbackFunction = optixLogCallback, .logCallbackLevel = 4 };

#ifdef _DEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

	OPTIX_CHECK(optixDeviceContextCreate(0, &options, &m_context));
}
void RenderingInterface::createAccelerationStructures(const SceneData& scene, const glm::vec3& cameraPosition)
{
	CUdeviceptr vertexBuffer{};
	CUdeviceptr sbtOffsetBuffer{};
	CUdeviceptr aabbBuffer{};
	CUdeviceptr worldToWorldCameraTransformBuffer{};
	CUdeviceptr instanceBuffer{};
	CUdeviceptr tempBuffer{};

	float preTransform[12]{
		1.0f, 0.0f, 0.0f, -cameraPosition.x,
			0.0f, 1.0f, 0.0f, -cameraPosition.y,
			0.0f, 0.0f, 1.0f, -cameraPosition.z, };

	const uint32_t instanceCount{ 2 };

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&vertexBuffer), sizeof(scene.vertices[0]) * scene.vertices.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbtOffsetBuffer), sizeof(scene.SBTIndices[0]) * scene.SBTIndices.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&worldToWorldCameraTransformBuffer), sizeof(preTransform)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(vertexBuffer), scene.vertices.data(), sizeof(scene.vertices[0]) * scene.vertices.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbtOffsetBuffer), scene.SBTIndices.data(), sizeof(scene.SBTIndices[0]) * scene.SBTIndices.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(worldToWorldCameraTransformBuffer), preTransform, sizeof(preTransform), cudaMemcpyHostToDevice));

	{
		OptixAccelBuildOptions accelBuildOptions{
			.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
				.operation = OPTIX_BUILD_OPERATION_BUILD };
		uint32_t geometryFlags[scene.trianglePrimSBTCount]{};
		for (int i{ 0 }; i < ARRAYSIZE(geometryFlags); ++i)
			geometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
		OptixBuildInput gasBuildInputs[]{
			OptixBuildInput{.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
				.triangleArray = OptixBuildInputTriangleArray{.vertexBuffers = &vertexBuffer,
					.numVertices = static_cast<unsigned int>(scene.vertices.size()),
					.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
					.vertexStrideInBytes = sizeof(float) * 4,
					.preTransform = worldToWorldCameraTransformBuffer,
					.flags = geometryFlags,
					.numSbtRecords = scene.trianglePrimSBTCount,
					.sbtIndexOffsetBuffer = sbtOffsetBuffer,
					.sbtIndexOffsetSizeInBytes = sizeof(scene.SBTIndices[0]),
					.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 } } };
		OptixAccelBufferSizes computedBufferSizes{};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accelBuildOptions, gasBuildInputs, ARRAYSIZE(gasBuildInputs), &computedBufferSizes));
		size_t compactedSizeOffset{ ALIGNED_SIZE(computedBufferSizes.tempSizeInBytes, 8ull) };
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), compactedSizeOffset + sizeof(size_t)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_gasBuffer), computedBufferSizes.outputSizeInBytes));
		OptixAccelEmitDesc emittedProperty{};
		emittedProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emittedProperty.result = reinterpret_cast<CUdeviceptr>(reinterpret_cast<uint8_t*>(tempBuffer) + compactedSizeOffset);
		OPTIX_CHECK(optixAccelBuild(m_context, 0, &accelBuildOptions,
					gasBuildInputs, ARRAYSIZE(gasBuildInputs),
					tempBuffer, computedBufferSizes.tempSizeInBytes,
					m_gasBuffer, computedBufferSizes.outputSizeInBytes, &m_gasHandle, &emittedProperty, 1));
		size_t compactedGasSize{};
		CUDA_CHECK(cudaMemcpy(&compactedGasSize, reinterpret_cast<void*>(emittedProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
		if (compactedGasSize < computedBufferSizes.outputSizeInBytes)
		{
			CUdeviceptr noncompactedGasBuffer{ m_gasBuffer };
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_gasBuffer), compactedGasSize));
			OPTIX_CHECK(optixAccelCompact(m_context, 0, m_gasHandle, m_gasBuffer, compactedGasSize, &m_gasHandle));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(noncompactedGasBuffer)));
		}
	}

	{
		OptixAabb aabbs[]{ scene.diskLight.getOptixAABB(-cameraPosition) };
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&aabbBuffer), sizeof(OptixAabb) * scene.lightCount));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(aabbBuffer), aabbs, sizeof(OptixAabb) * scene.lightCount, cudaMemcpyHostToDevice));
		OptixAccelBuildOptions customPrimAccelBuildOptions{
			.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
				.operation = OPTIX_BUILD_OPERATION_BUILD };
		uint32_t geometryFlags[scene.lightCount]{};
		for (int i{ 0 }; i < ARRAYSIZE(geometryFlags); ++i)
			geometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
		OptixBuildInput customPrimBuildInputs[]{
			OptixBuildInput{.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
				.customPrimitiveArray = OptixBuildInputCustomPrimitiveArray{.aabbBuffers = &aabbBuffer,
					.numPrimitives = scene.lightCount,
					.flags = geometryFlags,
					.numSbtRecords = scene.lightCount,
					.primitiveIndexOffset = 0}} };
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
	}

	OptixInstance* instances{};
	CUDA_CHECK(cudaHostAlloc(&instances, sizeof(OptixInstance) * instanceCount, cudaHostAllocMapped));
	CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&instanceBuffer), instances, 0));
	instances[0].instanceId = 0;
	instances[0].sbtOffset = 0;
	instances[0].traversableHandle = m_gasHandle;
	instances[0].visibilityMask = 0xFF;
	instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
	instances[0].transform[0]  = 1.0f;
	instances[0].transform[1]  = 0.0f;
	instances[0].transform[2]  = 0.0f;
	instances[0].transform[3]  = 0.0f;
	instances[0].transform[4]  = 0.0f;
	instances[0].transform[5]  = 1.0f;
	instances[0].transform[6]  = 0.0f;
	instances[0].transform[7]  = 0.0f;
	instances[0].transform[8]  = 0.0f;
	instances[0].transform[9]  = 0.0f;
	instances[0].transform[10] = 1.0f;
	instances[0].transform[11] = 0.0f;
	instances[1].instanceId = 1;
	instances[1].sbtOffset = scene.trianglePrimSBTCount;
	instances[1].traversableHandle = m_customPrimHandle;
	instances[1].visibilityMask = 0xFF;
	instances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
	instances[1].transform[0]  = 1.0f;
	instances[1].transform[1]  = 0.0f;
	instances[1].transform[2]  = 0.0f;
	instances[1].transform[3]  = 0.0f;
	instances[1].transform[4]  = 0.0f;
	instances[1].transform[5]  = 1.0f;
	instances[1].transform[6]  = 0.0f;
	instances[1].transform[7]  = 0.0f;
	instances[1].transform[8]  = 0.0f;
	instances[1].transform[9]  = 0.0f;
	instances[1].transform[10] = 1.0f;
	instances[1].transform[11] = 0.0f;
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
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(vertexBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbtOffsetBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(aabbBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(worldToWorldCameraTransformBuffer)));
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
			.usesPrimitiveTypeFlags = static_cast<uint32_t>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM) };

	TCHAR buffer[MAX_PATH]{};
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::filesystem::path progpath{ buffer };
	std::ifstream ifstr{ progpath.remove_filename() / "pathtrace.ptx", std::ios_base::binary | std::ios_base::ate };
	size_t inputSize{ static_cast<size_t>(ifstr.tellg()) };
	char* input{ new char[inputSize] };
	ifstr.seekg(0);
	ifstr.read(input, inputSize);

	OptixModule& optixModule{ m_ptModule };
	OPTIX_CHECK_LOG(optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, input, inputSize, OPTIX_LOG, &OPTIX_LOG_SIZE, &optixModule));
	delete[] input;


	OptixProgramGroupOptions programGroupOptions{ .payloadType = payloadTypes + 0 };
	{
		OptixProgramGroupDesc descs[m_ptProgramGroupCount]{};
		descs[RenderingInterface::RAYGEN] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN, .raygen = {.module = optixModule, .entryFunctionName = Program::raygenName} };
		descs[RenderingInterface::MISS] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_MISS, .raygen = {.module = optixModule, .entryFunctionName = Program::missName} };
		descs[RenderingInterface::TRIANGLE] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP, .hitgroup = {.moduleCH = optixModule, .entryFunctionNameCH = Program::closehitTriangleName} };
		descs[RenderingInterface::DISK] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP, .hitgroup = {.moduleCH = optixModule, .entryFunctionNameCH = Program::closehitDiskName, .moduleIS = optixModule, .entryFunctionNameIS = Program::intersectionDiskName} };
		descs[RenderingInterface::CALLABLE] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES, .callables = {.moduleDC = optixModule, .entryFunctionNameDC = Program::callableName} };
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
	TCHAR charbuffer[MAX_PATH]{};
	GetModuleFileName(NULL, charbuffer, MAX_PATH);
	std::filesystem::path progpath{ charbuffer };
	std::ifstream ifstr{ progpath.remove_filename() / "image.ptx", std::ios_base::binary | std::ios_base::ate };
	size_t inputSize{ static_cast<size_t>(ifstr.tellg()) };
	char* input{ new char[inputSize + 1] };
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	input[inputSize] = '\0';
	CUDA_CHECK(cuModuleLoadData(&m_imageModule, input));
	CUDA_CHECK(cuModuleGetFunction(&m_resolveRenderDataFunc, m_imageModule, m_renderResolveFunctionName.c_str()));
	delete[] input;
}
void RenderingInterface::fillMaterials(const SceneData& scene)
{
	std::unordered_map<SpectralData::SpectralDataType, int> loadMats{};
	std::vector<MaterialData> matData(scene.materialDescriptors.size());
	std::function addMat{ [](const SpectralData::SpectralDataType sdt, uint16_t& spectrumIndex, std::unordered_map<SpectralData::SpectralDataType, int>& loadMats)
		{
			if (sdt != SpectralData::SpectralDataType::DESC)
			{
				if (loadMats.contains(sdt))
				{
					spectrumIndex = loadMats.at(sdt);
				}
				else
				{
					spectrumIndex = loadMats.size();
					loadMats.insert({ sdt, spectrumIndex });
				}
			}
		} };
	for (int i{ 0 }; i < scene.materialDescriptors.size(); ++i)
	{
		addMat(scene.materialDescriptors[i].baseIOR, matData[i].indexOfRefractSpectrumDataIndex, loadMats);
		addMat(scene.materialDescriptors[i].baseAC, matData[i].absorpCoefSpectrumDataIndex, loadMats);
		addMat(scene.materialDescriptors[i].baseEmission, matData[i].emissionSpectrumDataIndex, loadMats);
		matData[i].bxdfIndexSBT = scene.materialDescriptors[i].bxdfIndex;
		matData[i].mfRoughnessValue = scene.materialDescriptors[i].roughness;
	}
	//
	lightEmissionSpectrumIndex = matData[3].emissionSpectrumDataIndex;
	//
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_materialData), sizeof(MaterialData) * matData.size()));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_materialData), matData.data(), sizeof(MaterialData) * matData.size(), cudaMemcpyHostToDevice));
	DenselySampledSpectrum* spectrums{ new DenselySampledSpectrum[loadMats.size()]};
	for (const auto& [specType, index] : loadMats)
		spectrums[index] = SpectralData::loadSpectrum(specType);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_spectralData), sizeof(DenselySampledSpectrum) * loadMats.size()));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_spectralData), spectrums, sizeof(DenselySampledSpectrum) * loadMats.size(), cudaMemcpyHostToDevice));
	delete[] spectrums;
}
void RenderingInterface::createSBT(const SceneData& scene)
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
	m_sbt.callablesRecordCount = 1;
	m_sbt.callablesRecordStrideInBytes = sizeof(OptixRecordCallable);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.callablesRecordBase), m_sbt.callablesRecordStrideInBytes * m_sbt.callablesRecordCount));
	OptixRecordCallable callableRecord{};
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::CALLABLE], &callableRecord));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.callablesRecordBase), &callableRecord, m_sbt.callablesRecordStrideInBytes * m_sbt.callablesRecordCount, cudaMemcpyHostToDevice));

	uint32_t hitgroupCount{ 0 };
	//Fill hitgroup records for ordinary geometry (material data) | Trace stride, trace offset and instance offset affects these
	OptixRecordHitgroup matHitgroupRecords[scene.trianglePrimSBTCount]{};
	hitgroupCount += ARRAYSIZE(matHitgroupRecords);
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::TRIANGLE], matHitgroupRecords[0].header));
	for (int i{ 1 }; i < ARRAYSIZE(matHitgroupRecords); ++i)
		memcpy(matHitgroupRecords[i].header, matHitgroupRecords[0].header, sizeof(matHitgroupRecords[0].header));
	for (int i{ 0 }; i < ARRAYSIZE(matHitgroupRecords); ++i)
		matHitgroupRecords[i].data = static_cast<uint32_t>(i);

	//Fill hitgroup records for lights (light data)               | Trace stride, trace offset and instance offset affects these
	OptixRecordHitgroup lightHitgroupRecords[scene.lightCount]{};
	hitgroupCount += ARRAYSIZE(lightHitgroupRecords);
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::DISK], lightHitgroupRecords[0].header));
	lightHitgroupRecords[0].data = scene.lightMaterialIndex;

	OptixRecordHitgroup* hitgroups{ new OptixRecordHitgroup[hitgroupCount] };
	int i{ 0 };
	for (auto& hg : matHitgroupRecords)
		hitgroups[i++] = hg;
	for (auto& hg : lightHitgroupRecords)
		hitgroups[i++] = hg;
	m_sbt.hitgroupRecordCount = hitgroupCount;
	m_sbt.hitgroupRecordStrideInBytes = sizeof(OptixRecordHitgroup);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase), m_sbt.hitgroupRecordStrideInBytes * m_sbt.hitgroupRecordCount));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase), hitgroups, m_sbt.hitgroupRecordStrideInBytes * m_sbt.hitgroupRecordCount, cudaMemcpyHostToDevice));
	delete[] hitgroups;
}
void RenderingInterface::fillSpectralCurvesData()
{
	DenselySampledSpectrum sensorSpectralCurves[]{ SpectralData::CIE::X(), SpectralData::CIE::Y(), SpectralData::CIE::Z() };
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sensorSpectralCurvesData), sizeof(DenselySampledSpectrum) * ARRAYSIZE(sensorSpectralCurves)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sensorSpectralCurvesData), sensorSpectralCurves, sizeof(DenselySampledSpectrum) * ARRAYSIZE(sensorSpectralCurves), cudaMemcpyHostToDevice));
}
void RenderingInterface::prepareDataForRendering(const Camera& camera, const RenderContext& renderContext, const SceneData& scene)
{
	m_sampleCount = renderContext.getSampleCount();
	m_launchWidth = renderContext.getRenderWidth();
	m_launchHeight = renderContext.getRenderHeight();
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_renderData),
				m_rDataComponentSize * m_rDataComponentCount * renderContext.getRenderWidth() * renderContext.getRenderHeight()));
	LaunchParameters launchParameters{
			.filmWidth = static_cast<uint32_t>(renderContext.getRenderWidth()),
			.filmHeight = static_cast<uint32_t>(renderContext.getRenderHeight()),
			.invFilmWidth = renderContext.getRenderInvWidth(),
			.invFilmHeight = renderContext.getRenderInvHeight(),
			.maxPathDepth = static_cast<uint32_t>(renderContext.getPathLength()),
			.samplingState = { .offset = static_cast<uint32_t>(m_currentSampleOffset), .count = static_cast<uint32_t>(m_currentSampleCount) },
			.renderingData = m_renderData,
			.camU = camera.getU(),
			.camV = camera.getV(),
			.camW = camera.getW(),
			.camPerspectiveScaleW = static_cast<float>(glm::tan((glm::radians(45.0) * 0.5))) * (static_cast<float>(renderContext.getRenderWidth()) / static_cast<float>(renderContext.getRenderHeight())),
			.camPerspectiveScaleH = static_cast<float>(glm::tan((glm::radians(45.0) * 0.5))),
			.illuminantSpectralDistributionIndex = lightEmissionSpectrumIndex, //Change
			.diskLightPosition = scene.diskLight.pos - camera.getPosition(), //Change
			.diskLightRadius = scene.diskLight.radius, //Change
			.diskFrame = scene.diskLight.frame, //Change
			.diskNormal = scene.diskLight.normal, //Change
			.diskArea = scene.diskLight.area, //Change
			.lightScale = scene.diskLight.scale, //Change
			.diskSurfacePDF = 1.0f / scene.diskLight.area, //Change
			.materials = m_materialData,
			.spectrums = m_spectralData,
			.sensorSpectralCurveA = m_sensorSpectralCurvesData + sizeof(DenselySampledSpectrum) * 0,
			.sensorSpectralCurveB = m_sensorSpectralCurvesData + sizeof(DenselySampledSpectrum) * 1,
			.sensorSpectralCurveC = m_sensorSpectralCurvesData + sizeof(DenselySampledSpectrum) * 2,
			.traversable = m_iasBuffer };

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_lpBuffer), sizeof(LaunchParameters)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_lpBuffer), &launchParameters, sizeof(LaunchParameters), cudaMemcpyHostToDevice));
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

	glGenVertexArrays(1, &m_VAO);

	const char* vShaderCode{
		"#version 430 core\n"

		"out vec2 uv;\n"

		"void main()\n"
		"{\n"
			"uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
			"gl_Position = vec4(uv * 2.0f - 1.0f, 0.0f, 1.0f);\n"
		"}\0"
	};
	const char* fShaderCode{
		"#version 430 core\n"
		"uniform sampler2D tex;\n"

		"in vec2 uv;\n"

		"out vec4 FragColor;\n"

		"void main()\n"
		"{\n"
			"vec3 color = texture(tex, uv).xyz;\n"
			"FragColor = vec4(color, 1.0f);\n"
		"}\0"
	};

	uint32_t vertexShader{ glCreateShader(GL_VERTEX_SHADER) };
	glShaderSource(vertexShader, 1, &vShaderCode, NULL);
	glCompileShader(vertexShader);
	checkGLShaderCompileErrors(vertexShader);

	uint32_t fragmentShader{ glCreateShader(GL_FRAGMENT_SHADER) };
	glShaderSource(fragmentShader, 1, &fShaderCode, NULL);
	glCompileShader(fragmentShader);
	checkGLShaderCompileErrors(fragmentShader);

	m_drawProgram = glCreateProgram();
	glAttachShader(m_drawProgram, vertexShader);
	glAttachShader(m_drawProgram, fragmentShader);
	glLinkProgram(m_drawProgram);
	checkGLProgramLinkingErrors(m_drawProgram);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

void RenderingInterface::resolveRender(const glm::mat3& colorspaceTransform)
{
	CUDA_CHECK(cudaGraphicsMapResources(1, &m_graphicsResource));
	CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&m_imageCudaArray, m_graphicsResource, 0, 0));
	cudaResourceDesc resDesc{ .resType = cudaResourceTypeArray, .res = { m_imageCudaArray } };
	CUDA_CHECK(cudaCreateSurfaceObject(&m_imageCudaSurface, &resDesc));

	glm::mat3 colspTransform{ colorspaceTransform };
	void* params[]{ &m_launchWidth, &m_launchHeight, &colspTransform, &m_renderData, &m_imageCudaSurface };
	CUDA_CHECK(cuLaunchKernel(m_resolveRenderDataFunc,
				DISPATCH_SIZE(m_launchWidth, 16), DISPATCH_SIZE(m_launchHeight, 16), 1, 
				16, 16, 1,
				0, 
				0, 
				params, nullptr));

	CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_graphicsResource));
}
void RenderingInterface::updateSubLaunchData()
{
	LaunchParameters::SamplingState currentSamplingState{
		.offset = static_cast<uint32_t>(m_currentSampleOffset),
		.count = static_cast<uint32_t>(m_currentSampleCount) };

	CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(m_lpBuffer + offsetof(LaunchParameters, samplingState)),
				reinterpret_cast<void*>(&currentSamplingState),
				sizeof(currentSamplingState),
				cudaMemcpyHostToDevice));
}
void RenderingInterface::updateSamplingState()
{
	m_currentSampleOffset += m_currentSampleCount;
	m_currentSampleCount = std::min(std::max(0, m_sampleCount - static_cast<int>(m_currentSampleOffset)), 8);
}
void RenderingInterface::launch()
{
	OPTIX_CHECK(optixLaunch(m_pipeline, 0, m_lpBuffer, sizeof(LaunchParameters), &m_sbt, m_launchWidth, m_launchHeight, 1));
}

RenderingInterface::RenderingInterface(const Camera& camera, const RenderContext& renderContext, const SceneData& scene)
{
	createOptixContext();
	createRenderResolveProgram();
	createAccelerationStructures(scene, camera.getPosition());
	createModulesProgramGroupsPipeline();
	fillMaterials(scene);
	createSBT(scene);
	fillSpectralCurvesData();
	prepareDataForRendering(camera, renderContext, scene);
	prepareDataForPreviewDrawing();
}
RenderingInterface::~RenderingInterface()
{
	OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[RAYGEN]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[MISS]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[TRIANGLE]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[DISK]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[CALLABLE]));
	OPTIX_CHECK(optixModuleDestroy(m_ptModule));
	OPTIX_CHECK(optixDeviceContextDestroy(m_context));

	glDeleteTextures(1, &m_glTexture);
	glDeleteVertexArrays(1, &m_VAO);
	glDeleteProgram(m_drawProgram);
	CUDA_CHECK(cudaGraphicsUnregisterResource(m_graphicsResource));
	CUDA_CHECK(cuModuleUnload(m_imageModule));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.callablesRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_gasBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_customPrimBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_iasBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sensorSpectralCurvesData)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_materialData)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_spectralData)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_lpBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_renderData)));
}

void RenderingInterface::render(const glm::mat3& colorspaceTransform)
{
	static bool first{ true };
	if (first) [[unlikely]]
		first = false;
	else
		resolveRender(colorspaceTransform);
	if (m_currentSampleCount == 0)
	{
		m_renderingIsFinished = true;
		return;
	}
	updateSubLaunchData();
	updateSamplingState();
	launch();
}
void RenderingInterface::drawPreview()
{
	glClearColor(0.4f, 1.0f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_glTexture);
	glBindVertexArray(m_VAO);
	glUseProgram(m_drawProgram);

	glDrawArrays(GL_TRIANGLES, 0, 3);
}
