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
#include "../core/light.h"
#include "../core/util.h"
#include "../core/debug_macros.h"
#include "../core/callbacks.h"
#include "../kernels/optix_programs_desc.h"

void RenderingInterface::createOptixContext()
{
	OptixDeviceContextOptions options{ .logCallbackFunction = optixLogCallback, .logCallbackLevel = 4 };

#ifdef _DEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
	// options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif

	OPTIX_CHECK(optixDeviceContextCreate(0, &options, &m_context));
}
void RenderingInterface::fillMaterials(const SceneData& scene)
{
	std::vector<MaterialData> matData(scene.materialDescriptors.size());
	std::function addSpec{ [this](const SpectralData::SpectralDataType sdt, uint16_t& spectrumIndex)
		{
			if (sdt != SpectralData::SpectralDataType::NONE)
			{
				if (m_loadedSpectra.contains(sdt))
				{
					SpectrumRecord& rec{ m_loadedSpectra.at(sdt) };
					rec.refcount += 1;
					spectrumIndex = rec.index;
				}
				else
				{
					spectrumIndex = m_loadedSpectra.size();
					m_loadedSpectra.insert({sdt, SpectrumRecord{spectrumIndex, 1}});
				}
			}
		} };
	for (int i{ 0 }; i < scene.materialDescriptors.size(); ++i)
	{
		addSpec(scene.materialDescriptors[i].baseIOR, matData[i].indexOfRefractSpectrumDataIndex);
		addSpec(scene.materialDescriptors[i].baseAC, matData[i].absorpCoefSpectrumDataIndex);
		addSpec(scene.materialDescriptors[i].baseEmission, matData[i].emissionSpectrumDataIndex);
		matData[i].bxdfIndexSBT = bxdfTypeToIndex(scene.materialDescriptors[i].bxdf);
		matData[i].mfRoughnessValue = scene.materialDescriptors[i].roughness;
	}
	for(auto& model : scene.models)
		for(auto& mesh : model.meshes)
			for(auto& submesh : mesh.submeshes)
			{
				MaterialData& mat{ matData[submesh.materialIndex] };

				mat.indexType = submesh.indexType;
				CUdeviceptr& iBuf{ m_indexBuffers.emplace_back() };
				size_t iBufferByteSize{ (submesh.indexType == IndexType::UINT_32 ? sizeof(uint32_t) * 3 : sizeof(uint16_t) * 3) * submesh.primitiveCount };
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(iBuf)),
							iBufferByteSize));
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(iBuf), submesh.indices,
							iBufferByteSize,
							cudaMemcpyHostToDevice));
				mat.indices = iBuf;


				mat.attributes |= MaterialData::AttributeTypeBitfield::NORMAL;

				CUdeviceptr& aBuf{ m_normalBuffers.emplace_back() };
				size_t aBufferByteSize{ CONTAINER_ELEMENT_SIZE(submesh.normals) * submesh.normals.size() };
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(aBuf)),
							aBufferByteSize));
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(aBuf), submesh.normals.data(),
							aBufferByteSize,
							cudaMemcpyHostToDevice));
				mat.attributeData = aBuf;
			}
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_materialData), sizeof(MaterialData) * matData.size()));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_materialData), matData.data(), sizeof(MaterialData) * matData.size(), cudaMemcpyHostToDevice));
	DenselySampledSpectrum* spectra{ new DenselySampledSpectrum[m_loadedSpectra.size()]};
	for (const auto& [specType, rec] : m_loadedSpectra)
		spectra[rec.index] = SpectralData::loadSpectrum(specType);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_spectralData), sizeof(DenselySampledSpectrum) * m_loadedSpectra.size()));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_spectralData), spectra, sizeof(DenselySampledSpectrum) * m_loadedSpectra.size(), cudaMemcpyHostToDevice));
	delete[] spectra;
}
void RenderingInterface::fillLightData(const SceneData& scene, const glm::vec3& cameraPosition)
{
	R_ASSERT_LOG(scene.lightCount != 0, "There needs to be at least one light");

	if (scene.diskLights.size() != 0)
	{
		static size_t diskCount{ 0 };
		if (scene.diskLights.size() != diskCount)
		{
			diskCount = scene.diskLights.size();
			if (m_diskLights != CUdeviceptr{})
				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_diskLights)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_diskLights), scene.diskLights.size() * sizeof(DiskLightData)));
			m_launchParameters.lights.disks = m_diskLights;
		}
		DiskLightData* diskLightData{ new DiskLightData[scene.diskLights.size()] };
		for (int i{ 0 }; i < scene.diskLights.size(); ++i)
		{
			const SceneData::DiskLight& dl{ scene.diskLights[i] };
			diskLightData[i] = {
				.position = dl.getPosition() - cameraPosition,
				.powerScale = dl.getPowerScale(),
				.frame = dl.getFrame(),
				.radius = dl.getRadius(),
				.materialIndex = static_cast<uint16_t>(dl.getMaterialIndex())};
		}
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_diskLights), diskLightData, scene.diskLights.size() * sizeof(DiskLightData), cudaMemcpyHostToDevice));
		delete[] diskLightData;
	}

	if (scene.sphereLights.size() != 0)
	{
		static size_t sphereCount{ 0 };
		if (scene.sphereLights.size() != sphereCount)
		{
			sphereCount = scene.sphereLights.size();
			if (m_sphereLights != CUdeviceptr{})
				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sphereLights)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sphereLights), scene.sphereLights.size() * sizeof(SphereLightData)));
			m_launchParameters.lights.spheres = m_sphereLights;
		}
		SphereLightData* sphereLightData{ new SphereLightData[scene.sphereLights.size()] };
		for (int i{ 0 }; i < scene.sphereLights.size(); ++i)
		{
			const SceneData::SphereLight& sl{ scene.sphereLights[i] };
			sphereLightData[i] = {
				.position = sl.getPosition() - cameraPosition,
				.powerScale = sl.getPowerScale(),
				.frame = sl.getFrame(),
				.radius = sl.getRadius(),
				.materialIndex = static_cast<uint16_t>(sl.getMaterialIndex())};
		}
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sphereLights), sphereLightData, scene.sphereLights.size() * sizeof(SphereLightData), cudaMemcpyHostToDevice));
		delete[] sphereLightData;
	}

	uint16_t orderedSizes[KSampleableLightCount]{};
	orderedSizes[KSphereLightIndex] = scene.sphereLights.size();
	orderedSizes[KDiskLightIndex] = scene.diskLights.size();
	if (m_launchParameters.lights.orderedCount == CUdeviceptr{})
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.lights.orderedCount), sizeof(uint16_t) * ARRAYSIZE(orderedSizes)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_launchParameters.lights.orderedCount), orderedSizes, sizeof(uint16_t) * ARRAYSIZE(orderedSizes), cudaMemcpyHostToDevice));

	m_launchParameters.lights.lightCount = static_cast<float>(scene.lightCount);
}
void RenderingInterface::createAccelerationStructures(const SceneData& scene, const glm::vec3& cameraPosition)
{
	buildGeometryAccelerationStructures(scene);
	buildLightAccelerationStructure(scene, LightType::SPHERE);
	buildLightAccelerationStructure(scene, LightType::DISK);
	buildInstanceAccelerationStructure(scene, cameraPosition);
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
		descs[RenderingInterface::CALLABLE_CONDUCTOR_BXDF] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
			.callables = {.moduleDC = optixModule, .entryFunctionNameDC = Program::conductorBxDFName} };
		descs[RenderingInterface::CALLABLE_DIELECTRIC_BXDF] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
			.callables = {.moduleDC = optixModule, .entryFunctionNameDC = Program::dielectricBxDFName} };
		descs[RenderingInterface::CALLABLE_DIELECTRIC_ABSORBING_BXDF] = OptixProgramGroupDesc{
			.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
			.callables = {.moduleDC = optixModule, .entryFunctionNameDC = Program::dielectricAbsorbingBxDFName} };
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
	constexpr int callableCount{ 3 };
	m_sbt.callablesRecordCount = callableCount;
	m_sbt.callablesRecordStrideInBytes = sizeof(OptixRecordCallable);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.callablesRecordBase), m_sbt.callablesRecordStrideInBytes * m_sbt.callablesRecordCount));
	OptixRecordCallable callableRecords[callableCount]{};
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::CALLABLE_CONDUCTOR_BXDF], callableRecords + bxdfTypeToIndex(SceneData::BxDF::CONDUCTOR)));
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::CALLABLE_DIELECTRIC_BXDF], callableRecords + bxdfTypeToIndex(SceneData::BxDF::DIELECTRIC)));
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::CALLABLE_DIELECTRIC_ABSORBING_BXDF], callableRecords + bxdfTypeToIndex(SceneData::BxDF::DIELECTRIC_ABSORBING)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.callablesRecordBase), callableRecords, m_sbt.callablesRecordStrideInBytes * m_sbt.callablesRecordCount, cudaMemcpyHostToDevice));

	uint32_t hitgroupCount{ 0 };
	//Fill hitgroup records for lights (light data)               | Trace stride, trace offset and instance offset affects these
	OptixRecordHitgroup lightHitgroupRecords[KSampleableLightCount]{};
	hitgroupCount += ARRAYSIZE(lightHitgroupRecords);
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::SPHERE], lightHitgroupRecords[0].header));
	OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::DISK], lightHitgroupRecords[1].header));

	//Fill hitgroup records for ordinary geometry (material data) | Trace stride, trace offset and instance offset affects these
	std::vector<OptixRecordHitgroup> matHitgroupRecords{};
	for(auto& model : scene.models)
		for(auto& mesh : model.meshes)
			for(auto& submesh : mesh.submeshes)
			{
				OptixRecordHitgroup record{};
				OPTIX_CHECK(optixSbtRecordPackHeader(m_ptProgramGroups[RenderingInterface::TRIANGLE], record.header));
				record.data = submesh.materialIndex;
				matHitgroupRecords.push_back(record);
			}
	hitgroupCount += matHitgroupRecords.size();

	OptixRecordHitgroup* hitgroups{ new OptixRecordHitgroup[hitgroupCount] };
	int i{ 0 };
	for (auto& hg : lightHitgroupRecords)
		hitgroups[i++] = hg;
	for (auto& hg : matHitgroupRecords)
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

	DenselySampledSpectrum spectralBasis[]{ SpectralData::CIE::BasisR(), SpectralData::CIE::BasisG(), SpectralData::CIE::BasisB() };
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_spectralBasisData), sizeof(DenselySampledSpectrum) * ARRAYSIZE(spectralBasis)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_spectralBasisData), spectralBasis, sizeof(DenselySampledSpectrum) * ARRAYSIZE(spectralBasis), cudaMemcpyHostToDevice));
}
void RenderingInterface::prepareDataForRendering(const Camera& camera, const RenderContext& renderContext)
{
	m_mode = renderContext.getRenderMode();
	m_sampleCount = renderContext.getSampleCount();
	m_pathLength = renderContext.getPathLength();
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
		.camPerspectiveScaleW = static_cast<float>(glm::tan((glm::radians(45.0) * 0.5))) * (static_cast<float>(renderContext.getRenderWidth()) / static_cast<float>(renderContext.getRenderHeight())),
		.camPerspectiveScaleH = static_cast<float>(glm::tan((glm::radians(45.0) * 0.5))) };
	m_launchParameters.maxPathLength = static_cast<uint32_t>(renderContext.getPathLength());
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
		"uniform vec2 uvScale;\n"
		"uniform vec2 uvOffset;\n"

		"in vec2 uv;\n"

		"out vec4 FragColor;\n"

		"void main()\n"
		"{\n"
			"vec2 fitUV = uv * uvScale + uvOffset;\n"
			"vec3 color = texture(tex, fitUV).xyz;\n"
			"if (fitUV.x < 0.0f || fitUV.x > 1.0f || fitUV.y < 0.0f || fitUV.y > 1.0f) color.xyz = vec3(0.0f);\n"
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

void RenderingInterface::buildGeometryAccelerationStructures(const SceneData& scene)
{
	for(auto buf : m_gasBuffers)
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
	m_gasBuffers.clear();

	CUdeviceptr tempBuffer{};

	for (int i{ 0 }; i < scene.models.size(); ++i)
	{
		const SceneData::Model& model{ scene.models[i] };
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

			m_gasHandles.push_back(gasHandle);
			m_gasBuffers.push_back(gasBuffer);
		}
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
void RenderingInterface::buildInstanceAccelerationStructure(const SceneData& scene, const glm::vec3& cameraPosition)
{
	if (m_iasBuffer != CUdeviceptr{})
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_iasBuffer)));

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
	int sbtOffset{ 0 };
	for(auto& model : scene.models)
	{
		for(auto& instance : model.instances)
		{
			instances[inst].instanceId = inst;
			instances[inst].sbtOffset = m_spherePrimitiveSBTRecordCount + m_customPrimitiveSBTRecordCount + sbtOffset;
			instances[inst].traversableHandle = m_gasHandles[instance.meshIndex];
			instances[inst].visibilityMask = 0xFF;
			instances[inst].flags = OPTIX_INSTANCE_FLAG_NONE;
			instances[inst].transform[0]  = instance.transform[0][0];
			instances[inst].transform[1]  = instance.transform[0][1];
			instances[inst].transform[2]  = instance.transform[0][2];
			instances[inst].transform[3]  = -cameraPosition.x + instance.transform[0][3];
			instances[inst].transform[4]  = instance.transform[1][0];
			instances[inst].transform[5]  = instance.transform[1][1];
			instances[inst].transform[6]  = instance.transform[1][2];
			instances[inst].transform[7]  = -cameraPosition.y + instance.transform[1][3];
			instances[inst].transform[8]  = instance.transform[2][0];
			instances[inst].transform[9]  = instance.transform[2][1];
			instances[inst].transform[10] = instance.transform[2][2];
			instances[inst].transform[11] = -cameraPosition.z + instance.transform[2][3];
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
void RenderingInterface::changeMaterial(int index, const SceneData::MaterialDescriptor& desc, const SceneData::MaterialDescriptor& prevDesc)
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
	MaterialData matData{};
	CUDA_CHECK(cudaMemcpy(&matData, reinterpret_cast<MaterialData*>(m_materialData) + index, sizeof(MaterialData), cudaMemcpyDeviceToHost));
	matData.bxdfIndexSBT = static_cast<uint32_t>(bxdfTypeToIndex(desc.bxdf));
	matData.mfRoughnessValue = desc.roughness;
	if (desc.baseIOR != prevDesc.baseIOR)
	{
		remSpecRef(prevDesc.baseIOR);
		matData.indexOfRefractSpectrumDataIndex = static_cast<uint16_t>(setNewSpectrum(desc.baseIOR));
	}
	else
	{
		matData.indexOfRefractSpectrumDataIndex = static_cast<uint16_t>(getSpectrum(desc.baseIOR));
	}
	if (desc.baseAC != prevDesc.baseAC)
	{
		remSpecRef(prevDesc.baseAC);
		matData.absorpCoefSpectrumDataIndex = static_cast<uint16_t>(setNewSpectrum(desc.baseAC));
	}
	else
	{
		matData.absorpCoefSpectrumDataIndex = static_cast<uint16_t>(getSpectrum(desc.baseAC));
	}
	if (desc.baseEmission != prevDesc.baseEmission)
	{
		remSpecRef(prevDesc.baseEmission);
		matData.emissionSpectrumDataIndex = static_cast<uint16_t>(setNewSpectrum(desc.baseEmission));
	}
	else
	{
		matData.emissionSpectrumDataIndex = static_cast<uint16_t>(getSpectrum(desc.baseEmission));
	}

	CUDA_CHECK(cudaMemcpy(reinterpret_cast<MaterialData*>(m_materialData) + index, &matData, sizeof(MaterialData), cudaMemcpyHostToDevice));
}
void RenderingInterface::addMaterial(const SceneData::MaterialDescriptor& desc)
{
	R_ERR_LOG("Adding materials is not implemented yet. Should not happen");
	// MaterialData matData{};
	// matData.bxdfIndexSBT = static_cast<uint32_t>(bxdfTypeToIndex(desc.bxdf));
	// matData.mfRoughnessValue = desc.roughness;
	// matData.indexOfRefractSpectrumDataIndex = static_cast<uint16_t>(setNewSpectrum(desc.baseIOR));
	// matData.absorpCoefSpectrumDataIndex = static_cast<uint16_t>(setNewSpectrum(desc.baseAC));
	// matData.emissionSpectrumDataIndex = static_cast<uint16_t>(setNewSpectrum(desc.baseEmission));
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
		case SceneData::BxDF::CONDUCTOR:
			return 0;
			break;
		case SceneData::BxDF::DIELECTRIC:
			return 1;
			break;
		case SceneData::BxDF::DIELECTRIC_ABSORBING:
			return 2;
			break;
		default:
			R_ERR_LOG("Unknown BxDF type.")
			break;
	}
	return -1;
}

void RenderingInterface::resolveRender(const glm::mat3& colorspaceTransform)
{
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
void RenderingInterface::processChanges(RenderContext& renderContext, Camera& camera, SceneData& scene)
{
	m_mode = renderContext.getRenderMode();

	if (m_sampleCount != renderContext.getSampleCount())
	{
		m_sampleCount = renderContext.getSampleCount();
	}
	if (m_pathLength != renderContext.getPathLength())
	{
		m_pathLength = renderContext.getPathLength();

		m_launchParameters.maxPathLength = m_pathLength;
	}	
	if (m_launchWidth != renderContext.getRenderWidth() || m_launchHeight != renderContext.getRenderHeight())
	{
		m_launchWidth = renderContext.getRenderWidth();
		m_launchHeight = renderContext.getRenderHeight();
		LaunchParameters::ResolutionState newResolutionState{
			.filmWidth = static_cast<uint32_t>(renderContext.getRenderWidth()),
			.filmHeight = static_cast<uint32_t>(renderContext.getRenderHeight()),
			.invFilmWidth = renderContext.getRenderInvWidth(),
			.invFilmHeight = renderContext.getRenderInvHeight(),
			.camPerspectiveScaleW = static_cast<float>(glm::tan((glm::radians(45.0) * 0.5))) * (static_cast<float>(renderContext.getRenderWidth()) / static_cast<float>(renderContext.getRenderHeight())),
			.camPerspectiveScaleH = static_cast<float>(glm::tan((glm::radians(45.0) * 0.5))) };

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
	}
	if (m_imageExposure != renderContext.getImageExposure())
		m_imageExposure = renderContext.getImageExposure();

	if (camera.orientationChanged())
	{
		m_launchParameters.cameraState.camU = camera.getU();
		m_launchParameters.cameraState.camV = camera.getV();
		m_launchParameters.cameraState.camW = camera.getW();
	}
	if (camera.depthOfFieldChanged())
	{
		m_launchParameters.cameraState.depthOfFieldEnabled = camera.depthOfFieldEnabled();
		m_launchParameters.cameraState.appertureSize = camera.getAperture();
		m_launchParameters.cameraState.focusDistance = camera.getFocusDistance();
	}

	bool refillLightData{ camera.positionChanged() || scene.lightChangesMade() };
	if (refillLightData)
		fillLightData(scene, camera.getPosition());

	bool rebuildLightAS{ scene.lightChangesMade() };
	if (rebuildLightAS)
	{
		if (scene.sphereLightsChanged)
			buildLightAccelerationStructure(scene, LightType::SPHERE);
		if (scene.diskLightsChanged)
			buildLightAccelerationStructure(scene, LightType::DISK);
	}

	bool rebuildInstanceAS{ camera.positionChanged() || scene.lightChangesMade() };
	if (rebuildInstanceAS)
		buildInstanceAccelerationStructure(scene, camera.getPosition());

	for (auto& cd : scene.changedDescriptors)
	{
		if (scene.newMaterialDescriptorAdded)
		{
			addMaterial(cd.first);
		}
		else
		{
			SceneData::MaterialDescriptor& prevDesc{ scene.materialDescriptors[cd.second] };
			changeMaterial(cd.second, cd.first, prevDesc);
			prevDesc = cd.first;
		}
	}

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_lpBuffer), &m_launchParameters, sizeof(m_launchParameters), cudaMemcpyHostToDevice, m_streams[1]));

	m_currentSampleCount = 1;
	m_currentSampleOffset = 0;
	m_processedSampleCount = 0;

	scene.acceptChanges();
	renderContext.acceptChanges();
	camera.acceptChanges();
	m_renderingIsFinished = false;
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
	if (m_mode == RenderContext::Mode::RENDER)
		sublaunchSize = std::min(static_cast<int>(std::pow(std::max(3, m_currentSampleCount), 1.5)), maxSublaunchSize);
	else
		sublaunchSize = 1;

	m_currentSampleCount = std::min(std::max(0, m_sampleCount - static_cast<int>(m_currentSampleOffset)), sublaunchSize);
}
void RenderingInterface::launch()
{
	OPTIX_CHECK(optixLaunch(m_pipeline, m_streams[1], m_lpBuffer, sizeof(LaunchParameters), &m_sbt, m_launchWidth, m_launchHeight, 1));
	CUDA_CHECK(cudaEventRecord(m_execEvent, m_streams[1]));
}
void RenderingInterface::cleanup()
{
	OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[RAYGEN]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[MISS]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[TRIANGLE]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[DISK]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[CALLABLE_CONDUCTOR_BXDF]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[CALLABLE_DIELECTRIC_BXDF]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_ptProgramGroups[CALLABLE_DIELECTRIC_ABSORBING_BXDF]));
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
	for(auto buf : m_gasBuffers)
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
	for(auto buf : m_indexBuffers)
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
	for(auto buf : m_normalBuffers)
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_customPrimBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_spherePrimBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_iasBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sensorSpectralCurvesData)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_spectralBasisData)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_lpBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_renderData)));
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

RenderingInterface::RenderingInterface(const Camera& camera, const RenderContext& renderContext, const SceneData& scene)
{
	createOptixContext();
	createRenderResolveProgram();
	createAccelerationStructures(scene, camera.getPosition());
	createModulesProgramGroupsPipeline();
	fillMaterials(scene);
	createSBT(scene);
	fillSpectralCurvesData();
	fillLightData(scene, camera.getPosition());
	prepareDataForRendering(camera, renderContext);
	prepareDataForPreviewDrawing();

	CUDA_CHECK(cudaEventCreateWithFlags(&m_execEvent, cudaEventDisableTiming));
	int lowP;
	int highP;
	CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowP, &highP));
	CUDA_CHECK(cudaStreamCreateWithPriority(m_streams + 0, cudaStreamNonBlocking, highP));
	CUDA_CHECK(cudaStreamCreateWithPriority(m_streams + 1, cudaStreamNonBlocking, lowP));
}

void RenderingInterface::render(RenderContext& renderContext, Camera& camera, SceneData& scene, bool changesMade)
{
	if (cudaEventQuery(m_execEvent) != cudaSuccess)
		return;

	static bool first{ true };
	if (first) [[unlikely]]
		first = false;
	else
	{
		resolveRender(renderContext.getColorspaceTransform());
		CUDA_SYNC_STREAM(m_streams[0]);
	}
	if (changesMade)
		processChanges(renderContext, camera, scene);
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
