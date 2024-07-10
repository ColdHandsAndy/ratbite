#pragma once

#include <glad/glad.h>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

#include "scene.h"
#include "window.h"
#include "camera.h"
#include "render_context.h"

class RenderingInterface
{
private:
	OptixDeviceContext m_context{};
	OptixPipeline m_pipeline{};
	OptixModule m_ptModule{};
	OptixTraversableHandle m_gasHandle{};
	CUdeviceptr m_gasBuffer{};
	OptixTraversableHandle m_customPrimHandle{};
	CUdeviceptr m_customPrimBuffer{};
	OptixTraversableHandle m_iasHandle{};
	CUdeviceptr m_iasBuffer{};
	CUdeviceptr m_materialData{};
	CUdeviceptr m_spectralData{};
	CUdeviceptr m_sensorSpectralCurvesData{};
	CUdeviceptr m_lpBuffer{};
	CUdeviceptr m_renderData{};
	enum PathtracerProgramGroup
	{
		RAYGEN,
		MISS,
		TRIANGLE,
		DISK,
		CALLABLE_CONDUCTOR_BXDF,
		CALLABLE_DIELECTRIC_BXDF,
		ALL_GROUPS
	};
	static constexpr uint32_t m_ptProgramGroupCount{ ALL_GROUPS };
	OptixProgramGroup m_ptProgramGroups[m_ptProgramGroupCount]{};
	OptixShaderBindingTable m_sbt{};

	CUmodule m_imageModule{};
	CUfunction m_resolveRenderDataFunc{};
	const std::string m_renderResolveFunctionName{ "renderResolve" };
	GLuint m_glTexture{};
	uint32_t m_VAO{};
	uint32_t m_drawProgram{};
	cudaGraphicsResource_t m_graphicsResource{};
	cudaArray_t m_imageCudaArray{};
	cudaSurfaceObject_t m_imageCudaSurface{};

	cudaEvent_t m_exexEvent{};
	cudaStream_t m_streams[5]{};

	// TEMP
	uint32_t lightEmissionSpectrumIndex{};
	//
	RenderContext::Mode m_mode{};
	constexpr static inline uint32_t m_rDataComponentSize{ sizeof(double) };
	constexpr static inline uint32_t m_rDataComponentCount{ 4 };
	int m_launchWidth{};
	int m_launchHeight{};
	int m_sampleCount{};
	int m_pathLength{};

	int m_currentSampleOffset{ 0 };
	int m_currentSampleCount{ 1 };
	int m_processedSampleCount{ 0 };
	bool m_renderingIsFinished{ false };
	bool m_sublaunchIsFinished{ false };

	template <typename T>
		struct Record
		{
			__align__(OPTIX_SBT_RECORD_ALIGNMENT) uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
			T data;
		};
	struct RecordDataEmpty {};
	typedef uint32_t RecordDataPack32;
	typedef Record<RecordDataEmpty> OptixRecordRaygen;
	typedef Record<RecordDataEmpty> OptixRecordMiss;
	typedef Record<RecordDataEmpty> OptixRecordCallable;
	typedef Record<RecordDataPack32> OptixRecordHitgroup;

	void createOptixContext();
	void createAccelerationStructures(const SceneData& scene, const glm::vec3& cameraPosition);
	void createModulesProgramGroupsPipeline();
	void createRenderResolveProgram();
	void fillMaterials(const SceneData& scene);
	void createSBT(const SceneData& scene);
	void fillSpectralCurvesData();
	void prepareDataForRendering(const Camera& camera, const RenderContext& renderContext, const SceneData& scene);
	void prepareDataForPreviewDrawing();
	
	void resolveRender(const glm::mat3& colorspaceTransform);
	void processChanges(RenderContext& renderContext, Camera& camera, SceneData& scene);
	void updateSubLaunchData();
	void updateSamplingState();
	void launch();
public:
	RenderingInterface(const Camera& camera, const RenderContext& renderContext, const SceneData& scene);
	RenderingInterface() = delete;
	RenderingInterface(RenderingInterface&&) = delete;
	RenderingInterface(const RenderingInterface&) = delete;
	RenderingInterface& operator=(RenderingInterface&&) = delete;
	RenderingInterface& operator=(const RenderingInterface&) = delete;
	~RenderingInterface() = default;

	void render(RenderContext& renderContext, Camera& camera, SceneData& scene, bool changesMade);
	void drawPreview(int winWidth, int winHeight) const;
	bool renderingIsFinished() const { return m_renderingIsFinished; }
	bool sublaunchIsFinished() const { return m_sublaunchIsFinished; }
	int getProcessedSampleCount() const { return m_processedSampleCount; }
	void cleanup();
};
