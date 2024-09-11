#pragma once

#include <unordered_map>
#include <stack>
#include <glad/glad.h>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

#include "../core/scene.h"
#include "../core/window.h"
#include "../core/camera.h"
#include "../core/render_context.h"
#include "../core/launch_parameters.h"
#include "../core/texture.h"

class RenderingInterface
{
private:
	OptixDeviceContext m_context{};
	OptixPipeline m_pipeline{};

	OptixModule m_ptModule{};
	OptixModule m_builtInSphereModule{};

	OptixTraversableHandle m_customPrimHandle{};
	CUdeviceptr m_customPrimBuffer{};
	CUdeviceptr m_spherePrimitiveHandle{};
	CUdeviceptr m_spherePrimBuffer{};

	std::vector<OptixTraversableHandle> m_gasHandles{};
	std::vector<CUdeviceptr> m_gasBuffers{};
	std::vector<CUdeviceptr> m_indexBuffers{};
	std::vector<CUdeviceptr> m_attributeBuffers{};
	std::vector<CudaImage> m_images{};
	std::vector<CudaTexture> m_textures{};
	enum LookUpTable
	{
		CONDUCTOR_ALBEDO = 0,
		DIELECTRIC_OUTER_ALBEDO = 1,
		DIELECTRIC_INNER_ALBEDO = 2,
		REFLECTIVE_DIELECTRIC_OUTER_ALBEDO = 3,
		REFLECTIVE_DIELECTRIC_INNER_ALBEDO = 4,

		DESC
	};
	CudaCombinedTexture m_lookUpTables[DESC]{};

	OptixTraversableHandle m_iasHandle{};
	CUdeviceptr m_iasBuffer{};

	CUdeviceptr m_materialData{};
	CUdeviceptr m_spectralData{};
	CUdeviceptr m_sensorSpectralCurvesData{};
	CUdeviceptr m_spectralBasisData{};
	CUdeviceptr m_diskLights{};
	CUdeviceptr m_sphereLights{};
	CUdeviceptr m_lpBuffer{};
	CUdeviceptr m_renderData{};
	enum PathtracerProgramGroup
	{
		RAYGEN,
		MISS,
		TRIANGLE,
		DISK,
		SPHERE,
		PURE_CONDUCTOR_BXDF,
		PURE_DIELECTRIC_BXDF,
		COMPLEX_SURFACE_BXDF,
		ALL_GROUPS
	};
	static constexpr uint32_t m_ptProgramGroupCount{ ALL_GROUPS };
	OptixProgramGroup m_ptProgramGroups[m_ptProgramGroupCount]{};
	OptixShaderBindingTable m_sbt{};
	static constexpr uint32_t m_spherePrimitiveSBTRecordCount{ 1 };
	static constexpr uint32_t m_customPrimitiveSBTRecordCount{ 1 };
	static constexpr uint32_t m_trianglePrimitiveSBTRecordOffest{ m_spherePrimitiveSBTRecordCount + m_customPrimitiveSBTRecordCount };

	CUmodule m_imageModule{};
	CUfunction m_resolveRenderDataFunc{};
	const std::string m_renderResolveFunctionName{ "renderResolve" };
	GLuint m_glTexture{};
	uint32_t m_VAO{};
	uint32_t m_drawProgram{};
	cudaGraphicsResource_t m_graphicsResource{};
	cudaArray_t m_imageCudaArray{};
	cudaSurfaceObject_t m_imageCudaSurface{};

	cudaEvent_t m_execEvent{};
	cudaStream_t m_streams[5]{};

	LaunchParameters m_launchParameters{};

	RenderContext::Mode m_mode{};
	constexpr static inline uint32_t m_rDataComponentSize{ sizeof(double) };
	constexpr static inline uint32_t m_rDataComponentCount{ 4 };
	int m_launchWidth{};
	int m_launchHeight{};
	int m_sampleCount{};
	int m_pathLength{};
	float m_imageExposure{ 0.0f };

	int m_currentSampleOffset{ 0 };
	int m_currentSampleCount{ 1 };
	int m_processedSampleCount{ 0 };
	bool m_renderingIsFinished{ false };
	bool m_sublaunchIsFinished{ false };

	struct SpectrumRecord
	{
		int index{};
		int refcount{};
	};
	typedef std::unordered_map<SpectralData::SpectralDataType, SpectrumRecord> SpectraMap;
	std::unordered_map<SpectralData::SpectralDataType, SpectrumRecord> m_loadedSpectra{};
	std::stack<int> m_freeSpectra{};

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
	void createRenderResolveProgram();
	void createAccelerationStructures(const SceneData& scene, const glm::vec3& cameraPosition);
	void createModulesProgramGroupsPipeline();
	void fillMaterials(SceneData& scene);
	void createSBT(const SceneData& scene);
	void fillSpectralCurvesData();
	void loadLookUpTables();
	void fillLightData(const SceneData& scene, const glm::vec3& cameraPosition);
	void prepareDataForRendering(const Camera& camera, const RenderContext& renderContext);
	void prepareDataForPreviewDrawing();

	void buildGeometryAccelerationStructures(const SceneData& scene);
	void buildLightAccelerationStructure(const SceneData& scene, LightType type);
	void buildInstanceAccelerationStructure(const SceneData& scene, const glm::vec3& cameraPosition);
	void changeMaterial(int index, const SceneData::MaterialDescriptor& desc, const SceneData::MaterialDescriptor& prevDesc);
	void addMaterial(const SceneData::MaterialDescriptor& desc);
	int bxdfTypeToIndex(SceneData::BxDF type);
	void updateSubLaunchData();
	void updateSamplingState();
	
	void processChanges(RenderContext& renderContext, Camera& camera, SceneData& scene);
	void resolveRender(const glm::mat3& colorspaceTransform);
	void launch();
public:
	RenderingInterface(const Camera& camera, const RenderContext& renderContext, SceneData& scener);
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
	uint32_t setNewSpectrum(SpectralData::SpectralDataType type);
	uint32_t getSpectrum(SpectralData::SpectralDataType type);
	void cleanup();
};
