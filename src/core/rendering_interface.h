#pragma once

#include <unordered_map>
#include <map>
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
#include "../core/command.h"

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
	CudaCombinedTexture m_envMap{};

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

	CUmodule m_CDFModule{};
	std::array<CUfunction, 4> m_buildCDFFunctions{};
	const std::array<const char*, 4> m_buildCDFFuncNames{
		"buildCDFLatLongRGBUpTo4k",
		"buildCDFLatLongRGBUpTo8k",
		"buildCDFLatLongRGBUpTo16k",
		"buildCDFLatLongRGBUpTo32k",
	};
	CUfunction m_invertCDFToIndicesFunction{};
	const char* m_invertCDFToIndicesFuncName{ "invertCDFToIndices" };

	cudaEvent_t m_execEvent{};
	cudaStream_t m_streams[5]{};

	struct ModelResource
	{
		std::vector<OptixTraversableHandle> gasHandles{};
		std::vector<CUdeviceptr> gasBuffers{};
		std::vector<CUdeviceptr> indexBuffers{};
		std::vector<CUdeviceptr> attributeBuffers{};
		std::vector<CudaImage> images{};
		std::vector<CudaTexture> textures{};
		std::vector<uint32_t> materialIndices{};
	};
	std::map<uint32_t, ModelResource> m_modelResources{};

	struct LightResource
	{
		uint32_t materialIndex{};
	};
	std::map<uint32_t, LightResource> m_lightResources{};

	LaunchParameters m_launchParameters{};

	RenderContext::Mode m_mode{};
	constexpr static inline uint32_t m_rDataComponentSize{ sizeof(double) };
	constexpr static inline uint32_t m_rDataComponentCount{ 4 };
	int m_launchWidth{};
	int m_launchHeight{};
	int m_sampleCount{};
	float m_imageExposure{ 0.0f };

	int m_currentSampleOffset{ 0 };
	int m_currentSampleCount{ 1 };
	int m_processedSampleCount{ 0 };
	bool m_renderingIsFinished{ false };
	bool m_sublaunchIsFinished{ false };

	int m_matDataCount{ 0 };
	std::stack<uint32_t> m_freeMaterialsIndices{};

	struct SpectrumRecord
	{
		int index{};
		int refcount{};
	};
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
	void createCDFBuildProgram();
	void createConstantSBTRecords();
	void createModulesProgramGroupsPipeline();
	void fillSpectralCurvesData();
	void loadLookUpTables();
	void prepareDataForRendering(const Camera& camera, const RenderContext& renderContext);
	void prepareDataForPreviewDrawing();

	void loadModel(SceneData::Model& model);
	void loadLights(SceneData& scene, const Camera& camera);
	void removeModel(uint32_t modelID);
	void removeLight(uint32_t lightID);
	void loadEnvironmentMap(const char* path);

	void fillModelMaterials(RenderingInterface::ModelResource& modelRes, SceneData::Model& model);
	uint32_t fillLightMaterial(const SceneData::MaterialDescriptor& desc);
	void buildGeometryAccelerationStructures(RenderingInterface::ModelResource& modelRes, SceneData::Model& model);
	void buildLightAccelerationStructure(const SceneData& scene, LightType type);
	void uploadLightData(const SceneData& scene, const glm::vec3& cameraPosition, bool resizeBuffers);
	void updateHitgroupSBTRecords(const SceneData& scene);
	void updateInstanceAccelerationStructure(const SceneData& scene, const Camera& camera);


	uint32_t addMaterial(const MaterialData& matData);
	int changeSpectrum(SpectralData::SpectralDataType newSpecType, SpectralData::SpectralDataType oldSpecType);
	uint32_t setNewSpectrum(SpectralData::SpectralDataType type);
	uint32_t getSpectrum(SpectralData::SpectralDataType type);

	int bxdfTypeToIndex(SceneData::BxDF type);
	void updateSubLaunchData();
	void updateSamplingState();
	

	void processCommands(CommandBuffer& commands, RenderContext& renderContext, Camera& camera, SceneData& scene);
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

	void render(CommandBuffer& commands, RenderContext& renderContext, Camera& camera, SceneData& scene);
	void drawPreview(int winWidth, int winHeight) const;
	GLuint getPreview() const { return m_glTexture; }
	bool renderingIsFinished() const { return m_renderingIsFinished; }
	bool sublaunchIsFinished() const { return m_sublaunchIsFinished; }
	int getProcessedSampleCount() const { return m_processedSampleCount; }
	void cleanup();
};
