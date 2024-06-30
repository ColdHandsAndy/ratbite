#include <iostream>
#include <fstream>
#include <cstdint>
#include <array>
#include <string>
#include <chrono>
#include <filesystem>
#include <format>
#include <functional>
#include <unordered_map>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <optix_stack_size.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/dual_quaternion.hpp>

#include "../core/debug_macros.h"
#include "../core/util_macros.h"
#include "../core/launch_parameters.h"
#include "../kernels/optix_programs_desc.h"
#include "../core/color.h"
#include "../core/spectral.h"
#include "../core/material.h"

struct Camera
{
	glm::dualquat cameraFromWorld{};
	glm::vec3 pos{};
	glm::vec3 u{};
	glm::vec3 v{};
	glm::vec3 w{};

	Camera(const glm::vec3& position, const glm::vec3& viewDirection, const glm::vec3& upDirection)
	{
		pos = position;
		w = glm::normalize(viewDirection);
		u = glm::normalize(glm::cross(glm::normalize(upDirection), w));
		v = glm::cross(w, u);
		glm::mat3 rot{ glm::inverse(glm::mat3{ u, v, w }) };
		glm::vec3 trans{ rot * (-position) };
		cameraFromWorld = glm::dualquat_cast(glm::mat3x4(glm::vec4(u, trans.x), glm::vec4(v, trans.y), glm::vec4(w, trans.z)));
	}
	Camera() = default;
	~Camera() = default;
};
struct Geometry
{
	constexpr static uint32_t triangleCount{ 30 };
	constexpr static uint32_t geometryMaterialCount{ 3 };
	constexpr static uint32_t trianglePrimSBTCount{ geometryMaterialCount };

	const std::array<glm::vec4, triangleCount * 3> vertices
	{ {
		// Floor
		{    -0.0f,    0.0f,    0.0f, 0.0f  },
		{    -0.0f,    0.0f,  559.2f, 0.0f  },
		{  -556.0f,    0.0f,  559.2f, 0.0f  },

		{    -0.0f,    0.0f,    0.0f, 0.0f  },
		{  -556.0f,    0.0f,  559.2f, 0.0f  },
		{  -556.0f,    0.0f,    0.0f, 0.0f  },

		// Ceiling
		{    -0.0f,  548.8f,    0.0f, 0.0f  },
		{  -556.0f,  548.8f,    0.0f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },

		{    -0.0f,  548.8f,    0.0f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },
		{    -0.0f,  548.8f,  559.2f, 0.0f  },

		// Back wall
		{    -0.0f,    0.0f,  559.2f, 0.0f  },
		{    -0.0f,  548.8f,  559.2f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },

		{    -0.0f,    0.0f,  559.2f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },
		{  -556.0f,    0.0f,  559.2f, 0.0f  },

		// Right wall
		{    -0.0f,    0.0f,    0.0f, 0.0f  },
		{    -0.0f,  548.8f,    0.0f, 0.0f  },
		{    -0.0f,  548.8f,  559.2f, 0.0f  },

		{    -0.0f,    0.0f,    0.0f, 0.0f  },
		{    -0.0f,  548.8f,  559.2f, 0.0f  },
		{    -0.0f,    0.0f,  559.2f, 0.0f  },

		// Left wall
		{  -556.0f,    0.0f,    0.0f, 0.0f  },
		{  -556.0f,    0.0f,  559.2f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },

		{  -556.0f,    0.0f,    0.0f, 0.0f  },
		{  -556.0f,  548.8f,  559.2f, 0.0f  },
		{  -556.0f,  548.8f,    0.0f, 0.0f  },

		// Short block
		{  -130.0f,  165.0f,   65.0f, 0.0f  },
		{  - 82.0f,  165.0f,  225.0f, 0.0f  },
		{  -242.0f,  165.0f,  274.0f, 0.0f  },

		{  -130.0f,  165.0f,   65.0f, 0.0f  },
		{  -242.0f,  165.0f,  274.0f, 0.0f  },
		{  -290.0f,  165.0f,  114.0f, 0.0f  },

		{  -290.0f,    0.0f,  114.0f, 0.0f  },
		{  -290.0f,  165.0f,  114.0f, 0.0f  },
		{  -240.0f,  165.0f,  272.0f, 0.0f  },

		{  -290.0f,    0.0f,  114.0f, 0.0f  },
		{  -240.0f,  165.0f,  272.0f, 0.0f  },
		{  -240.0f,    0.0f,  272.0f, 0.0f  },

		{  -130.0f,    0.0f,   65.0f, 0.0f  },
		{  -130.0f,  165.0f,   65.0f, 0.0f  },
		{  -290.0f,  165.0f,  114.0f, 0.0f  },

		{  -130.0f,    0.0f,   65.0f, 0.0f  },
		{  -290.0f,  165.0f,  114.0f, 0.0f  },
		{  -290.0f,    0.0f,  114.0f, 0.0f  },

		{   -82.0f,    0.0f,  225.0f, 0.0f  },
		{   -82.0f,  165.0f,  225.0f, 0.0f  },
		{  -130.0f,  165.0f,   65.0f, 0.0f  },

		{  - 82.0f,    0.0f,  225.0f, 0.0f  },
		{  -130.0f,  165.0f,   65.0f, 0.0f  },
		{  -130.0f,    0.0f,   65.0f, 0.0f  },

		{  -240.0f,    0.0f,  272.0f, 0.0f  },
		{  -240.0f,  165.0f,  272.0f, 0.0f  },
		{   -82.0f,  165.0f,  225.0f, 0.0f  },

		{  -240.0f,    0.0f,  272.0f, 0.0f  },
		{   -82.0f,  165.0f,  225.0f, 0.0f  },
		{   -82.0f,    0.0f,  225.0f, 0.0f  },

		// Tall block
		{  -423.0f,  330.0f,  247.0f, 0.0f  },
		{  -265.0f,  330.0f,  296.0f, 0.0f  },
		{  -314.0f,  330.0f,  455.0f, 0.0f  },

		{  -423.0f,  330.0f,  247.0f, 0.0f  },
		{  -314.0f,  330.0f,  455.0f, 0.0f  },
		{  -472.0f,  330.0f,  406.0f, 0.0f  },

		{  -423.0f,    0.0f,  247.0f, 0.0f  },
		{  -423.0f,  330.0f,  247.0f, 0.0f  },
		{  -472.0f,  330.0f,  406.0f, 0.0f  },

		{  -423.0f,    0.0f,  247.0f, 0.0f  },
		{  -472.0f,  330.0f,  406.0f, 0.0f  },
		{  -472.0f,    0.0f,  406.0f, 0.0f  },

		{  -472.0f,    0.0f,  406.0f, 0.0f  },
		{  -472.0f,  330.0f,  406.0f, 0.0f  },
		{  -314.0f,  330.0f,  456.0f, 0.0f  },

		{  -472.0f,    0.0f,  406.0f, 0.0f  },
		{  -314.0f,  330.0f,  456.0f, 0.0f  },
		{  -314.0f,    0.0f,  456.0f, 0.0f  },

		{  -314.0f,    0.0f,  456.0f, 0.0f  },
		{  -314.0f,  330.0f,  456.0f, 0.0f  },
		{  -265.0f,  330.0f,  296.0f, 0.0f  },

		{  -314.0f,    0.0f,  456.0f, 0.0f  },
		{  -265.0f,  330.0f,  296.0f, 0.0f  },
		{  -265.0f,    0.0f,  296.0f, 0.0f  },

		{  -265.0f,    0.0f,  296.0f, 0.0f  },
		{  -265.0f,  330.0f,  296.0f, 0.0f  },
		{  -423.0f,  330.0f,  247.0f, 0.0f  },

		{  -265.0f,    0.0f,  296.0f, 0.0f  },
		{  -423.0f,  330.0f,  247.0f, 0.0f  },
		{  -423.0f,    0.0f,  247.0f, 0.0f  },

		// Ceiling light
		/*
			{  -343.0f,  548.6f,  227.0f, 0.0f  },
			{  -213.0f,  548.6f,  227.0f, 0.0f  },
			{  -213.0f,  548.6f,  332.0f, 0.0f  },

			{  -343.0f,  548.6f,  227.0f, 0.0f  },
			{  -213.0f,  548.6f,  332.0f, 0.0f  },
			{  -343.0f,  548.6f,  332.0f, 0.0f  }
			*/
	} };

	std::array<uint32_t, triangleCount> SBTIndices
	{ {
		0, 0,                          // Floor         -- white lambert
		0, 0,                          // Ceiling       -- white lambert
		0, 0,                          // Back wall     -- white lambert
		1, 1,                          // Right wall    -- green lambert
		2, 2,                          // Left wall     -- red lambert
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Short block   -- white lambert
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
		//3, 3                           // Ceiling light -- emmissive
	} };

	static constexpr uint32_t lightCount{ 1 };
	struct Disk
	{
		const glm::vec3 pos{ -278.0f, 514.0f, 279.5f };
		const float radius{ 100.0f };
		const float area{ 2.0f * glm::pi<float>() * radius };
		const glm::vec3 normal{ glm::normalize(glm::vec3{0.0f, -1.0f, 0.0f}) };
		const glm::quat frame{ genDiskFrame() };
		const float scale{ 1.5f };
		//uint32_t pdfStructureIndex{};

		glm::quat genDiskFrame() const
		{
			glm::vec3 tang{ glm::normalize(glm::cross(glm::abs(glm::dot(glm::vec3{0.0f, 1.0f, 0.0f}, normal)) < 0.9999f ? glm::vec3{0.0f, 1.0f, 0.0f} : glm::vec3{1.0f, 0.0f, 0.0f}, normal)) };
			glm::vec3 bitang{ glm::cross(normal, tang) };
			return glm::quat_cast(glm::mat3(tang, bitang, normal));
		}

		OptixAabb getOptixAABB() const
		{
			OptixAabb t{};
			float xSpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.x * normal.x)) };
			t.maxX = xSpan == 0.0f ? std::nextafterf(pos.x, FLT_MAX) : pos.x + xSpan * radius;
			t.minX = xSpan == 0.0f ? std::nextafterf(pos.x, -FLT_MAX) : pos.x - xSpan * radius;
			float ySpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.y * normal.y)) };
			t.maxY = ySpan == 0.0f ? std::nextafterf(pos.y, FLT_MAX) : pos.y + ySpan * radius;
			t.minY = ySpan == 0.0f ? std::nextafterf(pos.y, -FLT_MAX) : pos.y - ySpan * radius;
			float zSpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.z * normal.z)) };
			t.maxZ = zSpan == 0.0f ? std::nextafterf(pos.z, FLT_MAX) : pos.z + zSpan * radius;
			t.minZ = zSpan == 0.0f ? std::nextafterf(pos.z, -FLT_MAX) : pos.z - zSpan * radius;
			return t;
		}
		OptixAabb getOptixAABB(const glm::vec3& translation) const
		{
			OptixAabb t{};
			float xSpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.x * normal.x)) };
			t.maxX = xSpan == 0.0f ? std::nextafterf(pos.x, FLT_MAX) : pos.x + xSpan * radius;
			t.minX = xSpan == 0.0f ? std::nextafterf(pos.x, -FLT_MAX) : pos.x - xSpan * radius;
			t.maxX += translation.x;
			t.minX += translation.x;
			float ySpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.y * normal.y)) };
			t.maxY = ySpan == 0.0f ? std::nextafterf(pos.y, FLT_MAX) : pos.y + ySpan * radius;
			t.minY = ySpan == 0.0f ? std::nextafterf(pos.y, -FLT_MAX) : pos.y - ySpan * radius;
			t.maxY += translation.y;
			t.minY += translation.y;
			float zSpan{ glm::max(0.0f, glm::sqrt(1.0f - normal.z * normal.z)) };
			t.maxZ = zSpan == 0.0f ? std::nextafterf(pos.z, FLT_MAX) : pos.z + zSpan * radius;
			t.minZ = zSpan == 0.0f ? std::nextafterf(pos.z, -FLT_MAX) : pos.z - zSpan * radius;
			t.maxZ += translation.z;
			t.minZ += translation.z;
			return t;
		}
	} diskLight{};
	static constexpr uint32_t lightMaterialIndex{ 3 };

	struct MaterialDescriptor //We need 'MaterialDescrioptor' to find material data before rendering
	{
		SpectralData::SpectralDataType baseIOR{};
		SpectralData::SpectralDataType baseAC{};
		SpectralData::SpectralDataType baseEmission{};
		float roughness{};
	};
	const std::array<MaterialDescriptor, geometryMaterialCount + lightCount> materialDescriptors
	{ {
		MaterialDescriptor{.baseIOR = SpectralData::SpectralDataType::C_METAL_TIO2_IOR, .baseAC = SpectralData::SpectralDataType::C_METAL_TIO2_AC, .baseEmission = SpectralData::SpectralDataType::DESC, .roughness = 0.0f},
		MaterialDescriptor{.baseIOR = SpectralData::SpectralDataType::C_METAL_AU_IOR, .baseAC = SpectralData::SpectralDataType::C_METAL_AU_AC, .baseEmission = SpectralData::SpectralDataType::DESC, .roughness = 0.0f},
		MaterialDescriptor{.baseIOR = SpectralData::SpectralDataType::C_METAL_CU_IOR, .baseAC = SpectralData::SpectralDataType::C_METAL_CU_AC, .baseEmission = SpectralData::SpectralDataType::DESC, .roughness = 0.0f},
		MaterialDescriptor{.baseIOR = SpectralData::SpectralDataType::C_METAL_AG_IOR, .baseAC = SpectralData::SpectralDataType::C_METAL_AG_AC, .baseEmission = SpectralData::SpectralDataType::ILLUM_D65, .roughness = 0.0f},
	} };
	std::array<MaterialData, geometryMaterialCount + lightCount> materialData
	{ {
		MaterialData{.bxdfIndexSBT = 0, .indexOfRefractSpectrumDataIndex = 0, .absorpCoefSpectrumDataIndex = 0, .emissionSpectrumDataIndex = 0, .mfRoughnessValue = 0.0f},
		MaterialData{.bxdfIndexSBT = 0, .indexOfRefractSpectrumDataIndex = 0, .absorpCoefSpectrumDataIndex = 0, .emissionSpectrumDataIndex = 0, .mfRoughnessValue = 0.0f},
		MaterialData{.bxdfIndexSBT = 0, .indexOfRefractSpectrumDataIndex = 0, .absorpCoefSpectrumDataIndex = 0, .emissionSpectrumDataIndex = 0, .mfRoughnessValue = 0.0f},
		MaterialData{.bxdfIndexSBT = 0, .indexOfRefractSpectrumDataIndex = 0, .absorpCoefSpectrumDataIndex = 0, .emissionSpectrumDataIndex = 0, .mfRoughnessValue = 0.0f},
	} };
} static geometryData;
struct ImageData
{
	uint32_t width{};
	uint32_t height{};
	size_t renderDataSize{};
	constexpr static inline uint32_t rDataComponentSize{ sizeof(double) };
	constexpr static inline uint32_t rDataComponentCount{ 4 };

	CUdeviceptr renderData{};
	bool mapped{ false };

	uint32_t glTexture{};
	cudaGraphicsResource_t graphicsResource{};
	cudaArray_t presentDataCArray{};
	cudaSurfaceObject_t presentDataCSurface{};

	uint32_t VAO{};
	uint32_t drawProgram{};

	glm::mat3 colorspaceTransform{ glm::identity<glm::mat3>() };

	ImageData(uint32_t iWidth, uint32_t iHeight) : width{ iWidth }, height{ iHeight }, renderDataSize{ rDataComponentSize * rDataComponentCount * iWidth * iHeight }
	{
	}
	~ImageData()
	{
	}

	void drawImage()
	{
		glClearColor(0.4f, 1.0f, 0.8f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, glTexture);
		glBindVertexArray(VAO);
		glUseProgram(drawProgram);

		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	void initialize()
	{
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&renderData), renderDataSize));
		glGenTextures(1, &glTexture);
		glBindTexture(GL_TEXTURE_2D, glTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);
		CUDA_CHECK(cudaGraphicsGLRegisterImage(&graphicsResource, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

		glGenVertexArrays(1, &VAO);

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
		checkShaderCompileErrors(vertexShader);

		uint32_t fragmentShader{ glCreateShader(GL_FRAGMENT_SHADER) };
		glShaderSource(fragmentShader, 1, &fShaderCode, NULL);
		glCompileShader(fragmentShader);
		checkShaderCompileErrors(fragmentShader);

		drawProgram = glCreateProgram();
		glAttachShader(drawProgram, vertexShader);
		glAttachShader(drawProgram, fragmentShader);
		glLinkProgram(drawProgram);
		checkProgramLinkingErrors(drawProgram);

		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
	}
	void destroy()
	{
		CUDA_CHECK(cudaGraphicsUnregisterResource(graphicsResource));
	}

	void setColorspace(Color::RGBColorspace cs)
	{
		colorspaceTransform = Color::generateColorspaceConversionMatrix(cs);
	}

	CUdeviceptr getRenderingData()
	{
		R_ASSERT(renderData != CUdeviceptr{});
		return renderData;
	}

	void mapData()
	{
		if (mapped)
		{
			R_LOG("Image Resource is already mapped\n");
			return;
		}
		CUDA_CHECK(cudaGraphicsMapResources(1, &graphicsResource));
		CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&presentDataCArray, graphicsResource, 0, 0));
		cudaResourceDesc resDesc{ .resType = cudaResourceTypeArray, .res = { presentDataCArray } };
		CUDA_CHECK(cudaCreateSurfaceObject(&presentDataCSurface, &resDesc));
		mapped = true;
	}
	void unmapData()
	{
		if (!mapped)
		{
			R_LOG("Image Resource is not mapped\n");
			return;
		}
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &graphicsResource));
		mapped = false;
	}

private:
	void checkShaderCompileErrors(uint32_t id)
	{
		int success;
		char infoLog[2048];

		glGetShaderiv(id, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(id, 1024, NULL, infoLog);
			std::cerr << std::format("OpenGL shader compilation error.\n\tLog: {}\n", infoLog);
		}
	}
	void checkProgramLinkingErrors(uint32_t id)
	{
		int success;
		char infoLog[2048];

		glGetProgramiv(id, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(id, 1024, NULL, infoLog);
			std::cerr << std::format("OpenGL program linking error.\n\tLog: {}\n", infoLog);
		}
	}
};

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

class CudaOptixState
{
public:
	OptixDeviceContext context{};

	OptixPipeline pipeline{};

	OptixModule ptModule{};

	OptixTraversableHandle gasHandle{};
	CUdeviceptr gasBuffer{};
	OptixTraversableHandle customPrimHandle{};
	CUdeviceptr customPrimBuffer{};
	OptixTraversableHandle iasHandle{};
	CUdeviceptr iasBuffer{};

	CUdeviceptr materialData{};
	CUdeviceptr spectralData{};

	CUdeviceptr sensorSpectralCurvesData{};

	CUdeviceptr lpBuffer{};


	CUmodule imageResolveModule{};

	CUfunction resolveImageDataFunc{};
	const std::string imageResolveFunctionName{ "resolveImage" };

	
	enum PathtracerProgramGroup
	{
		RAYGEN,
		MISS,
		TRIANGLE,
		DISK,
		CALLABLE,
		ALL_GROUPS
	};
	static constexpr uint32_t ptProgramGroupCount{ ALL_GROUPS };
	OptixProgramGroup ptProgramGroups[ptProgramGroupCount]{};

	OptixShaderBindingTable sbt{};

	void cleanup()
	{
		OPTIX_CHECK(optixPipelineDestroy(pipeline));
		OPTIX_CHECK(optixProgramGroupDestroy(ptProgramGroups[RAYGEN]));
		OPTIX_CHECK(optixProgramGroupDestroy(ptProgramGroups[MISS]));
		OPTIX_CHECK(optixProgramGroupDestroy(ptProgramGroups[TRIANGLE]));
		OPTIX_CHECK(optixProgramGroupDestroy(ptProgramGroups[DISK]));
		OPTIX_CHECK(optixProgramGroupDestroy(ptProgramGroups[CALLABLE]));
		OPTIX_CHECK(optixModuleDestroy(ptModule));
		OPTIX_CHECK(optixDeviceContextDestroy(context));

		CUDA_CHECK(cuModuleUnload(imageResolveModule));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.callablesRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(gasBuffer)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(customPrimBuffer)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(iasBuffer)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sensorSpectralCurvesData)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(materialData)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(spectralData)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(lpBuffer)));
	}
} stateCudaOptix;
static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*);
static void GLAPIENTRY openGLLogCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);

void initializeOpenGL(GLFWwindow** window, uint32_t windowWidth, uint32_t windowHeight)
{
	R_ASSERT(glfwInit());
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	*window = glfwCreateWindow(windowWidth, windowHeight, "ratbite", NULL, NULL);
	R_ASSERT(*window != nullptr);
	glfwMakeContextCurrent(*window);
	R_ASSERT(gladLoadGL());

#ifdef _DEBUG
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(openGLLogCallback, 0);
#endif

	glEnable(GL_FRAMEBUFFER_SRGB);
	glViewport(0, 0, windowWidth, windowHeight);
}
void initializeCUDA()
{
	CUDA_CHECK(cudaFree(0));
	CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
}
void initializeOptiX()
{
	OPTIX_CHECK(optixInit());
}
OptixDeviceContext createOptixContext()
{
	OptixDeviceContext context{};
	OptixDeviceContextOptions options{ .logCallbackFunction = optixLogCallback, .logCallbackLevel = 4 };

#ifdef _DEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

	OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));

	return context;
}
void createAccelerationStructures(CudaOptixState& state, const glm::vec3& cameraPosition, const Geometry& geometry)
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

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&vertexBuffer), sizeof(geometry.vertices[0]) * geometry.vertices.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbtOffsetBuffer), sizeof(geometry.SBTIndices[0]) * geometry.SBTIndices.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&worldToWorldCameraTransformBuffer), sizeof(preTransform)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(vertexBuffer), geometry.vertices.data(), sizeof(geometry.vertices[0]) * geometry.vertices.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbtOffsetBuffer), geometry.SBTIndices.data(), sizeof(geometry.SBTIndices[0]) * geometry.SBTIndices.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(worldToWorldCameraTransformBuffer), preTransform, sizeof(preTransform), cudaMemcpyHostToDevice));

	{
		OptixAccelBuildOptions accelBuildOptions{
			.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
			.operation = OPTIX_BUILD_OPERATION_BUILD };
		uint32_t geometryFlags[geometry.trianglePrimSBTCount]{};
		for (int i{ 0 }; i < ARRAYSIZE(geometryFlags); ++i)
			geometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
		OptixBuildInput gasBuildInputs[]{
			OptixBuildInput{.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
				.triangleArray = OptixBuildInputTriangleArray{.vertexBuffers = &vertexBuffer,
					.numVertices = static_cast<unsigned int>(geometry.vertices.size()),
					.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
					.vertexStrideInBytes = sizeof(float) * 4,
					.preTransform = worldToWorldCameraTransformBuffer,
					.flags = geometryFlags,
					.numSbtRecords = geometry.trianglePrimSBTCount,
					.sbtIndexOffsetBuffer = sbtOffsetBuffer,
					.sbtIndexOffsetSizeInBytes = sizeof(geometry.SBTIndices[0]),
					.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 } } };
		OptixAccelBufferSizes computedBufferSizes{};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accelBuildOptions, gasBuildInputs, ARRAYSIZE(gasBuildInputs), &computedBufferSizes));
		size_t compactedSizeOffset{ ALIGNED_SIZE(computedBufferSizes.tempSizeInBytes, 8ull) };
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), compactedSizeOffset + sizeof(size_t)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.gasBuffer), computedBufferSizes.outputSizeInBytes));
		OptixAccelEmitDesc emittedProperty{};
		emittedProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emittedProperty.result = reinterpret_cast<CUdeviceptr>(reinterpret_cast<uint8_t*>(tempBuffer) + compactedSizeOffset);
		OPTIX_CHECK(optixAccelBuild(state.context, 0, &accelBuildOptions,
							  gasBuildInputs, ARRAYSIZE(gasBuildInputs),
							  tempBuffer, computedBufferSizes.tempSizeInBytes,
							  state.gasBuffer, computedBufferSizes.outputSizeInBytes, &state.gasHandle, &emittedProperty, 1));
		size_t compactedGasSize{};
		CUDA_CHECK(cudaMemcpy(&compactedGasSize, reinterpret_cast<void*>(emittedProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
		if (compactedGasSize < computedBufferSizes.outputSizeInBytes)
		{
			CUdeviceptr noncompactedGasBuffer{ state.gasBuffer };
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.gasBuffer), compactedGasSize));
			OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gasHandle, state.gasBuffer, compactedGasSize, &state.gasHandle));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(noncompactedGasBuffer)));
		}
	}

	{
		OptixAabb aabbs[]{ geometry.diskLight.getOptixAABB(-cameraPosition) };
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&aabbBuffer), sizeof(OptixAabb) * geometry.lightCount));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(aabbBuffer), aabbs, sizeof(OptixAabb) * geometry.lightCount, cudaMemcpyHostToDevice));
		OptixAccelBuildOptions customPrimAccelBuildOptions{
			.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
			.operation = OPTIX_BUILD_OPERATION_BUILD };
		uint32_t geometryFlags[geometry.lightCount]{};
		for (int i{ 0 }; i < ARRAYSIZE(geometryFlags); ++i)
			geometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
		OptixBuildInput customPrimBuildInputs[]{
			OptixBuildInput{.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
				.customPrimitiveArray = OptixBuildInputCustomPrimitiveArray{.aabbBuffers = &aabbBuffer,
					.numPrimitives = geometry.lightCount,
					.flags = geometryFlags,
					.numSbtRecords = geometry.lightCount,
					.primitiveIndexOffset = 0}} };
		OptixAccelBufferSizes computedBufferSizes{};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &customPrimAccelBuildOptions, customPrimBuildInputs, ARRAYSIZE(customPrimBuildInputs), &computedBufferSizes));
		size_t compactedSizeOffset{ ALIGNED_SIZE(computedBufferSizes.tempSizeInBytes, 8ull) };
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), compactedSizeOffset + sizeof(size_t)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.customPrimBuffer), computedBufferSizes.outputSizeInBytes));
		OptixAccelEmitDesc emittedProperty{};
		emittedProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emittedProperty.result = reinterpret_cast<CUdeviceptr>(reinterpret_cast<uint8_t*>(tempBuffer) + compactedSizeOffset);
		OPTIX_CHECK(optixAccelBuild(state.context, 0, &customPrimAccelBuildOptions,
							  customPrimBuildInputs, ARRAYSIZE(customPrimBuildInputs),
							  tempBuffer, computedBufferSizes.tempSizeInBytes,
							  state.customPrimBuffer, computedBufferSizes.outputSizeInBytes, &state.customPrimHandle, &emittedProperty, 1));
		size_t compactedSize{};
		CUDA_CHECK(cudaMemcpy(&compactedSize, reinterpret_cast<void*>(emittedProperty.result), sizeof(size_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
		if (compactedSize < computedBufferSizes.outputSizeInBytes)
		{
			CUdeviceptr noncompactedBuffer{ state.customPrimBuffer };
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.customPrimBuffer), compactedSize));
			OPTIX_CHECK(optixAccelCompact(state.context, 0, state.customPrimHandle, state.customPrimBuffer, compactedSize, &state.customPrimHandle));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(noncompactedBuffer)));
		}
	}

	OptixInstance* instances{};
	CUDA_CHECK(cudaHostAlloc(&instances, sizeof(OptixInstance) * instanceCount, cudaHostAllocMapped));
	CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&instanceBuffer), instances, 0));
	instances[0].instanceId = 0;
	instances[0].sbtOffset = 0;
	instances[0].traversableHandle = state.gasHandle;
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
	instances[1].sbtOffset = geometry.trianglePrimSBTCount;
	instances[1].traversableHandle = state.customPrimHandle;
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
	OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accelBuildOptions, iasBuildInputs, ARRAYSIZE(iasBuildInputs), &computedBufferSizes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), computedBufferSizes.tempSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.iasBuffer), computedBufferSizes.outputSizeInBytes));
	OPTIX_CHECK(optixAccelBuild(state.context, 0, &accelBuildOptions,
							 iasBuildInputs, ARRAYSIZE(iasBuildInputs), 
							 tempBuffer, computedBufferSizes.tempSizeInBytes, state.iasBuffer,
							 computedBufferSizes.outputSizeInBytes, &state.iasHandle, nullptr, 0));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(vertexBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbtOffsetBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(aabbBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(worldToWorldCameraTransformBuffer)));
}
void createModulesProgramGroupsPipeline(CudaOptixState& state)
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

	OptixModule& optixModule{ state.ptModule };
	OPTIX_CHECK_LOG(optixModuleCreate(state.context, &moduleCompileOptions, &pipelineCompileOptions, input, inputSize, OPTIX_LOG, &OPTIX_LOG_SIZE, &optixModule));
	delete[] input;


	OptixProgramGroupOptions programGroupOptions{ .payloadType = payloadTypes + 0 };
	{
		OptixProgramGroupDesc descs[state.ptProgramGroupCount]{};
		descs[CudaOptixState::RAYGEN] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN, .raygen = {.module = optixModule, .entryFunctionName = Program::raygenName} };
		descs[CudaOptixState::MISS] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_MISS, .raygen = {.module = optixModule, .entryFunctionName = Program::missName} };
		descs[CudaOptixState::TRIANGLE] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP, .hitgroup = {.moduleCH = optixModule, .entryFunctionNameCH = Program::closehitTriangleName} };
		descs[CudaOptixState::DISK] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP, .hitgroup = {.moduleCH = optixModule, .entryFunctionNameCH = Program::closehitDiskName, .moduleIS = optixModule, .entryFunctionNameIS = Program::intersectionDiskName} };
		descs[CudaOptixState::CALLABLE] = OptixProgramGroupDesc{ .kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES, .callables = {.moduleDC = optixModule, .entryFunctionNameDC = Program::callableName} };
		OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, descs, ARRAYSIZE(descs),
										  &programGroupOptions,
										  OPTIX_LOG, &OPTIX_LOG_SIZE,
										  state.ptProgramGroups));
	}
	OptixPipelineLinkOptions pipelineLinkOptions{ .maxTraceDepth = Program::maxTraceDepth };

	OPTIX_CHECK_LOG(optixPipelineCreate(state.context,
				&pipelineCompileOptions,
				&pipelineLinkOptions,
				state.ptProgramGroups,
				state.ptProgramGroupCount,
				OPTIX_LOG, &OPTIX_LOG_SIZE,
				&state.pipeline));

	OptixStackSizes stackSizes{};
	for (auto pg : state.ptProgramGroups)
		OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, state.pipeline));

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
		state.pipeline,
		dcStackSizeTraversal,
		dcStackSizeState,
		ccStackSize,
		maxTraversalDepth));
}
void createSBT(CudaOptixState& state, const Geometry& geometry)
{
	//Fill raygen and miss records
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.sbt.raygenRecord), sizeof(OptixRecordRaygen)));
	OptixRecordRaygen raygenRecord{};
	OPTIX_CHECK(optixSbtRecordPackHeader(state.ptProgramGroups[CudaOptixState::RAYGEN], &raygenRecord));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.sbt.raygenRecord), &raygenRecord, sizeof(OptixRecordRaygen), cudaMemcpyHostToDevice));

	state.sbt.missRecordCount = 1;
	state.sbt.missRecordStrideInBytes = sizeof(OptixRecordMiss);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.sbt.missRecordBase), state.sbt.missRecordStrideInBytes * state.sbt.missRecordCount));
	OptixRecordMiss missRecord{};
	OPTIX_CHECK(optixSbtRecordPackHeader(state.ptProgramGroups[CudaOptixState::MISS], &missRecord));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.sbt.missRecordBase), &missRecord, state.sbt.missRecordStrideInBytes * state.sbt.missRecordCount, cudaMemcpyHostToDevice));

	//Fill callable records (bxdfs)
	state.sbt.callablesRecordCount = 1;
	state.sbt.callablesRecordStrideInBytes = sizeof(OptixRecordCallable);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.sbt.callablesRecordBase), state.sbt.callablesRecordStrideInBytes * state.sbt.callablesRecordCount));
	OptixRecordCallable callableRecord{};
	OPTIX_CHECK(optixSbtRecordPackHeader(state.ptProgramGroups[CudaOptixState::CALLABLE], &callableRecord));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.sbt.callablesRecordBase), &callableRecord, state.sbt.callablesRecordStrideInBytes * state.sbt.callablesRecordCount, cudaMemcpyHostToDevice));

	uint32_t hitgroupCount{ 0 };
	//Fill hitgroup records for ordinary geometry (material data) | Trace stride, trace offset and instance offset affects these
	OptixRecordHitgroup matHitgroupRecords[geometry.trianglePrimSBTCount]{};
	hitgroupCount += ARRAYSIZE(matHitgroupRecords);
	OPTIX_CHECK(optixSbtRecordPackHeader(state.ptProgramGroups[CudaOptixState::TRIANGLE], matHitgroupRecords[0].header));
	for (int i{ 1 }; i < ARRAYSIZE(matHitgroupRecords); ++i)
		memcpy(matHitgroupRecords[i].header, matHitgroupRecords[0].header, sizeof(matHitgroupRecords[0].header));
	for (int i{ 0 }; i < ARRAYSIZE(matHitgroupRecords); ++i)
		matHitgroupRecords[i].data = static_cast<uint32_t>(i);

	//Fill hitgroup records for lights (light data)               | Trace stride, trace offset and instance offset affects these
	OptixRecordHitgroup lightHitgroupRecords[geometry.lightCount]{};
	hitgroupCount += ARRAYSIZE(lightHitgroupRecords);
	OPTIX_CHECK(optixSbtRecordPackHeader(state.ptProgramGroups[CudaOptixState::DISK], lightHitgroupRecords[0].header));
	lightHitgroupRecords[0].data = geometry.lightMaterialIndex;

	OptixRecordHitgroup* hitgroups{ new OptixRecordHitgroup[hitgroupCount] };
	int i{ 0 };
	for (auto& hg : matHitgroupRecords)
	hitgroups[i++] = hg;
	for (auto& hg : lightHitgroupRecords)
	hitgroups[i++] = hg;
	state.sbt.hitgroupRecordCount = hitgroupCount;
	state.sbt.hitgroupRecordStrideInBytes = sizeof(OptixRecordHitgroup);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.sbt.hitgroupRecordBase), state.sbt.hitgroupRecordStrideInBytes * state.sbt.hitgroupRecordCount));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase), hitgroups, state.sbt.hitgroupRecordStrideInBytes * state.sbt.hitgroupRecordCount, cudaMemcpyHostToDevice));
	delete[] hitgroups;
}
void prepareLaunchData(ImageData& image, LaunchParameters& parameters, uint32_t windowWidth, uint32_t windowHeight, CudaOptixState& state, const Camera& camera, Geometry& geometry)
{
	TCHAR charbuffer[MAX_PATH]{};
	GetModuleFileName(NULL, charbuffer, MAX_PATH);
	std::filesystem::path progpath{ charbuffer };
	std::ifstream ifstr{ progpath.remove_filename() / "image_resolve.ptx", std::ios_base::binary | std::ios_base::ate };
	size_t inputSize{ static_cast<size_t>(ifstr.tellg()) };
	char* input{ new char[inputSize + 1] };
	ifstr.seekg(0);
	ifstr.read(input, inputSize);
	input[inputSize] = '\0';
	CUDA_CHECK(cuModuleLoadData(&state.imageResolveModule, input));
	CUDA_CHECK(cuModuleGetFunction(&state.resolveImageDataFunc, state.imageResolveModule, state.imageResolveFunctionName.c_str()));
	delete[] input;

	image.setColorspace(Color::RGBColorspace::sRGB);

	{
		DenselySampledSpectrum sensorSpectralCurves[]{ SpectralData::CIE::X(), SpectralData::CIE::Y(), SpectralData::CIE::Z() };
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.sensorSpectralCurvesData), sizeof(DenselySampledSpectrum) * ARRAYSIZE(sensorSpectralCurves)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.sensorSpectralCurvesData), sensorSpectralCurves, sizeof(DenselySampledSpectrum) * ARRAYSIZE(sensorSpectralCurves), cudaMemcpyHostToDevice));
	}


	std::unordered_map<SpectralData::SpectralDataType, int> loadMats{};
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
	for (int i{ 0 }; i < geometry.materialDescriptors.size(); ++i)
	{
		addMat(geometry.materialDescriptors[i].baseIOR, geometry.materialData[i].indexOfRefractSpectrumDataIndex, loadMats);
		addMat(geometry.materialDescriptors[i].baseAC, geometry.materialData[i].absorpCoefSpectrumDataIndex, loadMats);
		addMat(geometry.materialDescriptors[i].baseEmission, geometry.materialData[i].emissionSpectrumDataIndex, loadMats);
	}
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.materialData), sizeof(MaterialData) * geometry.materialData.size()));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.materialData), geometry.materialData.data(), sizeof(MaterialData) * geometry.materialData.size(), cudaMemcpyHostToDevice));
	DenselySampledSpectrum* spectrums{ new DenselySampledSpectrum[loadMats.size()]};
	for (const auto& [specType, index] : loadMats)
	spectrums[index] = SpectralData::loadSpectrum(specType);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.spectralData), sizeof(DenselySampledSpectrum) * loadMats.size()));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.spectralData), spectrums, sizeof(DenselySampledSpectrum) * loadMats.size(), cudaMemcpyHostToDevice));
	delete[] spectrums;


	LaunchParameters launchParameters{
		.filmWidth = windowWidth,
		.filmHeight = windowHeight,
		.invFilmWidth = static_cast<float>(1.0 / windowWidth),
		.invFilmHeight = static_cast<float>(1.0 / windowHeight),
		.maxPathDepth = 4, //Change
		.samplingState = { .offset = 0, .count = 1 }, //Change
		.renderingData = image.getRenderingData(),
		.camU = camera.u,
		.camV = camera.v,
		.camW = camera.w,
		.camPerspectiveScaleW = static_cast<float>(glm::tan((glm::radians(45.0) * 0.5))) * (static_cast<float>(windowHeight) / static_cast<float>(windowWidth)),
		.camPerspectiveScaleH = static_cast<float>(glm::tan((glm::radians(45.0) * 0.5))),
		.illuminantSpectralDistributionIndex = geometry.materialData[3].emissionSpectrumDataIndex, //Change
		.diskLightPosition = geometry.diskLight.pos - camera.pos, //Change
		.diskLightRadius = geometry.diskLight.radius, //Change
		.diskFrame = geometry.diskLight.frame, //Change
		.diskNormal = geometry.diskLight.normal, //Change
		.diskArea = geometry.diskLight.area, //Change
		.lightScale = geometry.diskLight.scale, //Change
		.diskSurfacePDF = 1.0f / geometry.diskLight.area, //Change
		.materials = state.materialData,
		.spectrums = state.spectralData,
		.sensorSpectralCurveA = state.sensorSpectralCurvesData + sizeof(DenselySampledSpectrum) * 0,
		.sensorSpectralCurveB = state.sensorSpectralCurvesData + sizeof(DenselySampledSpectrum) * 1,
		.sensorSpectralCurveC = state.sensorSpectralCurvesData + sizeof(DenselySampledSpectrum) * 2,
		.traversable = state.iasBuffer };

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.lpBuffer), sizeof(LaunchParameters)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.lpBuffer), &launchParameters, sizeof(LaunchParameters), cudaMemcpyHostToDevice));
}

int main(int argc, char** argv)
{
	// TODO:
	// Setting up default rendering values
	// Initialization of the rendering interface
		// Setting up OptiX, CUDA, OpenGL infrastructure
	// Callback folder
	// Window resizing
	// Quasi-random sequence
	// Camera interface in pt kernel
	// Put header code in compilation units
	// Sample count heuristic

	constexpr uint32_t windowWidth{ 512 };
	constexpr uint32_t windowHeight{ 512 };
	const int samplesToRender{ 512 };

	Camera camera{ {-278.0, 273.0, -800.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0} };
	ImageData image{ windowWidth, windowHeight };
	LaunchParameters launchParameters{};

	GLFWwindow* window{};
	initializeOpenGL(&window, windowWidth, windowHeight);
	image.initialize();
	initializeCUDA();
	initializeOptiX();
	stateCudaOptix.context = createOptixContext();
	createAccelerationStructures(stateCudaOptix, camera.pos, geometryData);
	createModulesProgramGroupsPipeline(stateCudaOptix);
	createSBT(stateCudaOptix, geometryData);
	prepareLaunchData(image, launchParameters, windowWidth, windowHeight, stateCudaOptix, camera, geometryData);

	cudaStream_t stream{};
	CUDA_CHECK(cudaStreamCreate(&stream));
	cudaEvent_t execEvent{};
	CUDA_CHECK(cudaEventCreateWithFlags(&execEvent, cudaEventDisableTiming));
	cudaError_t eventRes{ cudaSuccess };
	LaunchParameters::SamplingState currentSamplingState{ .offset = 0, .count = 1 };
	bool renderingIsFinished{ false };
	while (!glfwWindowShouldClose(window))
	{
		eventRes = cudaEventQuery(execEvent);
		if (eventRes == cudaSuccess && !renderingIsFinished)
		{
			image.mapData();

			CUDA_SYNC_CHECK();

			// Copy launch parameters
			CUDA_CHECK(cudaMemcpy(
						reinterpret_cast<void*>(stateCudaOptix.lpBuffer + offsetof(LaunchParameters, samplingState)),
						reinterpret_cast<void*>(&currentSamplingState),
						sizeof(currentSamplingState),
						cudaMemcpyHostToDevice));
			// Process the results of the previous launch (Resolve, Tonemap and Store)
			uint32_t winW{ windowWidth };
			uint32_t winH{ windowHeight };
			void* params[]{ &winW, &winH, &image.colorspaceTransform, &image.renderData, &image.presentDataCSurface };
			CUDA_CHECK(cuLaunchKernel(stateCudaOptix.resolveImageDataFunc,
						DISPATCH_SIZE(winW, 16), DISPATCH_SIZE(winH, 16), 1, 
						16, 16, 1,
						0, 
						0, 
						params, nullptr));

			CUDA_SYNC_CHECK();

			image.unmapData();

			CUDA_SYNC_CHECK();

			OPTIX_CHECK(optixLaunch(stateCudaOptix.pipeline, stream, stateCudaOptix.lpBuffer, sizeof(LaunchParameters), &stateCudaOptix.sbt, windowWidth, windowHeight, 1));

			currentSamplingState.offset += currentSamplingState.count;
			currentSamplingState.count = std::min(std::max(0, samplesToRender - static_cast<int>(currentSamplingState.offset)), 8);
			if (currentSamplingState.count == 0)
				renderingIsFinished = true;
		}
		else if(eventRes == cudaErrorNotReady || renderingIsFinished)
		{
			// Draw current results
		}
		else
		{
			R_ASSERT_LOG(false, "Event error result\n");
		}

		image.drawImage();
		glfwSwapBuffers(window);

		glfwPollEvents();
	}

	image.destroy();
	stateCudaOptix.cleanup();
	glfwTerminate();

	return 0;
}

static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
	std::cerr << std::format("Optix Log: [  Level - {}  ] [  Tag - {}  ]:\n\t{}\n", level, tag, message);
}
static void GLAPIENTRY openGLLogCallback(GLenum source,
										 GLenum type,
										 GLuint id,
										 GLenum severity,
										 GLsizei length,
										 const GLchar* message,
										 const void* userParam)
{
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::string srcstr{};
	switch (source)
	{
		case GL_DEBUG_SOURCE_API:             srcstr = "Source - API"; break;
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   srcstr = "Source - Window System"; break;
		case GL_DEBUG_SOURCE_SHADER_COMPILER: srcstr = "Source - Shader Compiler"; break;
		case GL_DEBUG_SOURCE_THIRD_PARTY:     srcstr = "Source - Third Party"; break;
		case GL_DEBUG_SOURCE_APPLICATION:     srcstr = "Source - Application"; break;
		case GL_DEBUG_SOURCE_OTHER:           srcstr = "Source - Other"; break;
	};

	std::string typestr{};
	switch (type)
	{
		case GL_DEBUG_TYPE_ERROR:               typestr = "Type - Error"; break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typestr = "Type - Deprecated Behaviour"; break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  typestr = "Type - Undefined Behaviour"; break;
		case GL_DEBUG_TYPE_PORTABILITY:         typestr = "Type - Portability"; break;
		case GL_DEBUG_TYPE_PERFORMANCE:         typestr = "Type - Performance"; break;
		case GL_DEBUG_TYPE_MARKER:              typestr = "Type - Marker"; break;
		case GL_DEBUG_TYPE_PUSH_GROUP:          typestr = "Type - Push Group"; break;
		case GL_DEBUG_TYPE_POP_GROUP:           typestr = "Type - Pop Group"; break;
		case GL_DEBUG_TYPE_OTHER:               typestr = "Type - Other"; break;
	};

	std::string severitystr{};
	switch (severity)
	{
		case GL_DEBUG_SEVERITY_HIGH:         severitystr = "Severity - High"; break;
		case GL_DEBUG_SEVERITY_MEDIUM:       severitystr = "Severity - Medium"; break;
		case GL_DEBUG_SEVERITY_LOW:          severitystr = "Severity - Low"; break;
		case GL_DEBUG_SEVERITY_NOTIFICATION: severitystr = "Severity - Notification"; break;
	};

	std::cerr << std::format("OpenGL Log: [  {}  ] [  {}  ] [  {}  ]:\n\t{}\n", srcstr, typestr, severitystr, message);
}
