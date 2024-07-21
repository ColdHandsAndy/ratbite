#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cuda/std/cmath>

#include <glm/common.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

#include "../core/util_macros.h"

#define TONEMAP_ACES 1
#define TONEMAP_AGX 0
#define TONEMAP_UCHIMURA 0

CU_DEVICE CU_INLINE void gammaCorrect(glm::vec3& col)
{
	col = glm::vec3{cuda::std::pow(col.x, 1.0f / 2.2f), cuda::std::pow(col.y, 1.0f / 2.2f), cuda::std::pow(col.z, 1.0f / 2.2f)};
}
CU_DEVICE CU_INLINE void tonemap(glm::vec3& col)
{
#if TONEMAP_ACES
	float a{ 2.51f };
	float b{ 0.03f };
	float c{ 2.43f };
	float d{ 0.59f };
	float e{ 0.14f }; 
	col = glm::clamp((col * (a * col + b)) / (col * (c * col + d) + e), 0.0f, 1.0f);

	gammaCorrect(col);
#elif TONEMAP_AGX
	col = glm::clamp(col, 0.0f, 1.0f);
	const glm::mat3 agx_mat{
			0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f,
			0.0784335999999992f,  0.878468636469772f,  0.0784336f,
			0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f };
	constexpr float minEv{ -12.47393f };
	constexpr float maxEv{ 4.026069f };

	col = agx_mat * col;

	col = glm::clamp(glm::vec3{cuda::std::log2f(col.x), cuda::std::log2f(col.y), cuda::std::log2f(col.z)}, minEv, maxEv);
	col = (col - minEv) / (maxEv - minEv);

	auto contrastApprox{ [](const glm::vec3& x) -> glm::vec3 {
		glm::vec3 x2{ x * x };
		glm::vec3 x4{ x2 * x2 };

		return - 17.86f * x4 * x2 * x
			   + 78.01f * x4 * x2
			   - 126.7f * x4 * x
			   + 92.06f * x4
			   - 28.72f * x2 * x
			   + 4.361f * x2
			   - 0.1718f * x
			   + 0.002857f;
	} };
	col = contrastApprox(col);

	const glm::mat3 inv_agx_mat{
		1.1968790051201738155f, -0.052896851757456180321f, -0.052971635514443794537f,
		-0.098020881140136776078f, 1.1519031299041727435f, -0.098043450117124120312f,
		-0.099029744079720471434f, -0.098961176844843346553f, 1.1510736726411610622f };
	col = inv_agx_mat * col;
#elif TONEMAP_UCHIMURA
	const float P{ 1.0f };
    const float a{ 1.0f };
    const float m{ 0.22f };
    const float l{ 0.4f };
    const float c{ 1.33f };
    const float b{ 0.0f };
	float l0{ ((P - m) * l) / a };
	float S0{ m + l0 };
	float S1{ m + a * l0 };
	float C2{ (a * P) / (P - S1) };
	float CP{ -C2 / P };

	auto lmbd{ [&](float x) -> float {
		float w0{ 1.0f - glm::smoothstep(0.0f, m, x) };
		float w2{ glm::step(m + l0, x) };
		float w1{ 1.0f - w0 - w2 };

		float T{ m * glm::pow(x / m, c) + b };
		float S{ P - (P - S1) * exp(CP * (x - S0)) };
		float L{ m + a * (x - m) };
		float r{ T * w0 + L * w1 + S * w2 };

		return r;
	} };

	col = {lmbd(col.x), lmbd(col.y), lmbd(col.z)};
	gammaCorrect(col);
#endif
}

extern "C" __global__ void renderResolve(const uint32_t winWidth, const uint32_t winHeight, const glm::mat3 colorspaceTransform, const glm::dvec4* renderData, cudaSurfaceObject_t presentData)
{
	uint32_t x{ blockIdx.x * blockDim.x + threadIdx.x };
	uint32_t y{ blockIdx.y * blockDim.y + threadIdx.y };
	if (x >= winWidth || y >= winHeight)
		return;

	glm::dvec4 data{ renderData[y * winWidth + x] };
	double normval{ 1.0 / data.w };
	glm::vec3 normalized{ static_cast<float>(data.x * normval), 
						  static_cast<float>(data.y * normval), 
						  static_cast<float>(data.z * normval) };

	glm::vec3 color{ colorspaceTransform * normalized };

	tonemap(color);

	uchar4 res{ 
		static_cast<uint8_t>(color.x * 255.99999f),
		static_cast<uint8_t>(color.y * 255.99999f),
		static_cast<uint8_t>(color.z * 255.99999f),
		255,
	};

	constexpr uint32_t byteSize{ sizeof(res.x) * 4 };
	surf2Dwrite(res, presentData, x * byteSize, y);
	
	return;
}
