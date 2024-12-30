#pragma once

#include <cstdint>
#include <cmath>
#include <functional>

// Changed implemenation of
// https://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
namespace SoftwareRasterization
{
	struct SampleContext
	{
		float samplePointX{ 0.5f };
		float samplePointY{ 0.5f };
	};

	template<typename ...T>
	inline void rasterizeScanline(int leftmost, int rightmost, int scanline, const std::function<void(T...)>& rastFunction)
	{
		for (int i{ leftmost }; i <= rightmost; ++i)
		{
			rastFunction(i, scanline);
		}
	}
	template<typename ...T>
	inline void fillBottomFlatTriangle(const float* v0, const float* v1, const float* v2, const std::function<void(T...)>& rastFunction, const SampleContext& context)
	{
		const float& v0x{ v0[0] };
		const float& v0y{ v0[1] };

		float v1x{ v1[0] };
		float v1y{ v1[1] };
		float v2x{ v2[0] };
		float v2y{ v2[1] };
		if (v1x > v2x)
		{
			float tmp{};
			tmp = v1x;
			v1x = v2x;
			v2x = tmp;
			tmp = v1y;
			v1y = v2y;
			v2y = tmp;
		}

		const float invSlopeL{ (v1x - v0x) / (v0y - v1y) };
		const float invSlopeR{ (v2x - v0x) / (v0y - v2y) };

		const float toNearestSample{ v0y - (floor(v0y - context.samplePointY) + context.samplePointY) };
		float curXL{ v0x + invSlopeL * toNearestSample };
		float curXR{ v0x + invSlopeR * toNearestSample };

		const int top{ static_cast<int>(floor(v0y + (1.0f - context.samplePointY))) };
		const int bot{ static_cast<int>(floor(v1y + (1.0f - context.samplePointY))) };
		for (int scanline{ top - 1 }; scanline >= bot; --scanline)
		{
			const int leftmost{ static_cast<int>(floor(curXL + (1.0f - context.samplePointX))) };
			const int rightmost{ static_cast<int>(floor(curXR + (1.0f - context.samplePointX))) - 1 };
			rasterizeScanline(leftmost, rightmost, scanline, rastFunction);
			curXL += invSlopeL;
			curXR += invSlopeR;
		}
	}
	template<typename ...T>
	inline void fillTopFlatTriangle(const float* v0, const float* v1, const float* v2, const std::function<void(T...)>& rastFunction, const SampleContext& context)
	{
		const float& v2x{ v2[0] };
		const float& v2y{ v2[1] };

		float v0x{ v0[0] };
		float v0y{ v0[1] };
		float v1x{ v1[0] };
		float v1y{ v1[1] };
		if (v0x > v1x)
		{
			float tmp{};
			tmp = v0x;
			v0x = v1x;
			v1x = tmp;
			tmp = v0y;
			v0y = v1y;
			v1y = tmp;
		}

		const float invSlopeL{ (v0x - v2x ) / (v0y - v2y) };
		const float invSlopeR{ (v1x - v2x) / (v1y - v2y) };

		const float toNearestSample{ (ceil(v2y - context.samplePointY) + context.samplePointY) - v2y };
		float curXL{ v2x + invSlopeL * toNearestSample };
		float curXR{ v2x + invSlopeR * toNearestSample };

		const int top{ static_cast<int>(floor(v1y + (1.0f - context.samplePointY))) };
		const int bot{ static_cast<int>(floor(v2y + (1.0f - context.samplePointY))) };
		for (int scanline{ bot }; scanline < top; ++scanline)
		{
			const int leftmost{ static_cast<int>(floor(curXL + (1.0f - context.samplePointX))) };
			const int rightmost{ static_cast<int>(floor(curXR + (1.0f - context.samplePointX))) - 1 };
			rasterizeScanline(leftmost, rightmost, scanline, rastFunction);
			curXL += invSlopeL;
			curXR += invSlopeR;
		}
	}
	template<typename ...T>
	inline void drawTriangle2D(const float* v0, const float* v1, const float* v2, const std::function<void(T...)>& rastFunction, const SampleContext& context = SampleContext{})
	{
		const float* vs[3]{ v0, v1, v2 };
		if (vs[0][1] < vs[1][1]) { const float* tmp{ vs[1] }; vs[1] = vs[0]; vs[0] = tmp; }
		if (vs[1][1] < vs[2][1]) { const float* tmp{ vs[2] }; vs[2] = vs[1]; vs[1] = tmp; }
		if (vs[0][1] < vs[1][1]) { const float* tmp{ vs[1] }; vs[1] = vs[0]; vs[0] = tmp; }

		const float& v0x{ vs[0][0] };
		const float& v0y{ vs[0][1] };

		const float& v1x{ vs[1][0] };
		const float& v1y{ vs[1][1] };

		const float& v2x{ vs[2][0] };
		const float& v2y{ vs[2][1] };

		if (v1y == v2y)
		{
			fillBottomFlatTriangle(vs[0], vs[1], vs[2], rastFunction, context);
		}
		else if (v0y == v1y)
		{
			fillTopFlatTriangle(vs[0], vs[1], vs[2], rastFunction, context);
		}
		else
		{
			const float v3[]{ v0x + (v2x - v0x) * ((v1y - v0y) / (v2y - v0y)), v1y };
			fillBottomFlatTriangle(vs[0], vs[1], v3, rastFunction, context);
			fillTopFlatTriangle(vs[1], v3, vs[2], rastFunction, context);
		}
	}
}
