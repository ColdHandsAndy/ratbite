#pragma once

#include <cuda/std/cmath>
#include <glm/common.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include "../core/util.h"

namespace AABB
{
	// Bounded AABB
	struct BBox
	{
		float min[3]{};
		float max[3]{};

		CU_HOSTDEVICE CU_INLINE static BBox getDefault()
		{
			BBox res{};
			res.min[0] = INFINITY;
			res.min[1] = INFINITY;
			res.min[2] = INFINITY;
			res.max[0] = -INFINITY;
			res.max[1] = -INFINITY;
			res.max[2] = -INFINITY;
			return res;
		}
		CU_HOSTDEVICE CU_INLINE void getCenter(float& x, float& y, float& z) const
		{
			x = min[0] + (max[0] - min[0]) * 0.5f; y = min[1] + (max[1] - min[1]) * 0.5f; z = min[2] + (max[2] - min[2]) * 0.5f;
		}
		CU_HOSTDEVICE CU_INLINE void getDimensions(float& x, float& y, float& z) const
		{
			x = max[0] - min[0]; y = max[1] - min[1]; z = max[2] - min[2];
		}
		CU_HOSTDEVICE CU_INLINE float getVolume() const
		{
			float x{}, y{}, z{};
			getDimensions(x, y, z);
			return x * y * z;
		}
		CU_HOSTDEVICE CU_INLINE float getSurfaceArea() const
		{
			float x{}, y{}, z{};
			getDimensions(x, y, z);
			return x * y * 2.0f + x * z * 2.0f + y * z * 2.0f;
		}
		CU_HOSTDEVICE CU_INLINE bool isValid() const
		{
			return min[0] <= max[0] && min[1] <= max[1] && min[2] <= max[2];
		}
	};
	// Centered AABB
	struct CBox
	{
		float center[3]{};
		float extent[3]{}; // Half of the Box dimensions

		CU_HOSTDEVICE CU_INLINE static CBox getDefault()
		{
			CBox res{};
			res.center[0] = 0.0f;
			res.center[1] = 0.0f;
			res.center[2] = 0.0f;
			res.extent[0] = -INFINITY;
			res.extent[1] = -INFINITY;
			res.extent[2] = -INFINITY;
			return res;
		}
		CU_HOSTDEVICE CU_INLINE void getMin(float& x, float& y, float& z) const
		{
			x = center[0] - extent[0]; y = center[1] - extent[1]; z = center[2] - extent[2];
		}
		CU_HOSTDEVICE CU_INLINE void getMax(float& x, float& y, float& z) const
		{
			x = center[0] + extent[0]; y = center[1] + extent[1]; z = center[2] + extent[2];
		}
		CU_HOSTDEVICE CU_INLINE void getDimensions(float& x, float& y, float& z) const
		{
			x = extent[0] * 2.0f; y = extent[1] * 2.0f; z = extent[2] * 2.0f;
		}
		CU_HOSTDEVICE CU_INLINE float getVolume() const
		{
			float x{}, y{}, z{};
			getDimensions(x, y, z);
			return x * y * z;
		}
		CU_HOSTDEVICE CU_INLINE float getSurfaceArea() const
		{
			float x{}, y{}, z{};
			getDimensions(x, y, z);
			return x * y * 2.0f + x * z * 2.0f + y * z * 2.0f;
		}
	};

	CU_HOSTDEVICE CU_INLINE BBox createUnion(const BBox& a, const BBox& b)
	{
		BBox res{};
		res.min[0] = glm::min(a.min[0], b.min[0]);
		res.min[1] = glm::min(a.min[1], b.min[1]);
		res.min[2] = glm::min(a.min[2], b.min[2]);
		res.max[0] = glm::max(a.max[0], b.max[0]);
		res.max[1] = glm::max(a.max[1], b.max[1]);
		res.max[2] = glm::max(a.max[2], b.max[2]);
		return res;
	}
	CU_HOSTDEVICE CU_INLINE CBox createUnion(const CBox& a, const CBox& b)
	{
		CBox res{};
		float aMin[3]{};
		a.getMin(aMin[0], aMin[1], aMin[2]);
		float aMax[3]{};
		a.getMax(aMax[0], aMax[1], aMax[2]);
		float bMin[3]{};
		b.getMin(bMin[0], bMin[1], bMin[2]);
		float bMax[3]{};
		b.getMax(bMax[0], bMax[1], bMax[2]);
		float resMin[3]{
			glm::min(aMin[0], bMin[0]),
			glm::min(aMin[1], bMin[1]),
			glm::min(aMin[2], bMin[2]),
		};
		float resMax[3]{
			glm::max(aMax[0], bMax[0]),
			glm::max(aMax[1], bMax[1]),
			glm::max(aMax[2], bMax[2]),
		};
		float extent[3]{
			resMax[0] - resMin[0],
			resMax[1] - resMin[1],
			resMax[2] - resMin[2],
		};
		res.extent[0] = (resMax[0] - resMin[0]) * 0.5f;
		res.extent[1] = (resMax[1] - resMin[1]) * 0.5f;
		res.extent[2] = (resMax[2] - resMin[2]) * 0.5f;
		res.center[0] = resMin[0] + res.extent[0];
		res.center[1] = resMin[1] + res.extent[1];
		res.center[2] = resMin[2] + res.extent[2];
		return res;
	}
}

namespace Octohedral
{
	CU_HOSTDEVICE CU_INLINE glm::vec2 encode(const glm::vec3& vec)
	{
		const float& x{ vec.x };
		const float& y{ vec.y };
		const float& z{ vec.z };

		glm::vec2 p{ glm::vec2{x, y} * (1.0f / (glm::abs(x) + glm::abs(y) + glm::abs(z))) };
		glm::vec2 res{
			z <= 0.0f
			?
			(glm::vec2{1.0f} - glm::vec2{glm::abs(p.y), glm::abs(p.x)}) * glm::vec2{(p.x >= 0.0f) ? +1.0f : -1.0f, (p.y >= 0.0f) ? +1.0f : -1.0f}
			:
			p
		};
		return res;
	}
	CU_HOSTDEVICE CU_INLINE glm::vec2 encode(float x, float y, float z)
	{
		return encode(glm::vec3{x, y, z});
	}
	CU_HOSTDEVICE CU_INLINE glm::vec2 encode(float* vec)
	{
		return encode(glm::vec3{vec[0], vec[1], vec[2]});
	}
	CU_HOSTDEVICE CU_INLINE glm::vec3 decode(const glm::vec2& encvec)
	{
		const float& u{ encvec.x };
		const float& v{ encvec.y };

		glm::vec3 vec;
		vec.z = 1.0f - glm::abs(u) - glm::abs(v);
		vec.x = u;
		vec.y = v;

		float t{ glm::max(0.0f, -vec.z) };

		vec.x += vec.x >= 0.0f ? -t : t;
		vec.y += vec.y >= 0.0f ? -t : t;

		return glm::normalize(vec);
	}
	CU_HOSTDEVICE CU_INLINE void decode(float encx, float ency, float& x, float& y, float& z)
	{
		glm::vec3 vec{ decode(glm::vec2{encx, ency}) };
		x = vec.x; y = vec.y; z = vec.z;
	}
	CU_HOSTDEVICE CU_INLINE void decode(const float* encvec, float* decvec)
	{
		glm::vec3 vec{ decode(glm::vec2{encvec[0], encvec[1]}) };
		decvec[0] = vec.x; decvec[1] = vec.y; decvec[2] = vec.z;
	}

	CU_HOSTDEVICE CU_INLINE uint32_t encodeU32(const glm::vec3& vec)
	{
		constexpr float uint16MaxFloat{ 65535.99f };
		glm::vec2 pre{ encode(vec) };
		uint32_t x{ static_cast<uint32_t>(glm::clamp(pre.x * 0.5f + 0.5f, 0.0f, 1.0f) * uint16MaxFloat) << 0 };
		uint32_t y{ static_cast<uint32_t>(glm::clamp(pre.y * 0.5f + 0.5f, 0.0f, 1.0f) * uint16MaxFloat) << 16 };
		uint32_t res{ y | x };
		return res;
	}
	CU_HOSTDEVICE CU_INLINE uint32_t encodeU32(float x, float y, float z)
	{
		return encodeU32(glm::vec3{x, y, z});
	}
	CU_HOSTDEVICE CU_INLINE glm::vec3 decodeU32(uint32_t encvec)
	{
		constexpr float uint16MaxFloatNormalizer{ 1.0f / 65535.0f };
		glm::vec2 pre;
		pre.x = static_cast<float>(encvec & 0xFFFF) * uint16MaxFloatNormalizer;
		pre.y = static_cast<float>(encvec >> 16) * uint16MaxFloatNormalizer;
		return decode(pre * 2.0f - 1.0f);
	}
	CU_HOSTDEVICE CU_INLINE void decodeU32(uint32_t encvec, float& x, float& y, float& z)
	{
		glm::vec3 vec{ decodeU32(encvec) };
		x = vec.x; y = vec.y; z = vec.z;
	}
}

// Implementation from:
// https://github.com/yusuketokuyoshi/VSGL/blob/9cb1ce4c1455e673e93050d0f0d9fbc37951d1ef/VSGL/Shaders/SphericalGaussian.hlsli
namespace SphericalGaussian
{
	namespace Math
	{
		CU_HOSTDEVICE CU_INLINE float mulsign(const float x, const float y)
		{
#ifdef _CUDACC_
			return __uint_as_float((__float_as_uint(y) & 0x80000000) ^ __float_as_uint(x));
#else
			uint32_t resAsUint{ (reinterpret_cast<const uint32_t&>(y) & 0x80000000) ^ reinterpret_cast<const uint32_t&>(x) };
			float res{ reinterpret_cast<const float&>(resAsUint) };
			return res;
#endif
		}

		// exp(x) - 1 with cancellation of rounding errors.
		// [Nicholas J. Higham "Accuracy and Stability of Numerical Algorithms", Section 1.14.1, p.19]
		CU_HOSTDEVICE CU_INLINE float expm1(const float x)
		{
			const float u = exp(x);

			if (u == 1.0f)
			{
				return x;
			}

			const float y = u - 1.0f;

			if (abs(x) < 1.0f)
			{
				return y * x / log(u);
			}

			return y;
		}

		// (exp(x) - 1)/x with cancellation of rounding errors.
		// [Nicholas J. Higham "Accuracy and Stability of Numerical Algorithms", Section 1.14.1, p. 19]
		CU_HOSTDEVICE CU_INLINE float expm1_over_x(const float x)
		{
			const float u = exp(x);

			if (u == 1.0f)
			{
				return 1.0f;
			}

			const float y = u - 1.0f;

			if (abs(x) < 1.0f)
			{
				return y / log(u);
			}

			return y / x;
		}

		CU_HOSTDEVICE CU_INLINE float erf(const float x)
		{
			// Early return for large |x|.
			if (abs(x) >= 4.0f)
			{
				return mulsign(1.0f, x);
			}

			// Polynomial approximation based on the approximation posted in https://forums.developer.nvidia.com/t/optimized-version-of-single-precision-error-function-erff/40977
			if (abs(x) > 1.0f)
			{
				// The maximum error is smaller than the approximation described in Abramowitz and Stegun [1964 "Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables", 7.1.26, p.299].
				const float A1 = 1.628459513f;
				const float A2 = 9.15674746e-1f;
				const float A3 = 1.54329389e-1f;
				const float A4 = -3.51759829e-2f;
				const float A5 = 5.66795561e-3f;
				const float A6 = -5.64874616e-4f;
				const float A7 = 2.58907676e-5f;
				const float a = abs(x);
				const float y = 1.0f - exp2(-(((((((A7 * a + A6) * a + A5) * a + A4) * a + A3) * a + A2) * a + A1) * a));

				return mulsign(y, x);
			}

			// The maximum error is smaller than the 6th order Taylor polynomial.
			const float A1 = 1.128379121f;
			const float A2 = -3.76123011e-1f;
			const float A3 = 1.12799220e-1f;
			const float A4 = -2.67030653e-2f;
			const float A5 = 4.90735564e-3f;
			const float A6 = -5.58853149e-4f;
			const float x2 = x * x;

			return (((((A6 * x2 + A5) * x2 + A4) * x2 + A3) * x2 + A2) * x2 + A1) * x;
		}

		// Complementary error function erfc(x) = 1 - erf(x).
		// This implementation can have a numerical error for large x.
		// TODO: Precise implementation.
		CU_HOSTDEVICE CU_INLINE float erfc(const float x)
		{
			return 1.0f - erf(x);
		}	

		constexpr float PI{ 3.14159265359f };
	}

	struct Lobe
	{
		glm::vec3 axis{};
		float sharpness{};
		float logAmplitude{};
	};

	CU_HOSTDEVICE CU_INLINE float Evaluate(const glm::vec3 dir, const glm::vec3 axis, const float sharpness, const float logAmplitude = 0.0f)
	{
		return exp(logAmplitude + sharpness * (glm::dot(dir, axis) - 1.0f));
	}

	// Exact solution of an SG integral.
	CU_HOSTDEVICE CU_INLINE float Integral(const float sharpness)
	{
		return 4.0f * Math::PI * Math::expm1_over_x(-2.0f * sharpness);
	}

	// Approximate solution for an SG integral.
	// This approximation assumes sharpness is not small.
	// Don't input sharpness smaller than 0.5 to avoid the approximate solution larger than 4pi.
	CU_HOSTDEVICE CU_INLINE float ApproxIntegral(const float sharpness)
	{
		return 2.0f * Math::PI / sharpness;
	}

	// Product of two SGs.
	CU_HOSTDEVICE CU_INLINE Lobe Product(const glm::vec3 axis1, const float sharpness1, const glm::vec3 axis2, const float sharpness2)
	{
		const glm::vec3 axis = axis1 * sharpness1 + axis2 * sharpness2;
		const float sharpness = glm::length(axis);

		// Compute logAmplitude = sharpness - sharpness1 - sharpness2 using a numerically stable form.
		const float cosine = glm::clamp(glm::dot(axis1, axis2), -1.0f, 1.0f);
		const float sharpnessMin = min(sharpness1, sharpness2);
		const float sharpnessRatio = sharpnessMin / max(max(sharpness1, sharpness2), FLT_MIN);
		const float logAmplitude = 2.0f * sharpnessMin * (cosine - 1.0f) / (sqrt(2.0f * sharpnessRatio * cosine + sharpnessRatio * sharpnessRatio + 1.0f) + sharpnessRatio + 1.0f);

		const Lobe result = { axis / max(sharpness, FLT_MIN), sharpness, logAmplitude };

		return result;
	}

	// Approximate product integral.
	// [Iwasaki et al. 2012, "Interactive Bi-scale Editing of Highly Glossy Materials"].
	CU_HOSTDEVICE CU_INLINE float ApproxProductIntegral(const Lobe sg1, const Lobe sg2)
	{
		const float sharpnessSum = sg1.sharpness + sg2.sharpness;
		const float sharpness = sg1.sharpness * sg2.sharpness / sharpnessSum;

		return 2.0f * Math::PI * Evaluate(sg1.axis, sg2.axis, sharpness, sg1.logAmplitude + sg2.logAmplitude) / sharpnessSum;
	}

	// Approximate hemispherical integral of an SG / 2pi.
	// The parameter "cosine" is the cosine of the angle between the SG axis and the pole axis of the hemisphere.
	// [Tokuyoshi 2022 "Accurate Diffuse Lighting from Spherical Gaussian Lights"]
	CU_HOSTDEVICE CU_INLINE float HemisphericalIntegralOverTwoPi(const float cosine, const float sharpness)
	{
		// This function approximately computes the integral using an interpolation between the upper hemispherical integral and lower hemispherical integral.
		// First we compute the sigmoid-form interpolation factor.
		// Instead of a logistic approximation [Meder and Bruderlin 2018 "Hemispherical Gausians for Accurate Lighting Integration"],
		// we approximate the interpolation factor using the CDF of a Gaussian (i.e. normalized error function).

		// Our fitted steepness for the CDF.
		const float A = 0.6517328826907056171791055021459f;
		const float B = 1.3418280033141287699294252888649f;
		const float C = 7.2216687798956709087860872386955f;
		const float steepness = sharpness * sqrt((0.5f * sharpness + A) / ((sharpness + B) * sharpness + C));

		// Our approximation for the normalized hemispherical integral.
		const float lerpFactor = 0.5f + 0.5f * (Math::erf(steepness * glm::clamp(cosine, -1.0f, 1.0f)) / Math::erf(steepness));

		// Interpolation between the upper hemispherical integral and lower hemispherical integral.
		// Upper hemispherical integral: 2pi*(1 - e)/sharpness.
		// Lower hemispherical integral: 2pi*e*(1 - e)/sharpness.
		// Since this function returns the integral divided by 2pi, 2pi is eliminated from the code.
		const float e = exp(-sharpness);

		return glm::mix(e, 1.0f, lerpFactor) * Math::expm1_over_x(-sharpness);
	}

	// Approximate hemispherical integral of an SG.
	// The parameter "cosine" is the cosine of the angle between the SG axis and the pole axis of the hemisphere.
	CU_HOSTDEVICE CU_INLINE float HemisphericalIntegral(const float cosine, const float sharpness)
	{
		return 2.0f * Math::PI * HemisphericalIntegralOverTwoPi(cosine, sharpness);
	}

	// Approximate hemispherical integral for a vMF distribution (i.e. normalized SG).
	// The parameter "cosine" is the cosine of the angle between the SG axis and the pole axis of the hemisphere.
	// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 4]
	CU_HOSTDEVICE CU_INLINE float VMFHemisphericalIntegral(const float cosine, const float sharpness)
	{
		// Interpolation factor [Tokuyoshi 2022].
		const float A = 0.6517328826907056171791055021459f;
		const float B = 1.3418280033141287699294252888649f;
		const float C = 7.2216687798956709087860872386955f;
		const float steepness = sharpness * sqrt((0.5f * sharpness + A) / ((sharpness + B) * sharpness + C));
		const float lerpFactor = glm::clamp(0.5f + 0.5f * (Math::erf(steepness * glm::clamp(cosine, -1.0f, 1.0f)) / Math::erf(steepness)), 0.0f, 1.0f);

		// Interpolation between upper and lower hemispherical integrals .
		const float e = exp(-sharpness);

		return glm::mix(e, 1.0f, lerpFactor) / (e + 1.0f);
	}

	// Approximate product integral of an SG and clamped cosine / pi.
	// [Tokuyoshi 2022 "Accurate Diffuse Lighting from Spherical Gaussian Lights"]
	// This implementation is slower and less accurate than SGClampedCosineProductIntegralOverPi2024.
	// Use SGClampedCosineProductIntegralOverPi2024 instead of this function.
	CU_HOSTDEVICE CU_INLINE float ClampedCosineProductIntegralOverPi2022(const Lobe sg, const glm::vec3 normal)
	{
		const float LAMBDA = 0.00084560872241480124f;
		const float ALPHA = 1182.2467339678153f;
		const Lobe prodLobe = Product(sg.axis, sg.sharpness, normal, LAMBDA);
		const float integral0 = HemisphericalIntegralOverTwoPi(glm::dot(prodLobe.axis, normal), prodLobe.sharpness) * exp(prodLobe.logAmplitude + LAMBDA);
		const float integral1 = HemisphericalIntegralOverTwoPi(glm::dot(sg.axis, normal), sg.sharpness);

		return exp(sg.logAmplitude) * max(2.0f * ALPHA * (integral0 - integral1), 0.0f);
	}

	// Approximate product integral of an SG and clamped cosine.
	// This implementation is slower and less accurate than SGClampedCosineProductIntegral2024.
	// Use SGClampedCosineProductIntegral2024 instead of this function.
	CU_HOSTDEVICE CU_INLINE float ClampedCosineProductIntegral2022(const Lobe sg, const glm::vec3 normal)
	{
		return Math::PI * ClampedCosineProductIntegralOverPi2022(sg, normal);
	}

	// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 5]
	CU_HOSTDEVICE CU_INLINE float UpperClampedCosineIntegralOverTwoPi(const float sharpness)
	{
		if (sharpness <= 0.5f)
		{
			// Taylor-series approximation for the numerical stability.
			// TODO: Derive a faster polynomial approximation.
			return (((((((-1.0f / 362880.0f)
				* sharpness + 1.0f / 40320.0f)
					* sharpness - 1.0f / 5040.0f)
						* sharpness + 1.0f / 720.0f)
							* sharpness - 1.0f / 120.0f)
								* sharpness + 1.0f / 24.0f)
									* sharpness - 1.0f / 6.0f)
										* sharpness + 0.5f;
		}

		return (expm1(-sharpness) + sharpness) / (sharpness * sharpness);
	}

	// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 6]
	CU_HOSTDEVICE CU_INLINE float LowerClampedCosineIntegralOverTwoPi(const float sharpness)
	{
		const float e = exp(-sharpness);

		if (sharpness <= 0.5f)
		{
			// Taylor-series approximation for the numerical stability.
			// TODO: Derive a faster polynomial approximation.
			return e * (((((((((1.0f / 403200.0f)
				* sharpness - 1.0f / 45360.0f)
					* sharpness + 1.0f / 5760.0f)
						* sharpness - 1.0f / 840.0f)
							* sharpness + 1.0f / 144.0f)
								* sharpness - 1.0f / 30.0f)
									* sharpness + 1.0f / 8.0f)
										* sharpness - 1.0f / 3.0f)
											* sharpness + 0.5f);
		}

		return e * (-Math::expm1(-sharpness) - sharpness * e) / (sharpness * sharpness);
	}

	// Approximate product integral of an SG and clamped cosine / pi.
	// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 7]
	CU_HOSTDEVICE CU_INLINE float ClampedCosineProductIntegralOverPi2024(const float cosine, const float sharpness)
	{
		// Fitted approximation for t(sharpness).
		const float A = 2.7360831611272558028247203765204f;
		const float B = 17.02129778174187535455530451145f;
		const float C = 4.0100826728510421403939290030394f;
		const float D = 15.219156263147210594866010069381f;
		const float E = 76.087896272360737270901154261082f;
		const float t = sharpness * sqrt(0.5f * ((sharpness + A) * sharpness + B) / (((sharpness + C) * sharpness + D) * sharpness + E));
		const float tz = t * cosine;

		// In this HLSL implementation, we roughly implement erfc(x) = 1 - erf(x) which can have a numerical error for large x.
		// Therefore, unlike the original impelemntation [Tokuyoshi et al. 2024], we clamp the lerp factor with the machine epsilon / 2 for a conservative approximation.
		// This clamping is unnecessary for languages that have a precise erfc function (e.g., C++).
		// The original implementation [Tokuyoshi et al. 2024] uses a precise erfc function and does not clamp the lerp factor.
		const float INV_SQRTPI = 0.56418958354775628694807945156077f; // = 1/sqrt(pi).
		const float CLAMPING_THRESHOLD = 0.5f * FLT_EPSILON; // Set zero if a precise erfc function is available.
		const float lerpFactor = glm::clamp(
				max(0.5f * (cosine * Math::erfc(-tz) + Math::erfc(t)) - 0.5f * INV_SQRTPI * exp(-tz * tz) * Math::expm1(t * t * (cosine * cosine - 1.0f)) / t, CLAMPING_THRESHOLD),
				0.0f, 1.0f);

		// Interpolation between lower and upper hemispherical integrals.
		const float lowerIntegral = LowerClampedCosineIntegralOverTwoPi(sharpness);
		const float upperIntegral = UpperClampedCosineIntegralOverTwoPi(sharpness);

		return 2.0f * glm::mix(lowerIntegral, upperIntegral, lerpFactor);
	}

	// Approximate product integral of an SG and clamped cosine.
	// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 7]
	CU_HOSTDEVICE CU_INLINE float ClampedCosineProductIntegral2024(const float cosine, const float sharpness)
	{
		return Math::PI * ClampedCosineProductIntegralOverPi2024(cosine, sharpness);
	}

	// Approximate the reflection lobe with an SG lobe for microfacet BRDFs.
	// [Wang et al. 2009 "All-Frequency Rendering with Dynamic, Spatially-Varying Reflectance"]
	CU_HOSTDEVICE CU_INLINE Lobe ReflectionLobe(const glm::vec3 dir, const glm::vec3 normal, const float roughness2)
	{
		// Compute SG sharpness for the NDF.
		// Unlike Wang et al. [2009], we use the following equation based on the Appendix of [Tokuyoshi and Harada 2019 "Hierarchical Russian Roulette for Vertex Connections"].
		const float sharpnessNDF = 2.0f / roughness2 - 2.0f;

		// Approximate the reflection lobe axis using the peak of the NDF (i.e., the perfectly specular reflection direction).
		const glm::vec3 axis = glm::reflect(-dir, normal);

		// Jacobian of the transformation from halfvectors to reflection vectors.
		const float jacobian = 4.0f * abs(glm::dot(dir, normal));

		// Compute sharpness for the reflection lobe.
		const float sharpness = sharpnessNDF / jacobian;

		const Lobe result = { axis, sharpness, 0.0f };
		return result;
	}

	// Estimation of vMF sharpness (i.e., SG sharpness) from the average of directions in R^3.
	// [Banerjee et al. 2005 "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"]
	CU_HOSTDEVICE CU_INLINE float VMFAxisLengthToSharpness(const float axisLength)
	{
		return axisLength * (3.0f - axisLength * axisLength) / (1.0f - axisLength * axisLength);
	}

	// Inverse of VMFAxisLengthToSharpness.
	CU_HOSTDEVICE CU_INLINE float VMFSharpnessToAxisLength(const float sharpness)
	{
		// Solve x^3 - sx^2 - 3x + s = 0, where s = sharpness.
		// For x in [0, 1] and s in [0, infty), this equation has only a single solution.
		// [Xu and Wang 2015 "Realtime Rendering Glossy to Glossy Reflections in Screen Space"]
		// We solve this cubic equation in a numerically stable manner.
		// [Peters, C. 2016. "How to solve a cubic equation, revisited". http://momentsingraphics.de/?p=105]
		const float a = sharpness / 3.0f;
		const float b = a * a * a;
		const float c = sqrt(1.0f + 3.0f * (a * a) * (1.0f + a * a));
		const float theta = atan2(c, b) / 3.0f;
		const float SQRT3 = 1.7320508075688772935274463415059f; // = sqrt(3).
		const float d = sin(theta) * SQRT3 - cos(theta);

		return (sharpness > 0x1.0p25) ? 1.0f : sqrt(1.0f + a * a) * d + a;
	}
}
