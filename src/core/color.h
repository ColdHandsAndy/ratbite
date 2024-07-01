#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

#include "spectral.h"
#include "debug_macros.h"

namespace Color
{
	class XYZ
	{
	private:
		glm::vec3 data{};

	public:
		XYZ(float x, float y, float z) : data{ x, y, z } {};
		XYZ(const glm::vec3& xyz) : data{ xyz } {};
		XYZ(float a) : data{ a } {};
		XYZ() = default;
		~XYZ() = default;

		float& x() { return data.x; };
		float& y() { return data.y; };
		float& z() { return data.z; };

		operator glm::vec3() const { return data; }

		glm::vec2 xy()
		{
			float sum{ data.x + data.y + data.z };
			return glm::vec2{ data.x / sum, data.y / sum };
		}

		const static XYZ fromxyY(glm::vec2 xy, float Y = 1)
		{
			if (xy.y == 0.0f)
				return XYZ{ 0.0f };
			return XYZ{ xy.x * Y / xy.y, Y, (1 - xy.x - xy.y) * Y / xy.y };
		}
	};

	enum class RGBColorspace
	{
		sRGB,
		DCI_P3,
		Rec2020,
		DESC
	};

	inline XYZ spectrumToXYZ(const DenselySampledSpectrum& sp)
	{
		return glm::vec3{
			DenselySampledSpectrum::innerProduct(SpectralData::CIE::X(), sp), 
			DenselySampledSpectrum::innerProduct(SpectralData::CIE::Y(), sp),
			DenselySampledSpectrum::innerProduct(SpectralData::CIE::Z(), sp) } / SpectralData::CIE::CIE_Y_integral;
	}

	inline glm::mat3 generateColorspaceConversionMatrix(RGBColorspace cs)
	{
		glm::vec2 xyCoordsR{};
		glm::vec2 xyCoordsG{};
		glm::vec2 xyCoordsB{};
		XYZ wpPrimaries{};
		switch (cs)
		{
		case RGBColorspace::sRGB:
			xyCoordsR = glm::vec2{0.64f, 0.33f};
			xyCoordsG = glm::vec2{0.3f, 0.6f};
			xyCoordsB = glm::vec2{0.15f, 0.06f};
			wpPrimaries = spectrumToXYZ(DenselySampledSpectrum{SpectralData::loadSpectrum(SpectralData::SpectralDataType::ILLUM_D65)});
			break;
		case RGBColorspace::DCI_P3:
			xyCoordsR = glm::vec2{0.68f, 0.32f};
			xyCoordsG = glm::vec2{0.265f, 0.690f};
			xyCoordsB = glm::vec2{0.15f, 0.06f};
			wpPrimaries = spectrumToXYZ(DenselySampledSpectrum{ SpectralData::loadSpectrum(SpectralData::SpectralDataType::ILLUM_D65) });
			break;
		case RGBColorspace::Rec2020:
			xyCoordsR = glm::vec2{0.708f, 0.292f};
			xyCoordsG = glm::vec2{0.170f, 0.797f};
			xyCoordsB = glm::vec2{0.131f, 0.046f};
			wpPrimaries = spectrumToXYZ(DenselySampledSpectrum{ SpectralData::loadSpectrum(SpectralData::SpectralDataType::ILLUM_D65) });
			break;
		default:
			R_ASSERT_LOG(false, "Colorspace unknown");
			break;
		}
		glm::vec2 wpxyCoords{ wpPrimaries.xy() };
		XYZ r{ XYZ::fromxyY(xyCoordsR) };
		XYZ g{ XYZ::fromxyY(xyCoordsG) };
		XYZ b{ XYZ::fromxyY(xyCoordsB) };
		
		glm::mat3 rgb{ r.x(), r.y(), r.z(), g.x(), g.y(), g.z(), b.x(), b.y(), b.z() };
		XYZ c{ glm::inverse(rgb) * wpPrimaries };
		glm::mat3 t{ c.x(), 0.0f, 0.0f, 0.0f, c.y(), 0.0f, 0.0f, 0.0f, c.z() };
		return glm::inverse(rgb * t);
	}
}
