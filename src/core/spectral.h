#pragma once

#include <cstdint>
#include <array>

#include "../core/spectral_settings.h"
#include "../core/debug_macros.h"

class PiecewiseLinearSpectrum
{
private:
    float* m_lambda{};
    float* m_values{};
    uint32_t m_count{ 0 };

public:
    PiecewiseLinearSpectrum(const float* lambda, const float* values, uint32_t count, bool normalize);
    PiecewiseLinearSpectrum(const float* interleavedData, const uint32_t count, bool normalize);
    PiecewiseLinearSpectrum() = delete;
    ~PiecewiseLinearSpectrum();

    float sample(float lambda) const;

    void scale(float s);
    float max() const;
    float min() const;
};

class alignas(16) DenselySampledSpectrum
{
private:
    float m_spectrum[SpectralSettings::KSampledWavelengthsNumber];

public:
    DenselySampledSpectrum(const float* values, int count, int offset = 0);
    DenselySampledSpectrum(const PiecewiseLinearSpectrum& srcSpec);
    DenselySampledSpectrum() = default;
    ~DenselySampledSpectrum() = default;

    void fill(const float* values, int count, int offset = 0);
    void fill(const PiecewiseLinearSpectrum& srcSpec);
    void scale(float s);

    float max() const;
    float min() const;

    float sample(float lambda) const;

    static float innerProduct(const DenselySampledSpectrum& s0, const DenselySampledSpectrum& s1);
};

namespace SpectralData
{
    enum class SpectralDataType
    {
		NONE = 0,

        D_GLASS_BK7_IOR,
        D_GLASS_BAF10_IOR,
        D_GLASS_FK51A_IOR,
        D_GLASS_LASF9_IOR,
        D_GLASS_F5_IOR,
        D_GLASS_F10_IOR,
        D_GLASS_F11_IOR,
        C_METAL_AG_IOR,
        C_METAL_AL_IOR,
        C_METAL_AU_IOR,
        C_METAL_CU_IOR,
        C_METAL_CUZN_IOR,
        C_METAL_MGO_IOR,
        C_METAL_TIO2_IOR,
        C_METAL_AG_AC,
        C_METAL_AL_AC,
        C_METAL_AU_AC,
        C_METAL_CU_AC,
        C_METAL_CUZN_AC,
        C_METAL_MGO_AC,
        C_METAL_TIO2_AC,
        ILLUM_D65,
        ILLUM_F1,
        ILLUM_F2,
        ILLUM_F3,
        ILLUM_F4,
        ILLUM_F5,
        ILLUM_F6,
        ILLUM_F7,
        ILLUM_F8,
        ILLUM_F9,
        ILLUM_F10,
        ILLUM_F11,
        ILLUM_F12,

        DESC
    };

	static constexpr std::array dielectricSpectraNames{ std::to_array<const char*>({
		"Glass BK7",
		"Glass BAF10",
		"Glass FK51A",
		"Glass LASF9",
		"Glass F5",
		"Glass F10",
		"Glass F11",
		}) };
	static constexpr std::array dielectricIORSpectraTypes{ std::to_array<SpectralDataType>({
		SpectralDataType::D_GLASS_BK7_IOR,
		SpectralDataType::D_GLASS_BAF10_IOR,
		SpectralDataType::D_GLASS_FK51A_IOR,
		SpectralDataType::D_GLASS_LASF9_IOR,
		SpectralDataType::D_GLASS_F5_IOR,
		SpectralDataType::D_GLASS_F10_IOR,
		SpectralDataType::D_GLASS_F11_IOR,
		}) };
	static_assert(dielectricSpectraNames.size() == dielectricIORSpectraTypes.size());
	inline uint32_t dielectricIndexFromType(SpectralDataType type)
	{
		constexpr uint32_t typeCount{ dielectricIORSpectraTypes.size() };
		for (uint32_t i{ 0 }; i < typeCount; ++i)
		{
			if (type == dielectricIORSpectraTypes[i])
				return i;
		}
		R_ERR_LOG("Unknown type passed");
		return 0;
	}
	static constexpr std::array conductorSpectraNames{ std::to_array<const char*>({
		"Metal Ag",
		"Metal Al",
		"Metal Au",
		"Metal Cu",
		"Metal CuZn",
		"Metal MgO",
		"Metal TiO2",
		}) };
	static constexpr std::array conductorIORSpectraTypes{ std::to_array<SpectralDataType>({
		SpectralDataType::C_METAL_AG_IOR,
		SpectralDataType::C_METAL_AL_IOR,
		SpectralDataType::C_METAL_AU_IOR,
		SpectralDataType::C_METAL_CU_IOR,
		SpectralDataType::C_METAL_CUZN_IOR,
		SpectralDataType::C_METAL_MGO_IOR,
		SpectralDataType::C_METAL_TIO2_IOR,
		}) };
	static constexpr std::array conductorACSpectraTypes{ std::to_array<SpectralDataType>({
		SpectralDataType::C_METAL_AG_AC,
		SpectralDataType::C_METAL_AL_AC,
		SpectralDataType::C_METAL_AU_AC,
		SpectralDataType::C_METAL_CU_AC,
		SpectralDataType::C_METAL_CUZN_AC,
		SpectralDataType::C_METAL_MGO_AC,
		SpectralDataType::C_METAL_TIO2_AC,
		}) };
	static_assert(conductorSpectraNames.size() == conductorIORSpectraTypes.size() && conductorIORSpectraTypes.size() == conductorACSpectraTypes.size());
	inline uint32_t conductorIndexFromType(SpectralDataType type)
	{
		constexpr uint32_t typeCount{ conductorIORSpectraTypes.size() };
		for (uint32_t i{ 0 }; i < typeCount; ++i)
		{
			if (type == conductorIORSpectraTypes[i] || type == conductorACSpectraTypes[i])
				return i;
		}
		R_ERR_LOG("Unknown type passed");
		return 0;
	}
	static constexpr std::array emissionSpectraNames{ std::to_array<const char*>({
		"Illum D65",
		"Illum F1",
		"Illum F2",
		"Illum F3",
		"Illum F4",
		"Illum F5",
		"Illum F6",
		"Illum F7",
		"Illum F8",
		"Illum F9",
		"Illum F10",
		"Illum F11",
		"Illum F12",
		}) };
	static constexpr std::array emissionSpectraTypes{ std::to_array<SpectralDataType>({
		SpectralDataType::ILLUM_D65,
		SpectralDataType::ILLUM_F1,
		SpectralDataType::ILLUM_F2,
		SpectralDataType::ILLUM_F3,
		SpectralDataType::ILLUM_F4,
		SpectralDataType::ILLUM_F5,
		SpectralDataType::ILLUM_F6,
		SpectralDataType::ILLUM_F7,
		SpectralDataType::ILLUM_F8,
		SpectralDataType::ILLUM_F9,
		SpectralDataType::ILLUM_F10,
		SpectralDataType::ILLUM_F11,
		SpectralDataType::ILLUM_F12,
		}) };
	static_assert(emissionSpectraNames.size() == emissionSpectraTypes.size());
	inline uint32_t emitterIndexFromType(SpectralDataType type)
	{
		constexpr uint32_t typeCount{ emissionSpectraTypes.size() };
		for (uint32_t i{ 0 }; i < typeCount; ++i)
		{
			if (type == emissionSpectraTypes[i])
				return i;
		}
		R_ERR_LOG("Unknown type passed");
		return 0;
	}

    namespace CIE
    {
        static constexpr inline float CIE_Y_integral{ SpectralSettings::K_CIE_Y_Integral };
        const DenselySampledSpectrum& X();
        const DenselySampledSpectrum& Y();
        const DenselySampledSpectrum& Z();
        const DenselySampledSpectrum& BasisR();
        const DenselySampledSpectrum& BasisG();
        const DenselySampledSpectrum& BasisB();
    }

    DenselySampledSpectrum loadSpectrum(SpectralDataType sdt);
};
