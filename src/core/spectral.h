#pragma once

#include <cstdint>

#include "../core/spectral_settings.h"

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
        ILLUM_A,
        ILLUM_D50,
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


    namespace CIE
    {
        static constexpr inline float CIE_Y_integral{ SpectralSettings::K_CIE_Y_Integral };
        const DenselySampledSpectrum& X();
        const DenselySampledSpectrum& Y();
        const DenselySampledSpectrum& Z();
    }

    DenselySampledSpectrum loadSpectrum(SpectralDataType sdt);
};
