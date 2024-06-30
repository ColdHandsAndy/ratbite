#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cmath>

#include "../core/spectral_settings.h"
#include "../core/util_macros.h"

class SampledSpectrum
{
private:
	float spectrum[SpectralSettings::KSpectralSamplesNumber]{};

public:
	CU_DEVICE CU_INLINE SampledSpectrum() = default;
	CU_DEVICE CU_INLINE SampledSpectrum(float s)
	{
		for (float& v : spectrum)
			v = s;
	}
	CU_DEVICE CU_INLINE SampledSpectrum(const SampledSpectrum& s)
	{
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
			spectrum[i] = s.spectrum[i];
	}
	CU_DEVICE CU_INLINE SampledSpectrum& operator=(const SampledSpectrum& s)
	{
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			this->spectrum[i] = s.spectrum[i];
		}
		return *this;
	}


	CU_DEVICE CU_INLINE SampledSpectrum operator+(const SampledSpectrum& s) const
	{
		SampledSpectrum t{};
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			t.spectrum[i] = this->spectrum[i] + s.spectrum[i];
		}
		return t;
	}
	CU_DEVICE CU_INLINE SampledSpectrum operator-(const SampledSpectrum& s) const
	{
		SampledSpectrum t{};
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			t.spectrum[i] = this->spectrum[i] - s.spectrum[i];
		}
		return t;
	}
	CU_DEVICE CU_INLINE SampledSpectrum operator*(const SampledSpectrum& s) const
	{
		SampledSpectrum t{};
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			t.spectrum[i] = this->spectrum[i] * s.spectrum[i];
		}
		return t;
	}
	CU_DEVICE CU_INLINE SampledSpectrum operator/(const SampledSpectrum& s) const
	{
		SampledSpectrum t{};
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			t.spectrum[i] = this->spectrum[i] / s.spectrum[i];
		}
		return t;
	}
	CU_DEVICE CU_INLINE SampledSpectrum& operator+=(const SampledSpectrum & s)
	{
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			this->spectrum[i] += s.spectrum[i];
		}
		return *this;
	}
	CU_DEVICE CU_INLINE SampledSpectrum& operator-=(const SampledSpectrum & s)
	{
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			this->spectrum[i] -= s.spectrum[i];
		}
		return *this;
	}
	CU_DEVICE CU_INLINE SampledSpectrum& operator*=(const SampledSpectrum & s)
	{
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			this->spectrum[i] *= s.spectrum[i];
		}
		return *this;
	}
	CU_DEVICE CU_INLINE SampledSpectrum& operator/=(const SampledSpectrum& s)
	{
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			this->spectrum[i] /= s.spectrum[i];
		}
		return *this;
	}
	CU_DEVICE CU_INLINE SampledSpectrum operator*(float s) const
	{
		SampledSpectrum t{};
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			t.spectrum[i] = this->spectrum[i] * s;
		}
		return t;
	}
	CU_DEVICE CU_INLINE friend SampledSpectrum operator*(float sc, const SampledSpectrum& sp)
	{
		SampledSpectrum t{};
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			t.spectrum[i] = sp.spectrum[i] * sc;
		}
		return t;
	}
	CU_DEVICE CU_INLINE SampledSpectrum operator-() const
	{
		SampledSpectrum t{};
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			t.spectrum[i] = -(this->spectrum[i]);
		}
		return t;
	}
	CU_DEVICE CU_INLINE float max() const
	{
		float t{ this->spectrum[0] };
		for (int i{ 1 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			if (this->spectrum[i] > t)
				t = this->spectrum[i];
		}
		return t;
	}
	CU_DEVICE CU_INLINE float min() const
	{
		float t{ this->spectrum[0] };
		for (int i{ 1 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			if (this->spectrum[i] < t)
				t = this->spectrum[i];
		}
		return t;
	}
	CU_DEVICE CU_INLINE float average() const
	{
		float t{ this->spectrum[0] };
		for (int i{ 1 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			t += this->spectrum[i];
		}
		t /= SpectralSettings::KSpectralSamplesNumber;
		return t;
	}

	CU_DEVICE CU_INLINE float operator[](int i) const
	{
		return spectrum[i];
	}
	CU_DEVICE CU_INLINE float& operator[](int i)
	{
		return spectrum[i];
	}

	CU_DEVICE CU_INLINE static constexpr int getSampleCount() { return SpectralSettings::KSpectralSamplesNumber; }
};

class SampledWavelengths
{
private:
	SampledSpectrum lambda{};
	SampledSpectrum pdf{};

public:
	CU_DEVICE CU_INLINE SampledWavelengths() = default;
	CU_DEVICE CU_INLINE ~SampledWavelengths() = default;

	CU_DEVICE CU_INLINE static SampledWavelengths sampleUniform(float u, float lambdaMin = SpectralSettings::KMinSampledLambda, float lambdaMax = SpectralSettings::KMaxSampledLambda)
	{
		SampledWavelengths swl;

		swl.lambda[0] = lambdaMin * (1.0f - u) + lambdaMax * u;

		float delta{ (lambdaMax - lambdaMin) / static_cast<float>(SpectralSettings::KSpectralSamplesNumber) };
		for (int i{ 1 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			swl.lambda[i] = swl.lambda[i - 1] + delta;
			if (swl.lambda[i] > lambdaMax)
				swl.lambda[i] = lambdaMin + (swl.lambda[i] - lambdaMax);
		}

		for (int i = 0; i < SpectralSettings::KSpectralSamplesNumber; ++i)
			swl.pdf[i] = 1.0f / (lambdaMax - lambdaMin);

		return swl;
	}

	CU_DEVICE CU_INLINE static SampledWavelengths sampleVisible(float u)
	{
		SampledWavelengths swl;

		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			float up{ u + static_cast<float>(i) / SpectralSettings::KSpectralSamplesNumber };
			if (up > 1.0f)
				up -= 1.0f;

			swl.lambda[i] = sampleVisibleWavelengths(up);
			swl.pdf[i] = visibleWavelengthsPDF(swl.lambda[i]);
		}
		return swl;
	}

	CU_DEVICE CU_INLINE float operator[](int i) const { return lambda[i]; }
	CU_DEVICE CU_INLINE float& operator[](int i) { return lambda[i]; }

	CU_DEVICE CU_INLINE const SampledSpectrum& getLambda() const { return lambda; }
	CU_DEVICE CU_INLINE const SampledSpectrum& getPDF() const { return pdf; }

	CU_DEVICE CU_INLINE void terminateSecondary()
	{
		if (secondaryTerminated())
			return;

		for (int i{ 1 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
			pdf[i] = 0.0f;
		pdf[0] /= SpectralSettings::KSpectralSamplesNumber;
	}

	CU_DEVICE CU_INLINE bool secondaryTerminated() const
	{
		for (int i{ 1 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			if (pdf[i] != 0.0f)
				return false;
		}
		return true;
	}

private:
	CU_DEVICE CU_INLINE static float sampleVisibleWavelengths(float u)
	{
		return 538.0f - 138.888889f * atanhf(0.85691062f - 1.82750197f * u);
	}
	CU_DEVICE CU_INLINE static float visibleWavelengthsPDF(float lambda)
	{
		if (lambda < 360.0f || lambda > 830.0f)
			return 0.0f;
		float sqrtDiv{ coshf(0.0072f * (lambda - 538.0f)) };
		return 0.0039398042f / (sqrtDiv * sqrtDiv);
	}

};


class alignas(16) DenseSpectrum
{
private:
	float spectrum[SpectralSettings::KSampledWavelengthsNumber];

public:
	CU_DEVICE CU_INLINE DenseSpectrum() = default;
	CU_DEVICE CU_INLINE ~DenseSpectrum() = default;

	CU_DEVICE CU_INLINE SampledSpectrum sample(const SampledWavelengths& wavelengths) const
	{
		SampledSpectrum t{};
		for (int i{ 0 }; i < SpectralSettings::KSpectralSamplesNumber; ++i)
		{
			t[i] = spectrum[lroundf(wavelengths[i]) - SpectralSettings::KMinSampledLambda];
		}
		return t;
	}
};
