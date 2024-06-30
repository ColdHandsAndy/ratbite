#pragma once

namespace SpectralSettings
{
	constexpr int KSpectralSamplesNumber{ 4 };
	constexpr int KMinSampledLambda{ 360 };
	constexpr int KMaxSampledLambda{ 830 };
	constexpr int KSampledWavelengthsNumber{ KMaxSampledLambda - KMinSampledLambda + 1 };
	constexpr float K_CIE_Y_Integral{ 106.856895f };
}