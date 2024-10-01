#pragma once

#include <glm/mat3x3.hpp>

#include "color.h"
#include "debug_macros.h"

class RenderContext
{
public:
	enum class Mode : int
	{
		IMMEDIATE,
		GRADUAL,
		DESC
	};
private:
	Mode m_renderMode{ Mode::IMMEDIATE };
	int m_renderWidth{ 1 };
	int m_renderHeight{ 1 };
	float m_renderInvWidth{ 1.0f / m_renderWidth };
	float m_renderInvHeight{ 1.0f / m_renderHeight };
	int m_maxPathDepth{ 3 };
	int m_maxReflectedPathDepth{ 4 };
	int m_maxTransmittedPathDepth{ 12 };
	int m_sampleCount{ 8192 };
	glm::mat3 m_colorspaceTransform{ Color::generateColorspaceConversionMatrix(Color::RGBColorspace::sRGB) };
	float m_imageExposure{ 0.0f };

	bool m_paused{ false };
public:
	RenderContext(int renderWidth, int renderHeight, int maxPathDepth, int maxReflectedPathDepth, int maxTransmittedPathDepth, int sampleCount, Color::RGBColorspace colorspace) :
		m_renderWidth{ renderWidth }, m_renderHeight{ renderHeight },
		m_renderInvWidth{ 1.0f / renderWidth }, m_renderInvHeight{ 1.0f / renderHeight },
		m_maxPathDepth{ maxPathDepth }, m_sampleCount{ sampleCount }, m_colorspaceTransform{ Color::generateColorspaceConversionMatrix(colorspace) }
	{
		R_ASSERT_LOG(renderWidth > 0 && renderHeight > 0, "Render resolution is invalid\n");
		R_ASSERT_LOG(maxPathDepth > 0, "Path length is invalid\n");
		R_ASSERT_LOG(maxReflectedPathDepth > 0, "Path length is invalid\n");
		R_ASSERT_LOG(maxTransmittedPathDepth > 0, "Path length is invalid\n");
		R_ASSERT_LOG(sampleCount > 0, "Sample count is invalid\n");
	}
	RenderContext() = default;
	RenderContext(RenderContext&&) = delete;
	RenderContext(const RenderContext&) = delete;
	RenderContext &operator=(RenderContext&&) = delete;
	RenderContext &operator=(const RenderContext&) = delete;
	~RenderContext() = default;

	Mode getRenderMode() const { return m_renderMode; }
	int getRenderWidth() const { return m_renderWidth; }
	int getRenderHeight() const { return m_renderHeight; }
	float getRenderInvWidth() const { return m_renderInvWidth; }
	float getRenderInvHeight() const { return m_renderInvHeight; }
	int getMaxPathDepth() const { return m_maxPathDepth; }
	int getMaxReflectedPathDepth() const { return m_maxReflectedPathDepth; }
	int getMaxTransmittedPathDepth() const { return m_maxTransmittedPathDepth; }
	int getSampleCount() const { return m_sampleCount; }
	float getImageExposure() const { return m_imageExposure; }
	bool paused() const { return m_paused; }

	const glm::mat3& getColorspaceTransform() const { return m_colorspaceTransform; }

	void setRenderMode(Mode m) { m_renderMode = m; }
	void setRenderWidth(int w) { m_renderWidth = w; float iw{ 1.0f / w }; m_renderInvWidth = iw; }
	void setRenderHeight(int h) { m_renderHeight = h; float ih{ 1.0f / h }; m_renderInvHeight = ih; }
	void setMaxPathDepth(int pd) { m_maxPathDepth = pd; }
	void setMaxReflectedPathDepth(int pd) { m_maxReflectedPathDepth = pd; }
	void setMaxTransmittedPathDepth(int pd) { m_maxTransmittedPathDepth = pd; }
	void setSampleCount(int sc) { m_sampleCount = sc; }
	void setImageExposure(float exp) { m_imageExposure = exp; }
	void setPause(bool pause) { m_paused = pause; }
};
