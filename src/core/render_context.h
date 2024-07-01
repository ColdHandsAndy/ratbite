#pragma once

#include <glm/mat3x3.hpp>

#include "color.h"
#include "debug_macros.h"

class RenderContext
{
private:
	int m_renderWidth{};
	int m_renderHeight{};
	float m_renderInvWidth{};
	float m_renderInvHeight{};
	int m_pathLength{};
	int m_sampleCount{};
	glm::mat3 m_colorspaceTransform{};
	// Filter
	// Tonemapper
public:
	RenderContext(int renderWidth, int renderHeight, int pathLength, int sampleCount, Color::RGBColorspace colorspace) :
		m_renderWidth{ renderWidth }, m_renderHeight{ renderHeight },
		m_renderInvWidth{ 1.0f / renderWidth }, m_renderInvHeight{ 1.0f / renderHeight },
		m_pathLength{ pathLength }, m_sampleCount{ sampleCount }, m_colorspaceTransform{ Color::generateColorspaceConversionMatrix(colorspace) }
	{
		R_ASSERT_LOG(renderWidth > 0 && renderHeight > 0, "Render resolution is nonsensical\n");
		R_ASSERT_LOG(pathLength > 0, "Path length is nonsensical\n");
		R_ASSERT_LOG(sampleCount > 0, "Sample count is nonsensical\n");
	}
	RenderContext() = delete;
	RenderContext(RenderContext&&) = delete;
	RenderContext(const RenderContext&) = delete;
	RenderContext &operator=(RenderContext&&) = delete;
	RenderContext &operator=(const RenderContext&) = delete;
	~RenderContext() = default;

	int getRenderWidth() const { return m_renderWidth; }
	int getRenderHeight() const { return m_renderHeight; }
	float getRenderInvWidth() const { return m_renderInvWidth; }
	float getRenderInvHeight() const { return m_renderInvHeight; }
	int getPathLength() const { return m_pathLength; }
	int getSampleCount() const { return m_sampleCount; }

	const glm::mat3& getColorspaceTransform() const { return m_colorspaceTransform; }
};
