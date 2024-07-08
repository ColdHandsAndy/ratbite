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
		RENDER,
		DESC
	};
private:
	Mode m_renderMode{};
	int m_renderWidth{};
	int m_renderHeight{};
	float m_renderInvWidth{};
	float m_renderInvHeight{};
	int m_pathLength{};
	int m_sampleCount{};
	glm::mat3 m_colorspaceTransform{};
	// Filter
	// Tonemapper
	bool m_changesMade{ true };
	bool m_paused{ false };
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

	Mode getRenderMode() const { return m_renderMode; }
	int getRenderWidth() const { return m_renderWidth; }
	int getRenderHeight() const { return m_renderHeight; }
	float getRenderInvWidth() const { return m_renderInvWidth; }
	float getRenderInvHeight() const { return m_renderInvHeight; }
	int getPathLength() const { return m_pathLength; }
	int getSampleCount() const { return m_sampleCount; }
	bool changesMade() const { return m_changesMade; }
	bool paused() const { return m_paused; }

	const glm::mat3& getColorspaceTransform() const { return m_colorspaceTransform; }

	void setRenderMode(Mode m) { m_renderMode = m; m_changesMade = true; }
	void setRenderWidth(int w) { m_renderWidth = w; float iw{ 1.0f / w }; m_renderInvWidth = iw; m_changesMade = true; }
	void setRenderHeight(int h) { m_renderHeight = h; float ih{ 1.0f / h }; m_renderInvHeight = ih; m_changesMade = true; }
	void setPathLength(int pl) { m_pathLength = pl; m_changesMade = true; }
	void setSampleCount(int sc) { m_sampleCount = sc; m_changesMade = true; }
	void setPause(bool pause) { m_paused = pause; }
private:
	void acceptChanges() { m_changesMade = false; }

	friend class RenderingInterface;
};
