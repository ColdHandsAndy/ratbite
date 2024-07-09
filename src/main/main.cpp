#include <iostream>
#include <cstdint>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#define GLM_ENABLE_EXPERIMENTAL

#include "../core/debug_macros.h"
#include "../core/color.h"
#include "../core/window.h"
#include "../core/camera.h"
#include "../core/scene.h"
#include "../core/render_context.h"
#include "../core/rendering_interface.h"
#include "../core/ui.h"

void initialize()
{
	R_ASSERT(glfwInit());
	CUDA_CHECK(cudaFree(0));
	CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
	OPTIX_CHECK(optixInit());
}
void draw(Window* window, const RenderingInterface* rInterface, UI* ui)
{
	rInterface->drawPreview(window->getWidth(), window->getHeight());

	ui->renderImGui();

	glfwSwapBuffers(window->getGLFWwindow());
}
void menu(UI& ui, Camera& camera, RenderContext& rContext, SceneData& scene, int currentSampleCount)
{
	ui.startImGuiRecording();

	bool changed{ false };
	static bool pause{ false };
	ImGui::Begin("Menu");
	static RenderContext::Mode mode{ RenderContext::Mode::IMMEDIATE };
	const char* modeNames[]{ "Immediate", "Render" };
	const char* modeName{ modeNames[static_cast<int>(mode)] };
	ImGui::SeparatorText("Mode");
	changed = ImGui::SliderInt("##", reinterpret_cast<int*>(&mode), static_cast<int>(RenderContext::Mode::IMMEDIATE), static_cast<int>(RenderContext::Mode::RENDER), modeName);
	if (changed)
	{
		rContext.setRenderMode(mode);
		if (mode == RenderContext::Mode::IMMEDIATE)
			pause = false;
		else
			pause = true;
	}

	if (mode == RenderContext::Mode::IMMEDIATE)
	{
		ImGui::SeparatorText("Render settings");

		static int sampleCount{ rContext.getSampleCount() };
		changed = ImGui::InputInt("Sample count", &sampleCount);
		sampleCount = std::max(1, std::min(65535, sampleCount));
		if (changed) rContext.setSampleCount(sampleCount);

		static int pathLength{ rContext.getPathLength() };
		changed = ImGui::InputInt("Path length", &pathLength);
		pathLength = std::max(1, std::min(65535, pathLength));
		if (changed) rContext.setPathLength(pathLength);

		static int renderWidth{ rContext.getRenderWidth() };
		ImGui::DragInt("Render width", &renderWidth, 8, 1, 2048, "%d", ImGuiSliderFlags_AlwaysClamp);
		if ((renderWidth != rContext.getRenderWidth()) && !ui.keyboardIsCaptured())
			rContext.setRenderWidth(renderWidth);

		static int renderHeight{ rContext.getRenderHeight() };
		ImGui::DragInt("Render height", &renderHeight, 8, 1, 2048, "%d", ImGuiSliderFlags_AlwaysClamp);
		if ((renderHeight != rContext.getRenderHeight()) && !ui.keyboardIsCaptured())
			rContext.setRenderHeight(renderHeight);
	}
	else
	{
		const char* names[]{ "Rendering", "Paused" };
		const float hues[]{ 0.38f, 0.14f };
		const int index{ pause ? 1 : 0 };
		ImGui::PushID(0);
		ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(ImColor::HSV(hues[index], 0.6f, 0.6f)));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>(ImColor::HSV(hues[index], 0.7f, 0.7f)));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, static_cast<ImVec4>(ImColor::HSV(hues[index], 0.8f, 0.8f)));
		pause = ImGui::Button(names[index], ImVec2{80.0f, 30.0f}) ? !pause : pause;
		ImGui::PopStyleColor(3);
		ImGui::PopID();

		if (pause)
		{
			ImGui::SeparatorText("Render settings");

			static int sampleCount{ rContext.getSampleCount() };
			changed = ImGui::InputInt("Sample count", &sampleCount);
			sampleCount = std::max(1, std::min(65535, sampleCount));
			if (changed) rContext.setSampleCount(sampleCount);

			static int pathLength{ rContext.getPathLength() };
			changed = ImGui::InputInt("Path length", &pathLength);
			pathLength = std::max(1, std::min(65535, pathLength));
			if (changed) rContext.setPathLength(pathLength);

			static int renderWidth{ rContext.getRenderWidth() };
			ImGui::DragInt("Render width", &renderWidth, 8, 1, 2048, "%d", ImGuiSliderFlags_AlwaysClamp);
			if ((renderWidth != rContext.getRenderWidth()) && !ui.keyboardIsCaptured())
				rContext.setRenderWidth(renderWidth);

			static int renderHeight{ rContext.getRenderHeight() };
			ImGui::DragInt("Render height", &renderHeight, 8, 1, 2048, "%d", ImGuiSliderFlags_AlwaysClamp);
			if ((renderHeight != rContext.getRenderHeight()) && !ui.keyboardIsCaptured())
				rContext.setRenderHeight(renderHeight);
		}
	}

	if (ImGui::TreeNodeEx("Info", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Text("Sample count: %d", currentSampleCount);

		ImGui::TreePop();
	}

	ImGui::TextColored(ImColor(0.99f, 0.33f, 0.29f), "Press \"Space\" to hide this menu.");
	ImGui::End();

	rContext.setPause(pause);
}
void input(Window& window, UI& ui, Camera& camera, RenderContext& rContext)
{
	static bool first{ true };

	GLFWwindow* glfwwindow{ window.getGLFWwindow() };
	int state{};
	state = glfwGetKey(glfwwindow, GLFW_KEY_W);
	if (state == GLFW_PRESS)
		camera.addMoveDir(Camera::Direction::FORWARD);
	state = glfwGetKey(glfwwindow, GLFW_KEY_A);
	if (state == GLFW_PRESS)
		camera.addMoveDir(Camera::Direction::LEFT);
	state = glfwGetKey(glfwwindow, GLFW_KEY_S);
	if (state == GLFW_PRESS)
		camera.addMoveDir(Camera::Direction::BACKWARD);
	state = glfwGetKey(glfwwindow, GLFW_KEY_D);
	if (state == GLFW_PRESS)
		camera.addMoveDir(Camera::Direction::RIGHT);
	state = glfwGetKey(glfwwindow, GLFW_KEY_SPACE);
	if (state == GLFW_PRESS)
		camera.addMoveDir(Camera::Direction::UP);
	state = glfwGetKey(glfwwindow, GLFW_KEY_LEFT_SHIFT);
	if (state == GLFW_PRESS)
		camera.addMoveDir(Camera::Direction::DOWN);
	camera.move();

	bool rightMouseClick{ false };
	state = glfwGetMouseButton(glfwwindow, GLFW_MOUSE_BUTTON_LEFT);
	if (state == GLFW_PRESS)
		rightMouseClick = true;
	static double xposPrev{};
	static double yposPrev{};
	static double xpos{};
	static double ypos{};
	glfwGetCursorPos(glfwwindow, &xpos, &ypos);
	if (!first && rightMouseClick && !ui.mouseIsCaptured())
	{
		double xd{ (xpos - xposPrev) };
		double yd{ (ypos - yposPrev) };
		camera.rotate(xd, yd);
	}
	xposPrev = xpos;
	yposPrev = ypos;

	first = false;
}

int main(int argc, char** argv)
{
	// TODO:
	// Immediate mode
		// Changing light and material settings
	// Camera interface in pt kernel
	// Sample count heuristic
	// OBJ loading
	// GLTF loading

	initialize();

	constexpr uint32_t windowWidth{ 1280 };
	constexpr uint32_t windowHeight{ 720 };
	constexpr uint32_t renderWidth{ 128 };
	constexpr uint32_t renderHeight{ 128 };
	const int samplesToRender{ 1024 };
	const int pathLength{ 3 };
	
	Window window{ windowWidth, windowHeight };
	Camera camera{ {-278.0f, 273.0f, -800.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f} };
	SceneData scene{};
	RenderContext rContext{ renderWidth, renderHeight, pathLength, samplesToRender, Color::RGBColorspace::sRGB };
	RenderingInterface rInterface{ camera, rContext, scene };
	UI ui{ window.getGLFWwindow() };

	window.attachRenderingInterface(&rInterface);
	window.attachUI(&ui);

	while (!glfwWindowShouldClose(window.getGLFWwindow()))
	{
		glfwPollEvents();

		bool changesMade{ rContext.changesMade() || camera.changesMade() };
		if ((!rInterface.renderingIsFinished() || changesMade) && !rContext.paused())
			rInterface.render(rContext, camera, scene, changesMade);

		menu(ui, camera, rContext, scene, rInterface.getProcessedSampleCount());

		draw(&window, &rInterface, &ui);
		
		input(window, ui, camera, rContext);
	}

	ui.cleanup();
	rInterface.cleanup();
	glfwTerminate();
	return 0;
}
