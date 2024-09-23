#include <iostream>
#include <cstdint>
#include <filesystem>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION
#include <glad/glad.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <nfd_glfw3.h>
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
#include "../core/command.h"

void initialize()
{
	R_ASSERT(glfwInit());
	CUDA_CHECK(cudaFree(0));
	CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
	OPTIX_CHECK(optixInit());
	R_ASSERT(NFD_Init() == NFD_OKAY);
}
void draw(Window* window, const RenderingInterface* rInterface, UI* ui)
{
	ui->renderInterface();

	glfwSwapBuffers(window->getGLFWwindow());
}
void cleanup(UI& ui, RenderingInterface& rInterface)
{
	ui.cleanup();
	rInterface.cleanup();
	NFD_Quit();
	glfwTerminate();
}

int main(int argc, char** argv)
{
	initialize();

	constexpr uint32_t windowWidth{ 1280 };
	constexpr uint32_t windowHeight{ 720 };

	Window window{ windowWidth, windowHeight };
	Camera camera{ {0.0f, 0.0f, 5.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f} };
	SceneData scene{};
	RenderContext rContext{};
	RenderingInterface rInterface{ camera, rContext, scene };
	UI ui{ window.getGLFWwindow() };
	CommandBuffer commands{};

	window.attachRenderingInterface(&rInterface);
	window.attachUI(&ui);

	while (!glfwWindowShouldClose(window.getGLFWwindow()))
	{
		glfwPollEvents();

		if ((!rInterface.renderingIsFinished() || !commands.empty()) && !rContext.paused())
			rInterface.render(commands, rContext, camera, scene);

		ui.recordInterface(commands, window, camera, rContext, scene, rInterface.getPreview(), rInterface.getProcessedSampleCount());
		ui.recordInput(commands, window, camera, rContext);

		draw(&window, &rInterface, &ui);
	}

	cleanup(ui, rInterface);
	return 0;
}

std::filesystem::path generateExePath()
{
	TCHAR buffer[MAX_PATH]{};
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::filesystem::path exepath{ buffer };
	return exepath;
}
std::filesystem::path getExePath()
{
	static const std::filesystem::path exepath{ generateExePath() };
	return exepath;
}
std::filesystem::path getExeDir()
{
	return getExePath().remove_filename();
}
