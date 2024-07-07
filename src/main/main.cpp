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

void initialize()
{
	R_ASSERT(glfwInit());
	CUDA_CHECK(cudaFree(0));
	CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
	OPTIX_CHECK(optixInit());
}
void draw(Window* window, const RenderingInterface* rInterface)
{
	rInterface->drawPreview(window->getWidth(), window->getHeight());

	glfwSwapBuffers(window->getGLFWwindow());
}

int main(int argc, char** argv)
{
	// TODO:
	// Camera interface in pt kernel
	// Explicit cleanup
	// Sample count heuristic
	// Window focus (scissors)
	// GLTF loading

	initialize();

	constexpr uint32_t windowWidth{ 1280 };
	constexpr uint32_t windowHeight{ 720 };
	constexpr uint32_t renderWidth{ 256 };
	constexpr uint32_t renderHeight{ 256 };
	const int samplesToRender{ 1024 };
	const int pathLength{ 5 };
	
	Window window{ windowWidth, windowHeight };
	// Camera camera{ {-278.0f, 273.0f, -800.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f} };	
	Camera camera{ {-278.0f + 140.0f, 273.0f + 90.0f, -800.0f + 160.0f}, {-0.2f, -0.15f, 1.0f}, {0.0f, 1.0f, 0.0f} };	
	SceneData scene{};
	RenderContext rContext{ renderWidth, renderHeight, pathLength, samplesToRender, Color::RGBColorspace::sRGB };
	RenderingInterface rInterface{ camera, rContext, scene };

	window.attachRenderingInterface(&rInterface);

	while (!glfwWindowShouldClose(window.getGLFWwindow()))
	{
		if (!rInterface.renderingIsFinished())
			rInterface.render(rContext.getColorspaceTransform());

		std::cout << "Samples processed: " << rInterface.getProcessedSampleCount() << std::endl;

		draw(&window, &rInterface);

		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}
