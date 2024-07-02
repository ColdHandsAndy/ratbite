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

int main(int argc, char** argv)
{
	// TODO:
	// BxDFs
	// Setting up default rendering values
	// Camera interface in pt kernel
	// Sample count heuristic
	// Window focus (scissors)
	// GLTF loading

	initialize();

	constexpr uint32_t windowWidth{ 1280 };
	constexpr uint32_t windowHeight{ 720 };
	constexpr uint32_t renderWidth{ 256 };
	constexpr uint32_t renderHeight{ 256 };
	const int samplesToRender{ 512 };
	const int pathLength{ 4 };
	
	Window window{ windowWidth, windowHeight };
	Camera camera{ {-278.0, 273.0, -800.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0} };	
	SceneData scene{};
	RenderContext rContext{ renderWidth, renderHeight, pathLength, samplesToRender, Color::RGBColorspace::sRGB };
	RenderingInterface rInterface{ camera, rContext, scene };

	while (!glfwWindowShouldClose(window.getGLFWwindow()))
	{
		if (!rInterface.renderingIsFinished())
			rInterface.render(rContext.getColorspaceTransform());
		rInterface.drawPreview(window);

		std::cout << "Samples processed: " << rInterface.getProcessedSampleCount() << std::endl;

		glfwSwapBuffers(window.getGLFWwindow());
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}
