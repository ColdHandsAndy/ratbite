#include <iostream>
#include <cstdint>
#include <filesystem>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION
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

	ImColor infoColor{ 0.99f, 0.33f, 0.29f };

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
	}

	if (pause || mode == RenderContext::Mode::IMMEDIATE)
	{
		ImGui::SeparatorText("Render settings");

		static int sampleCount{ rContext.getSampleCount() };
		changed = ImGui::InputInt("Sample count", &sampleCount);
		sampleCount = std::max(1, std::min(1048576, sampleCount));
		if (changed) rContext.setSampleCount(sampleCount);

		static int pathLength{ rContext.getPathLength() };
		changed = ImGui::InputInt("Path length", &pathLength);
		pathLength = std::max(1, std::min(65535, pathLength));
		if (changed) rContext.setPathLength(pathLength);

		static int renderWidth{ rContext.getRenderWidth() };
		ImGui::DragInt("Render width", &renderWidth, 4, 1, 2048, "%d", ImGuiSliderFlags_AlwaysClamp);
		if ((renderWidth != rContext.getRenderWidth()) && !ui.keyboardIsCaptured())
			rContext.setRenderWidth(renderWidth);

		static int renderHeight{ rContext.getRenderHeight() };
		ImGui::DragInt("Render height", &renderHeight, 4, 1, 2048, "%d", ImGuiSliderFlags_AlwaysClamp);
		if ((renderHeight != rContext.getRenderHeight()) && !ui.keyboardIsCaptured())
			rContext.setRenderHeight(renderHeight);

		constexpr float exposureParameterization{ 10.0f };
		static float exposure{ rContext.getImageExposure() * (1.0f / exposureParameterization) };
		changed = ImGui::SliderFloat("Image exposure", &exposure, -1.0f, 1.0f, "%.5f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
		if (changed) rContext.setImageExposure(exposure * exposureParameterization);

		ImGui::SeparatorText("Camera settings");
		static bool checkbox{ camera.depthOfFieldEnabled() };
		changed = ImGui::Checkbox("Depth of Field", &checkbox);
		if (changed) camera.setDepthOfField(checkbox);
		if (checkbox)
		{
			static float apperture{ static_cast<float>(camera.getAperture()) };
			changed = ImGui::SliderFloat("Aperture", &apperture, 0.0f, 100.0f);
			if (changed) camera.setAperture(apperture);

			static float focusDistance{ static_cast<float>(camera.getFocusDistance()) };
			changed = ImGui::SliderFloat("Focus distance", &focusDistance, 0.01f, 1000.0f);
			if (changed) camera.setFocusDistance(focusDistance);
		}


		ImGui::SeparatorText("Lights settings");
		bool lightsChanged{ false };
		if (ImGui::TreeNode("Sphere lights"))
		{
			if (scene.sphereLights.size() == 0)
				ImGui::TextColored(infoColor, "None");

			static int sel{ -1 };
			for (int i{ 0 }; i < scene.sphereLights.size(); ++i)
			{
				char name[32]{};
				std::sprintf(name, "Sphere %d", i);
				if (ImGui::Selectable(name, sel == i))
				{
					sel = i;
				}
				if (sel == i)
				{
					SceneData::SphereLight& l{ scene.sphereLights[sel] };

					float v[3]{ l.getPosition().x, l.getPosition().y, l.getPosition().z };
					changed = ImGui::DragFloat3("Position", v, 10.0f, -10000.0f, -10000.0f);
					if (changed)
					{
						l.setPosition(glm::vec3{v[0], v[1], v[2]});
						lightsChanged = true;
					}

					float r{ l.getRadius() };
					changed = ImGui::DragFloat("Radius", &r, 5.0f, 0.0001f, 2000.0f);
					if (changed)
					{
						l.setRadius(r);
						lightsChanged = true;
					}

					float s{ l.getPowerScale() };
					changed = ImGui::DragFloat("Power", &s, 0.001f, 0.0f, 100.0f);
					if (changed)
					{
						l.setPowerScale(s);
						lightsChanged = true;
					}
				}
			}
			if (lightsChanged)
				scene.sphereLightsChanged = true;
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Disk lights"))
		{
			if (scene.diskLights.size() == 0)
				ImGui::TextColored(infoColor, "None");

			static int sel{ -1 };
			for (int i{ 0 }; i < scene.diskLights.size(); ++i)
			{
				char name[32]{};
				std::sprintf(name, "Disk %d", i);
				if (ImGui::Selectable(name, sel == i))
				{
					sel = i;
				}
				if (sel == i)
				{
					SceneData::DiskLight& l{ scene.diskLights[sel] };

					float v[3]{ l.getPosition().x, l.getPosition().y, l.getPosition().z };
					changed = ImGui::DragFloat3("Position", v, 10.0f, -10000.0f, -10000.0f);
					if (changed)
					{
						l.setPosition(glm::vec3{v[0], v[1], v[2]});
						lightsChanged = true;
					}

					float r{ l.getRadius() };
					changed = ImGui::DragFloat("Radius", &r, 5.0f, 0.0001f, 2000.0f);
					if (changed)
					{
						l.setRadius(r);
						lightsChanged = true;
					}

					float s{ l.getPowerScale() };
					changed = ImGui::DragFloat("Power", &s, 0.001f, 0.0f, 100.0f);
					if (changed)
					{
						l.setPowerScale(s);
						lightsChanged = true;
					}

					static float theta{};
					static float phi{};
					const glm::vec3& norm{ l.getNormal() };
					float xzL{ glm::length(glm::vec2(norm.x, norm.z)) };
					if (xzL > 0.0001f)
						phi = (norm.z > 0.0f ? 1.0f : -1.0f) * std::acos(norm.x / xzL);
					theta = std::acos(norm.y);
					changed = ImGui::DragFloat("Phi", &phi, 2.0f * glm::pi<float>() / 360.0f, glm::pi<float>(), glm::pi<float>());
					changed = ImGui::DragFloat("Theta", &theta, 2.0f * glm::pi<float>() / 360.0f, 0.0f, glm::pi<float>()) || changed;
					if (changed)
					{
						l.setNormal(glm::normalize(glm::vec3{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)}));
						lightsChanged = true;
					}
				}
			}
			if (lightsChanged)
				scene.diskLightsChanged = true;
			ImGui::TreePop();
		}


		ImGui::SeparatorText("Change material");

		static int currentItem{ 0 };
		if (!scene.materialDescriptors.empty())
		{
			if (ImGui::BeginCombo("Mesh", scene.materialDescriptors[currentItem].name.c_str()))
			{
				for (int i{ 0 }; i < scene.materialDescriptors.size(); ++i)
				{
					const bool selected{ currentItem == i };
					if (ImGui::Selectable(scene.materialDescriptors[i].name.c_str(), selected))
						currentItem = i;

					if (selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			SceneData::MaterialDescriptor* currentDesc{ nullptr };
			for (int i{ 0 }; i < scene.changedDescriptors.size(); ++i)
			{
				if (currentItem == scene.changedDescriptors[i].second)
					currentDesc = &(scene.changedDescriptors[i].first);
			}
			SceneData::MaterialDescriptor tempDescriptor{};
			if (currentDesc != nullptr)
				tempDescriptor = *currentDesc;
			else
				tempDescriptor = scene.materialDescriptors[currentItem];

			bool materialChanged{ false };

			int rb{};
			switch (tempDescriptor.bxdf)
			{
				case SceneData::BxDF::CONDUCTOR:
					rb = 0;
					break;
				case SceneData::BxDF::DIELECTRIC:
					rb = 1;
					break;
				default:
					R_ERR_LOG("Unknown BxDF.")
					break;
			}
			ImGui::RadioButton("Conductor", &rb, 0);
			ImGui::SameLine();
			ImGui::RadioButton("Dielectric", &rb, 1);
			SceneData::BxDF bx{};
			switch (rb)
			{
				case 0:
					bx = SceneData::BxDF::CONDUCTOR;
					break;
				case 1:
					bx = SceneData::BxDF::DIELECTRIC;
					break;
				default:
					R_ERR_LOG("Unknown output.")
					break;
			}
			if (bx != tempDescriptor.bxdf)
			{
				materialChanged = true;

				tempDescriptor.bxdf = bx;
				if (tempDescriptor.bxdf == SceneData::BxDF::CONDUCTOR)
				{
					tempDescriptor.baseIOR = SpectralData::SpectralDataType::C_METAL_AL_IOR;
					tempDescriptor.baseAC = SpectralData::SpectralDataType::C_METAL_AL_AC;
				}
				else if (tempDescriptor.bxdf == SceneData::BxDF::DIELECTRIC)
				{
					tempDescriptor.baseIOR = SpectralData::SpectralDataType::D_GLASS_F5_IOR;
					tempDescriptor.baseAC = SpectralData::SpectralDataType::NONE;
				}
			}

			if (tempDescriptor.bxdf == SceneData::BxDF::CONDUCTOR)
			{
				int32_t currentSpectrum{ static_cast<int32_t>(SpectralData::conductorIndexFromType(tempDescriptor.baseIOR)) };
				if (ImGui::ListBox("Conductor type", &currentSpectrum, SpectralData::conductorSpectraNames.data(), SpectralData::conductorSpectraNames.size(), 3))
				{
					materialChanged = true;

					tempDescriptor.baseIOR = SpectralData::conductorIORSpectraTypes[currentSpectrum];
					tempDescriptor.baseAC = SpectralData::conductorACSpectraTypes[currentSpectrum];
				}
			}
			else if (tempDescriptor.bxdf == SceneData::BxDF::DIELECTRIC)
			{
				int32_t currentSpectrum{ static_cast<int32_t>(SpectralData::dielectricIndexFromType(tempDescriptor.baseIOR)) };
				if (ImGui::ListBox("Dielectric type", &currentSpectrum, SpectralData::dielectricSpectraNames.data(), SpectralData::dielectricSpectraNames.size(), 3))
				{
					materialChanged = true;

					tempDescriptor.baseIOR = SpectralData::dielectricIORSpectraTypes[currentSpectrum];
					tempDescriptor.baseAC = SpectralData::SpectralDataType::NONE;
				}
			}

			if (tempDescriptor.baseEmission != SpectralData::SpectralDataType::NONE)
			{
				int32_t currentSpectrum{ static_cast<int32_t>(SpectralData::emitterIndexFromType(tempDescriptor.baseEmission)) };
				if (ImGui::ListBox("Emitter type", &currentSpectrum, SpectralData::emissionSpectraNames.data(), SpectralData::emissionSpectraNames.size(), 3))
				{
					materialChanged = true;

					tempDescriptor.baseEmission = SpectralData::emissionSpectraTypes[currentSpectrum];
				}
			}

			if (ImGui::DragFloat("Roughness", &tempDescriptor.roughness, 0.002f, 0.0f, 1.0f))
				materialChanged = true;

			if (materialChanged)
			{
				if (!currentDesc)
				{
					auto& cd{ scene.changedDescriptors.emplace_back() };
					cd.second = currentItem;
					cd.first = tempDescriptor;
				}
				else
				{
					*currentDesc = tempDescriptor;
				}
			}

		}
	}

	ImGui::SeparatorText("Info");
	ImGui::Text("Sample count: %d", currentSampleCount);

	ImGui::SeparatorText("##");
	ImGui::TextColored(infoColor, "Press \"H\" to hide this menu.");
	ImGui::End();

	rContext.setPause(pause);
}
void input(Window& window, UI& ui, Camera& camera, RenderContext& rContext)
{
	static bool first{ true };

	static double prevTime{ 0.0f };
	static double delta{};
	double newTime{ glfwGetTime() };
	delta = newTime - prevTime;
	prevTime = newTime;

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
	camera.move(delta);

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

	static int hKeyState{ GLFW_RELEASE };
	state = glfwGetKey(glfwwindow, GLFW_KEY_H);
	if (state == GLFW_RELEASE && hKeyState == GLFW_PRESS)
		ui.toggle();
	hKeyState = state;

	first = false;
}

int main(int argc, char** argv)
{
	// TODO:
	// Texture support (Texture loading, Ray differentials)
	// Dynamic asset loading
	// RGB base BxDF
	// Adaptive sampling

	initialize();

	constexpr uint32_t windowWidth{ 1280 };
	constexpr uint32_t windowHeight{ 720 };
	constexpr uint32_t renderWidth{ 360 };
	constexpr uint32_t renderHeight{ 150 };
	const int samplesToRender{ 4096 };
	const int pathLength{ 3 };
	
	Window window{ windowWidth, windowHeight };
	Camera camera{ {-278.0f, 273.0f, 800.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f} };
	SceneData scene{};
	RenderContext rContext{ renderWidth, renderHeight, pathLength, samplesToRender, Color::RGBColorspace::sRGB };
	RenderingInterface rInterface{ camera, rContext, scene };
	UI ui{ window.getGLFWwindow() };

	window.attachRenderingInterface(&rInterface);
	window.attachUI(&ui);

	while (!glfwWindowShouldClose(window.getGLFWwindow()))
	{
		glfwPollEvents();

		bool changesMade{ rContext.changesMade() || camera.changesMade() || scene.changesMade() };
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
