#include "../core/ui.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "../core/command.h"
#include "../core/window.h"
#include "../core/camera.h"
#include "../core/scene.h"
#include "../core/render_context.h"

namespace ImGuiWidgets
{
	bool Knob(const char* label, float* p_value, float v_min, float v_max, bool bounded = false)
	{
		ImGuiIO& io = ImGui::GetIO();
		ImGuiStyle& style = ImGui::GetStyle();

		float radius_outer = 20.0f;
		ImVec2 pos = ImGui::GetCursorScreenPos();
		float line_height = ImGui::GetTextLineHeight();
		ImDrawList* draw_list = ImGui::GetWindowDrawList();

		float ANGLE_MIN = 3.141592f * 1.5f;
		float ANGLE_MAX = 3.141592f * 3.5f;

		float width = max(ImGui::CalcTextSize(label).x, radius_outer*2);
		ImVec2 center = ImVec2(pos.x + width / 2, pos.y + radius_outer);
		ImGui::InvisibleButton(label, ImVec2(width, radius_outer*2 + line_height + style.ItemInnerSpacing.y));
		bool value_changed = false;
		bool is_active = ImGui::IsItemActive();
		bool is_hovered = ImGui::IsItemActive();
		if (is_active && io.MouseDelta.x != 0.0f)
		{
			float step = (v_max - v_min) / 200.0f;
			*p_value += io.MouseDelta.x * step;
			if (bounded)
			{
				if (*p_value < v_min) *p_value = v_min;
				if (*p_value > v_max) *p_value = v_max;
			}
			value_changed = true;
		}

		float t = (*p_value - v_min) / (v_max - v_min);
		float angle = ANGLE_MIN + (ANGLE_MAX - ANGLE_MIN) * t;
		float angle_cos = cosf(angle), angle_sin = sinf(angle);
		float radius_inner = radius_outer*0.40f;
		draw_list->AddCircleFilled(center, radius_outer, ImGui::GetColorU32(ImGuiCol_FrameBg), 16);
		draw_list->AddLine(ImVec2(center.x + angle_cos*radius_inner, center.y + angle_sin*radius_inner),
				ImVec2(center.x + angle_cos*(radius_outer-2), center.y + angle_sin*(radius_outer-2)),
				ImGui::GetColorU32(ImGuiCol_SliderGrabActive), 2.5f);
		draw_list->AddCircleFilled(center, radius_inner, ImGui::GetColorU32(is_active ? ImGuiCol_FrameBgActive : is_hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg), 16);
		draw_list->AddText(ImVec2(pos.x, pos.y + radius_outer * 2 + style.ItemInnerSpacing.y), ImGui::GetColorU32(ImGuiCol_Text), label);

		if (is_active || is_hovered)
		{
			ImGui::SetNextWindowPos(ImVec2(pos.x - style.WindowPadding.x, pos.y - line_height - style.ItemInnerSpacing.y - style.WindowPadding.y));
			ImGui::BeginTooltip();
			ImGui::Text("%.3f", *p_value);
			ImGui::EndTooltip();
		}

		return value_changed;
	}
}

void UI::recordDockspace(CommandBuffer& commands, Window& window, bool& openImageRenderSettings)
{
	ImGui::SetNextWindowPos(ImGui::GetMainViewport()->WorkPos);
	ImGui::SetNextWindowSize(ImVec2(window.getWidth(), window.getHeight()));

	ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoBringToFrontOnFocus |
		ImGuiWindowFlags_NoNavFocus |
		ImGuiWindowFlags_NoDocking |
		ImGuiWindowFlags_NoTitleBar |
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoCollapse |
		ImGuiWindowFlags_MenuBar;

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));  
	bool show{ ImGui::Begin("Dockspace", NULL, windowFlags) };
	ImGui::PopStyleVar();
	ImGui::DockSpace(ImGui::GetID("Dockspace"), ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
	if (show)
	{
		if (ImGui::BeginMenuBar())
		{
			ImGui::MenuItem("Render To Image", nullptr, &openImageRenderSettings);
			ImGui::EndMenuBar();
		}
	}
	ImGui::End();
}
void UI::recordImageRenderSettingsWindow(CommandBuffer& commands, RenderContext& rContext, GLuint renderResult, bool openImageRenderSettings)
{
	m_imageRenderSettingsWindowIsOpen = openImageRenderSettings ? 1 : m_imageRenderSettingsWindowIsOpen;
	if (m_imageRenderSettingsWindowIsOpen)
	{
		ImGui::Begin("Image render settings", &m_imageRenderSettingsWindowIsOpen,
				ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::SetWindowFocus();

		bool changed{ false };

		// Set width and height for potential render and keep preview proportions if needed
		static int renderWidth{};
		static int renderHeight{};
		if (openImageRenderSettings)
		{
			renderWidth = rContext.getRenderWidth();
			renderHeight = rContext.getRenderHeight();
		}
		static bool keepProportions{ true };
		static float prop{ renderWidth / static_cast<float>(renderHeight) };
		if (ImGui::Checkbox("Keep proportions", &keepProportions))
			prop = renderWidth / static_cast<float>(renderHeight);
		ImGui::PushItemWidth(ImGui::GetFontSize() * 6);
		changed = ImGui::InputInt("Image width", &renderWidth, 0);
		ImGui::PopItemWidth();
		if (changed && keepProportions)
			renderHeight = renderWidth / prop;
		renderWidth = std::max(1, std::min(renderWidth, 16384));
		ImGui::PushItemWidth(ImGui::GetFontSize() * 6);
		changed = ImGui::InputInt("Image height", &renderHeight, 0);
		ImGui::PopItemWidth();
		if (changed && keepProportions)
			renderWidth = renderHeight * prop;
		renderHeight = std::max(1, std::min(renderHeight, 16384));

		// Draw a small preview to see how render would look like
		ImDrawList* drawList{ ImGui::GetWindowDrawList() };
		const ImVec2 cursorPos{ ImGui::GetWindowPos().x + ImGui::GetCursorPos().x, ImGui::GetWindowPos().y + ImGui::GetCursorPos().y };
		float rectPixelWidth{};
		float rectPixelHeight{};
		if (renderWidth > renderHeight)
		{
			rectPixelWidth = 150.0f;
			rectPixelHeight = std::max(1.0f, rectPixelWidth * (static_cast<float>(renderHeight) / static_cast<float>(renderWidth)));
		}
		else
		{
			rectPixelHeight = 150.0f;
			rectPixelWidth = std::max(1.0f, rectPixelHeight * (static_cast<float>(renderWidth) / static_cast<float>(renderHeight)));
		}
		if (static_cast<int>(rectPixelHeight) != rContext.getRenderHeight() || static_cast<int>(rectPixelWidth) != rContext.getRenderWidth())
		{
			rContext.setRenderWidth(rectPixelWidth);
			rContext.setRenderHeight(rectPixelHeight);
			commands.pushCommand(Command{.type = CommandType::CHANGE_RENDER_RESOLUTION});
		}
		else
		{
			drawList->AddImage(reinterpret_cast<ImTextureID>(renderResult),
					ImVec2(cursorPos.x, cursorPos.y), ImVec2(cursorPos.x + rectPixelWidth, cursorPos.y + rectPixelHeight),
					ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
		}
		drawList->AddRect(cursorPos,ImVec2(cursorPos.x + rectPixelWidth, cursorPos.y + rectPixelHeight),
				ImGui::GetColorU32(ImGuiCol_Border, 1.0f));
		ImGui::Dummy(ImVec2(rectPixelWidth, rectPixelHeight));

		// Start render button
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 1.0f, 0.0f, 0.75f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
		if (ImGui::Button("Start"))
		{
			rContext.setRenderMode(RenderContext::Mode::GRADUAL);
			commands.pushCommand(Command{.type = CommandType::CHANGE_RENDER_MODE});
			rContext.setRenderWidth(renderWidth);
			rContext.setRenderHeight(renderHeight);
			commands.pushCommand(Command{ .type = CommandType::CHANGE_RENDER_RESOLUTION });
			m_imageRenderSettingsWindowIsOpen = false;
			m_imageRenderWindowIsOpen = true;
		}
		ImGui::PopStyleColor(4);

		ImGui::End();
	}
}
void UI::recordImageRenderWindow(CommandBuffer& commands, Window& window, RenderContext& rContext, GLuint renderResult, int currentSampleCount)
{
	if (m_imageRenderWindowIsOpen)
	{
		std::string title{ "Render (" + std::to_string(currentSampleCount) + " / " + std::to_string(rContext.getSampleCount()) + ')' };
		ImGui::Begin((title + "###RenderWindow").c_str(), &m_imageRenderWindowIsOpen,
				ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::SetWindowFocus();
		ImVec2 vMin{ ImGui::GetWindowContentRegionMin() };
		ImVec2 vMax{ vMin.x + static_cast<float>(rContext.getRenderWidth()),
					 vMin.y + static_cast<float>(rContext.getRenderHeight()) };
		vMin.x += ImGui::GetWindowPos().x;
		vMin.y += ImGui::GetWindowPos().y;
		vMax.x += ImGui::GetWindowPos().x;
		vMax.y += ImGui::GetWindowPos().y;
		ImGui::GetWindowDrawList()->AddImage(reinterpret_cast<ImTextureID>(renderResult), vMin, vMax, ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
		ImGui::Dummy(ImVec2(rContext.getRenderWidth(), rContext.getRenderHeight()));

		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 1.0f, 0.0f, 0.75f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
		bool save{ ImGui::Button("Save", ImVec2(rContext.getRenderWidth(), 0.0f)) };
		ImGui::PopStyleColor(4);
		if (save)
		{
			std::string savePath{ saveFilesWithFileDialogWindow(window.getGLFWwindow(), getExeDir().string().c_str(), "ptresult.png", "png") };
			if (!savePath.empty())
			{
				void* data{ malloc(rContext.getRenderWidth() * rContext.getRenderHeight() * 4) };
				GLuint fb{};
				glGenFramebuffers(1, &fb);
				glBindFramebuffer(GL_FRAMEBUFFER, fb);
				glBindTexture(GL_TEXTURE_2D, renderResult);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderResult, 0);
				glReadBuffer(GL_COLOR_ATTACHMENT0);
				glReadPixels(0, 0, rContext.getRenderWidth(), rContext.getRenderHeight(), GL_RGBA, GL_UNSIGNED_BYTE, data);
				stbi_flip_vertically_on_write(1);
				stbi_write_png(savePath.c_str(), rContext.getRenderWidth(), rContext.getRenderHeight(), 4, data, rContext.getRenderWidth() * 4);
				m_imageRenderWindowIsOpen = false;
				free(data);
				glDeleteFramebuffers(1, &fb);
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			}
		}

		ImGui::End();
		if (!m_imageRenderWindowIsOpen)
		{
			rContext.setRenderMode(RenderContext::Mode::IMMEDIATE);
			commands.pushCommand(Command{.type = CommandType::CHANGE_RENDER_MODE});
		}
	}
}
void UI::recordPreviewWindow(CommandBuffer& commands, RenderContext& rContext,
		float& renderWinWidth, float& renderWinHeight,
		GLuint renderResult, float renderScale)
{
	ImGui::Begin("Preview");

	ImVec2 vMin{ ImGui::GetWindowContentRegionMin() };
	ImVec2 vMax{ ImGui::GetWindowContentRegionMax() };
	vMin.x += ImGui::GetWindowPos().x;
	vMin.y += ImGui::GetWindowPos().y;
	vMax.x += ImGui::GetWindowPos().x;
	vMax.y += ImGui::GetWindowPos().y;
	renderWinWidth = vMax.x - vMin.x;
	renderWinHeight = vMax.y - vMin.y;

	int renderWidth{ std::max(std::min(4096, static_cast<int>(renderWinWidth * renderScale)), 1) };
	int renderHeight{ std::max(std::min(4096, static_cast<int>(renderWinHeight * renderScale)), 1) };
	if ((renderHeight != rContext.getRenderHeight() || renderWidth != rContext.getRenderWidth())
			&& !m_imageRenderWindowIsOpen && !m_imageRenderSettingsWindowIsOpen)
	{
		rContext.setRenderWidth(renderWidth);
		rContext.setRenderHeight(renderHeight);
		commands.pushCommand(Command{ .type = CommandType::CHANGE_RENDER_RESOLUTION });
	}

	if (renderWidth != 0 && renderHeight != 0 && !ImGui::IsWindowCollapsed()
			&& !m_imageRenderWindowIsOpen && !m_imageRenderSettingsWindowIsOpen)
	{
		ImGui::GetWindowDrawList()->AddImage(reinterpret_cast<ImTextureID>(renderResult), vMin, vMax, ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
		static bool startInside{ false };
		if (!startInside)
		{
			startInside = ImGui::IsMouseClicked(ImGuiMouseButton_Left)
				&& (ImGui::GetMousePos().x > vMin.x && ImGui::GetMousePos().x < vMax.x)
				&& (ImGui::GetMousePos().y > vMin.y && ImGui::GetMousePos().y < vMax.y);
		}
		m_previewWindowIsFocused = ImGui::IsWindowFocused();
		m_cursorIsDraggingOverPreviewWindow = startInside && ImGui::IsMouseDragging(ImGuiMouseButton_Left) && ImGui::IsWindowFocused();
		startInside = !ImGui::IsMouseReleased(ImGuiMouseButton_Left) && startInside;
	}

	ImGui::End();
}
void UI::recordRenderSettingsWindow(CommandBuffer& commands, Camera& camera, RenderContext& rContext, float& renderScale, float renderWinWidth, float renderWinHeight)
{
	bool changed{ false };

	ImGui::Begin("Render settings");

	changed = ImGui::SliderFloat("Preview scale", &renderScale, 0.005f, 1.0f, "%.4f");
	if (changed)
	{
		int renderWidth{ std::max(std::min(4096, static_cast<int>(renderWinWidth * renderScale)), 1) };
		int renderHeight{ std::max(std::min(4096, static_cast<int>(renderWinHeight * renderScale)), 1) };
		if ((renderHeight != rContext.getRenderHeight() || renderWidth != rContext.getRenderWidth())
				&& !m_imageRenderWindowIsOpen && !m_imageRenderSettingsWindowIsOpen)
		{
			rContext.setRenderWidth(renderWidth);
			rContext.setRenderHeight(renderHeight);
			commands.pushCommand(Command{ .type = CommandType::CHANGE_RENDER_RESOLUTION });
		}
	}

	static int sampleCount{ rContext.getSampleCount() };
	changed = ImGui::InputInt("Sample count", &sampleCount);
	sampleCount = std::max(1, std::min(1048576, sampleCount));
	if (changed)
	{
		rContext.setSampleCount(sampleCount);
		commands.pushCommand(Command{ .type = CommandType::CHANGE_SAMPLE_COUNT });
	}

	static float fieldOfView{};
	fieldOfView = static_cast<float>(glm::degrees(camera.getFieldOfView()));
	changed = ImGui::SliderFloat("Field of view", &fieldOfView, 1.0f, 179.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
	if (changed)
	{
		camera.setFieldOfView(glm::radians(fieldOfView));
		commands.pushCommand(Command{ .type = CommandType::CHANGE_CAMERA_FIELD_OF_VIEW} );
	}

	constexpr float exposureParameterization{ 20.0f };
	static float exposure{ rContext.getImageExposure() * (1.0f / exposureParameterization) };
	changed = ImGui::SliderFloat("Image exposure", &exposure, -1.0f, 1.0f, "%.5f", ImGuiSliderFlags_AlwaysClamp);
	if (changed)
	{
		rContext.setImageExposure(exposure * exposureParameterization);
		commands.pushCommand(Command{ .type = CommandType::CHANGE_IMAGE_EXPOSURE });
	}

	if (ImGui::TreeNode("Path depth"))
	{
		int pathDepth{};
		pathDepth = rContext.getMaxPathDepth();
		ImGui::Text("Max path depth");
		changed = ImGui::InputInt("###MPD", &pathDepth);
		pathDepth = std::max(1, std::min(65535, pathDepth));
		if (changed)
		{
			rContext.setMaxPathDepth(pathDepth);
			commands.pushCommand(Command{ .type = CommandType::CHANGE_PATH_DEPTH });
		}
		pathDepth = rContext.getMaxReflectedPathDepth();
		ImGui::Text("Max reflected path depth");
		changed = ImGui::InputInt("###MRPD", &pathDepth);
		pathDepth = std::max(1, std::min(65535, pathDepth));
		if (changed)
		{
			rContext.setMaxReflectedPathDepth(pathDepth);
			commands.pushCommand(Command{ .type = CommandType::CHANGE_PATH_DEPTH });
		}
		pathDepth = rContext.getMaxTransmittedPathDepth();
		ImGui::Text("Max transmitted path depth");
		changed = ImGui::InputInt("###MTPD", &pathDepth);
		pathDepth = std::max(1, std::min(65535, pathDepth));
		if (changed)
		{
			rContext.setMaxTransmittedPathDepth(pathDepth);
			commands.pushCommand(Command{ .type = CommandType::CHANGE_PATH_DEPTH });
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Depth of Field"))
	{
		static bool checkbox{ camera.depthOfFieldEnabled() };
		bool dofChanged{ false };
		changed = ImGui::Checkbox("Enabled", &checkbox);
		if (changed)
		{
			camera.setDepthOfField(checkbox);
			dofChanged = true;
		}
		if (checkbox)
		{
			static float apperture{ static_cast<float>(camera.getAperture()) };
			changed = ImGui::DragFloat("Aperture", &apperture, 0.001f, 0.001f, 100000.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
			if (changed)
			{
				camera.setAperture(apperture);
				dofChanged = true;
			}

			static float focusDistance{ static_cast<float>(camera.getFocusDistance()) };
			changed = ImGui::DragFloat("Focus distance", &focusDistance, 0.1f, 0.001f, 100000.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
			if (changed)
			{
				camera.setFocusDistance(focusDistance);
				dofChanged = true;
			}
		}
		if (dofChanged) commands.pushCommand(Command{ .type = CommandType::CHANGE_DEPTH_OF_FIELD_SETTINGS });
		ImGui::TreePop();
	}

	ImGui::End();
}
void UI::recordSceneGeneralSettings(CommandBuffer& commands, Window& window, SceneData& scene, Camera& camera, const ImVec4& infoColor)
{
	bool changed{ false };

	ImGui::SeparatorText("General");

	static float movingSpeed{ static_cast<float>(camera.getMovingSpeed()) };
	changed = ImGui::DragFloat("Moving speed", &movingSpeed, 0.5f, 0.01f, 1000.0f);
	if (changed) camera.setMovingSpeed(movingSpeed);
}
void UI::recordSceneModelsSettings(CommandBuffer& commands, Window& window, SceneData& scene, const ImVec4& infoColor)
{
	ImGui::SeparatorText("Models");

	if (ImGui::Button("Add files"))
	{
		std::vector<std::filesystem::path> filepaths{ getFilesFromFileDialogWindow(window.getGLFWwindow(), "A:/Models", "gltf,glb") };

		glm::mat4 transform{ glm::identity<glm::mat4>() };

		static CommandPayloads::Model* modelPayloads{};
		if (modelPayloads) delete[] modelPayloads;
		modelPayloads = new CommandPayloads::Model[filepaths.size()];
		for(int i{ 0 }; i < filepaths.size(); ++i)
		{
			std::filesystem::path& path{ filepaths[i] };
			(modelPayloads + i)->index = scene.loadModel(path, transform);
			commands.pushCommand(Command{.type = CommandType::ADD_MODEL, .payload = modelPayloads + i});
		}
	}

	if (ImGui::BeginChild("Models child window", ImVec2(-FLT_MIN, ImGui::GetTextLineHeightWithSpacing() * 4), ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeY))
	{
		if (scene.models.empty())
			ImGui::TextColored(infoColor, "None");

		static int selectedIndex{};
		for (int i{ 0 }; i < scene.models.size(); ++i)
		{
			auto& md{ scene.models[i] };
			if (ImGui::Button((md.name + "##" + std::to_string(i)).c_str(), ImVec2(-FLT_MIN, 0.0f)))
			{
				ImGui::OpenPopup("Model settings popup");
				selectedIndex = i;
			}
			else if (ImGui::IsItemClicked(ImGuiMouseButton_Right))
			{
				ImGui::OpenPopup("Remove model popup");
				selectedIndex = i;
			}
		}
		if (ImGui::BeginPopup("Model settings popup"))
		{
			auto& md{ scene.models[selectedIndex] };
			bool changesMade{ false };

			glm::vec3 curPos{ md.transform[3] };
			float xScale{ glm::length(md.transform[0]) };
			float yScale{ glm::length(md.transform[1]) };
			float zScale{ glm::length(md.transform[2]) };
			glm::mat3 curRot{ md.transform[0] / xScale, md.transform[1] / yScale, md.transform[2] / zScale };

			changesMade = ImGui::DragFloat3("Position", &curPos[0], 0.5f);

			ImGui::PushItemWidth(ImGui::GetFontSize() * 3 + ImGui::GetStyle().FramePadding.x * 2.0f);
			static float uniScale{ 1.0f };
			float prev{ uniScale };
			if (ImGui::DragFloat("Scale Uniform", &uniScale, 0.02f, 0.001f, 1000.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp))
			{
				xScale /= prev; yScale /= prev; zScale /= prev;
				xScale *= uniScale; yScale *= uniScale; zScale *= uniScale;
				changesMade = true;
			}
			else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
				uniScale = 1.0f;
			if (ImGui::DragFloat("Scale X", &xScale, 0.02f, 0.001f, 1000.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp))
				changesMade = true;
			ImGui::SameLine();
			if (ImGui::DragFloat("Scale Y", &yScale, 0.02f, 0.001f, 1000.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp))
				changesMade = true;
			ImGui::SameLine();
			if (ImGui::DragFloat("Scale Z", &zScale, 0.02f, 0.001f, 1000.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp))
				changesMade = true;
			ImGui::PopItemWidth();

			static glm::mat3 nonappliedRot{ curRot };
			static bool startedTurning{ false };
			static float rotAngleX{ 0.0f };
			if (ImGuiWidgets::Knob("Rotation X", &rotAngleX, 0.0f, glm::pi<float>() * 2.0f))
			{
				startedTurning = true;
				float cosTh{ std::cos(rotAngleX) };
				float sinTh{ std::sin(rotAngleX) };
				curRot = glm::mat3{
					glm::vec3{1.0f, 0.0f, 0.0f},
						glm::vec3{0.0f, cosTh, sinTh},
						glm::vec3{0.0f, -sinTh, cosTh}}
				* nonappliedRot;
				changesMade = true;
			}
			ImGui::SameLine();
			static float rotAngleY{ 0.0f };
			if (ImGuiWidgets::Knob("Rotation Y", &rotAngleY, 0.0f, glm::pi<float>() * 2.0f))
			{
				startedTurning = true;
				float cosTh{ std::cos(rotAngleY) };
				float sinTh{ std::sin(rotAngleY) };
				curRot = glm::mat3{
					glm::vec3{cosTh, 0.0f, -sinTh},
						glm::vec3{0.0f, 1.0f, 0.0f},
						glm::vec3{sinTh, 0.0f, cosTh}}
				* nonappliedRot;
				changesMade = true;
			}
			ImGui::SameLine();
			static float rotAngleZ{ 0.0f };
			if (ImGuiWidgets::Knob("Rotation Z", &rotAngleZ, 0.0f, glm::pi<float>() * 2.0f))
			{
				startedTurning = true;
				float cosTh{ std::cos(rotAngleZ) };
				float sinTh{ std::sin(rotAngleZ) };
				curRot = glm::mat3{
					glm::vec3{cosTh, sinTh, 0.0f},
						glm::vec3{-sinTh, cosTh, 0.0f},
						glm::vec3{0.0f, 0.0f, 1.0f}}
				* nonappliedRot;
				changesMade = true;
			}
			if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
			{
				rotAngleX = 0.0f;
				rotAngleY = 0.0f;
				rotAngleZ = 0.0f;
				nonappliedRot = curRot;
				startedTurning = false;
			}

			md.setNewTransform(glm::mat4x3{
				curRot[0] * xScale,
				curRot[1] * yScale,
				curRot[2] * zScale,
				curPos});

			if (changesMade)
			{
				static CommandPayloads::Model modelPayload{};
				modelPayload = { .index = static_cast<uint32_t>(selectedIndex) };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_MODEL_TRANSFORM, .payload = &modelPayload });
			}

			ImGui::EndPopup();
		}
		if (ImGui::BeginPopup("Remove model popup"))
		{
			if (ImGui::Button("Remove"))
			{
				static CommandPayloads::Model modelPayload{};
				modelPayload = { .id = scene.models[selectedIndex].id, .hadEmissiveData = scene.models[selectedIndex].hasEmissiveData() };
				scene.models.erase(scene.models.begin() + selectedIndex);
				commands.pushCommand(Command{ .type = CommandType::REMOVE_MODEL, .payload = &modelPayload });
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
	}
	ImGui::EndChild();
}
void UI::recordSceneLightsSettings(CommandBuffer& commands, SceneData& scene, const ImVec4& infoColor)
{
	bool changed{ false };

	ImGui::SeparatorText("Lights");

	if (ImGui::Button("Add sphere light"))
	{
		scene.sphereLights.emplace_back(glm::vec3{0.0f}, 1.0f, 0.1f,
				SceneData::MaterialDescriptor{.bxdf = SceneData::BxDF::PURE_CONDUCTOR,
				.baseIOR = SpectralData::SpectralDataType::C_METAL_AG_IOR,
				.baseAC = SpectralData::SpectralDataType::C_METAL_AG_AC,
				.baseEmission = SpectralData::SpectralDataType::ILLUM_D65,
				.roughness = 1.0f});
		static CommandPayloads::Light lightPayload{};
		lightPayload = { .type = LightType::SPHERE, .index = static_cast<uint32_t>(scene.sphereLights.size() - 1) };
		commands.pushCommand(Command{ .type = CommandType::ADD_LIGHT, .payload = &lightPayload });
	}
	ImGui::SameLine();
	if (ImGui::Button("Add disk light"))
	{
		scene.diskLights.emplace_back(glm::vec3{0.0f}, 1.0f, glm::vec3{0.0f, 0.0f, -1.0f}, 0.1f,
				SceneData::MaterialDescriptor{.bxdf = SceneData::BxDF::PURE_CONDUCTOR,
				.baseIOR = SpectralData::SpectralDataType::C_METAL_AG_IOR,
				.baseAC = SpectralData::SpectralDataType::C_METAL_AG_AC,
				.baseEmission = SpectralData::SpectralDataType::ILLUM_D65,
				.roughness = 1.0f});
		static CommandPayloads::Light lightPayload{};
		lightPayload = { .type = LightType::DISK, .index = static_cast<uint32_t>(scene.diskLights.size() - 1) };
		commands.pushCommand(Command{ .type = CommandType::ADD_LIGHT, .payload = &lightPayload });
	}

	if (ImGui::BeginChild("Lights child window", ImVec2(-FLT_MIN, ImGui::GetTextLineHeightWithSpacing() * 4), ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeY))
	{
		if (scene.getLightCount() == 0)
			ImGui::TextColored(infoColor, "None");

		static int selectedIndex{};
		for(int i{ 0 }; i < scene.sphereLights.size(); ++i)
		{
			char name[32]{};
			std::sprintf(name, "Sphere %d", i);
			if (ImGui::Button(name, ImVec2(-FLT_MIN, 0.0f)))
			{
				ImGui::OpenPopup("Sphere light settings popup");
				selectedIndex = i;
			}
			else if (ImGui::IsItemClicked(ImGuiMouseButton_Right))
			{
				ImGui::OpenPopup("Remove sphere light popup");
				selectedIndex = i;
			}
		}
		for(int i{ 0 }; i < scene.diskLights.size(); ++i)
		{
			char name[32]{};
			std::sprintf(name, "Disk %d", i);
			if (ImGui::Button(name, ImVec2(-FLT_MIN, 0.0f)))
			{
				ImGui::OpenPopup("Disk light settings popup");
				selectedIndex = i;
			}
			else if (ImGui::IsItemClicked(ImGuiMouseButton_Right))
			{
				ImGui::OpenPopup("Remove disk light popup");
				selectedIndex = i;
			}
		}

		if (ImGui::BeginPopup("Sphere light settings popup"))
		{
			SceneData::SphereLight& l{ scene.sphereLights[selectedIndex] };

			float v[3]{ l.getPosition().x, l.getPosition().y, l.getPosition().z };
			changed = ImGui::DragFloat3("Position", v, 0.05f);
			if (changed)
			{
				l.setPosition(glm::vec3{v[0], v[1], v[2]});
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::SPHERE, .index = static_cast<uint32_t>(selectedIndex) };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_LIGHT_POSITION, .payload = &lightPayload });
			}

			float r{ l.getRadius() };
			changed = ImGui::DragFloat("Radius", &r, 0.3f, 0.0001f, 10000.0f);
			if (changed)
			{
				l.setRadius(r);
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::SPHERE, .index = static_cast<uint32_t>(selectedIndex) };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_LIGHT_SIZE, .payload = &lightPayload });
			}

			float s{ l.getPowerScale() };
			changed = ImGui::DragFloat("Power", &s, 0.001f, 0.0f, 100.0f);
			if (changed)
			{
				l.setPowerScale(s);
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::SPHERE, .index = static_cast<uint32_t>(selectedIndex) };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_LIGHT_POWER, .payload = &lightPayload });
			}

			int32_t currentSpectrum{ static_cast<int32_t>(SpectralData::emitterIndexFromType(l.getMaterialDescriptor().baseEmission)) };
			if (ImGui::ListBox("Emission spectrum", &currentSpectrum, SpectralData::emissionSpectraNames.data(), SpectralData::emissionSpectraNames.size(), 3))
			{
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::SPHERE, .index = static_cast<uint32_t>(selectedIndex), .oldEmissionType = l.getMaterialDescriptor().baseEmission };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_LIGHT_EMISSION_SPECTRUM, .payload = &lightPayload });
				l.setEmissionSpectrum(SpectralData::emissionSpectraTypes[currentSpectrum]);
			}

			ImGui::EndPopup();
		}
		else if (ImGui::BeginPopup("Disk light settings popup"))
		{
			SceneData::DiskLight& l{ scene.diskLights[selectedIndex] };

			float v[3]{ l.getPosition().x, l.getPosition().y, l.getPosition().z };
			changed = ImGui::DragFloat3("Position", v, 0.05f);
			if (changed)
			{
				l.setPosition(glm::vec3{v[0], v[1], v[2]});
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::DISK, .index = static_cast<uint32_t>(selectedIndex) };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_LIGHT_POSITION, .payload = &lightPayload });
			}

			float r{ l.getRadius() };
			changed = ImGui::DragFloat("Radius", &r, 0.3f, 0.0001f, 10000.0f);
			if (changed)
			{
				l.setRadius(r);
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::DISK, .index = static_cast<uint32_t>(selectedIndex) };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_LIGHT_SIZE, .payload = &lightPayload });
			}

			float s{ l.getPowerScale() };
			changed = ImGui::DragFloat("Power", &s, 0.001f, 0.0f, 100.0f);
			if (changed)
			{
				l.setPowerScale(s);
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::DISK, .index = static_cast<uint32_t>(selectedIndex) };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_LIGHT_POWER, .payload = &lightPayload });
			}

			static float theta{};
			static float phi{};
			const glm::vec3& norm{ l.getNormal() };
			float xyL{ glm::length(glm::vec2(norm.x, norm.y)) };
			if (xyL > 0.0001f)
				phi = (norm.y > 0.0f ? 1.0f : -1.0f) * std::acos(norm.x / xyL);
			theta = std::acos(norm.z);
			changed = ImGui::DragFloat("Phi", &phi, 2.0f * glm::pi<float>() / 360.0f, glm::pi<float>(), glm::pi<float>());
			changed = ImGui::DragFloat("Theta", &theta, 2.0f * glm::pi<float>() / 360.0f, 0.0f, glm::pi<float>()) || changed;
			if (changed)
			{
				l.setNormal(glm::normalize(glm::vec3{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)}));
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::DISK, .index = static_cast<uint32_t>(selectedIndex) };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_LIGHT_ORIENTATION, .payload = &lightPayload });
			}

			int32_t currentSpectrum{ static_cast<int32_t>(SpectralData::emitterIndexFromType(l.getMaterialDescriptor().baseEmission)) };
			if (ImGui::ListBox("Emission spectrum", &currentSpectrum, SpectralData::emissionSpectraNames.data(), SpectralData::emissionSpectraNames.size(), 3))
			{
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::DISK, .index = static_cast<uint32_t>(selectedIndex), .oldEmissionType = l.getMaterialDescriptor().baseEmission };
				commands.pushCommand(Command{ .type = CommandType::CHANGE_LIGHT_EMISSION_SPECTRUM, .payload = &lightPayload });
				l.setEmissionSpectrum(SpectralData::emissionSpectraTypes[currentSpectrum]);
			}

			ImGui::EndPopup();
		}
		else if (ImGui::BeginPopup("Remove sphere light popup"))
		{
			changed = ImGui::Button("Remove");
			if (changed)
			{
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::SPHERE, .id = scene.sphereLights[selectedIndex].getID() };
				commands.pushCommand(Command{ .type = CommandType::REMOVE_LIGHT, .payload = &lightPayload });
				scene.sphereLights.erase(scene.sphereLights.begin() + selectedIndex);
				ImGui::CloseCurrentPopup();
			}

			ImGui::EndPopup();
		}
		else if (ImGui::BeginPopup("Remove disk light popup"))
		{
			changed = ImGui::Button("Remove");
			if (changed)
			{
				static CommandPayloads::Light lightPayload{};
				lightPayload = { .type = LightType::DISK, .id = scene.diskLights[selectedIndex].getID() };
				commands.pushCommand(Command{ .type = CommandType::REMOVE_LIGHT, .payload = &lightPayload });
				scene.diskLights.erase(scene.diskLights.begin() + selectedIndex);
				ImGui::CloseCurrentPopup();
			}

			ImGui::EndPopup();
		}
	}
	ImGui::EndChild();
}
void UI::recordSceneEnvironmentMapSettings(CommandBuffer& commands, Window& window, SceneData& scene, const ImVec4& infoColor)
{
	ImGui::SeparatorText("Environment Map");
	bool envMapPresent{ !scene.environmentMapPath.empty() };
	if (!envMapPresent)
		ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetColorU32(ImVec4(infoColor)));
	if (ImGui::SmallButton(((envMapPresent ? scene.environmentMapPath : "None") + "###EnvironmentMapButton").c_str()))
	{
		std::string resFile{ getFileFromFileDialogWindow(window.getGLFWwindow(), "A:/HDRIs", "hdr") };
		if (!resFile.empty())
		{
			scene.environmentMapPath = resFile;
			static CommandPayloads::EnvironmentMap envMapPayload{};
			envMapPayload.path = resFile;
			commands.pushCommand(Command{.type = CommandType::ADD_ENVIRONMENT_MAP, .payload = &envMapPayload});
		}
	}
	else if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && envMapPresent)
	{
		ImGui::OpenPopup("Remove environment map popup");
	}
	if (!envMapPresent)
		ImGui::PopStyleColor();

	if (ImGui::BeginPopup("Remove environment map popup"))
	{
		bool changed{ ImGui::Button("Remove") };
		if (changed)
		{
			commands.pushCommand(Command{ .type = CommandType::REMOVE_ENVIRONMENT_MAP });
			scene.environmentMapPath = {};
			ImGui::CloseCurrentPopup();
		}

		ImGui::EndPopup();
	}
}
void UI::recordCoordinateFrameWindow(Camera& camera)
{
	ImGui::Begin("XYZ");

	ImDrawList* drawList{ ImGui::GetWindowDrawList() };

	ImVec2 coordBGStartPos{ ImGui::GetCursorScreenPos() };
	float coordBGSize{ std::min(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y) };
	coordBGSize = std::max(50.0f, coordBGSize);

	drawList->AddRectFilled(coordBGStartPos, ImVec2(coordBGStartPos.x + coordBGSize, coordBGStartPos.y + coordBGSize),
			IM_COL32(0, 0, 0, 255));
	drawList->AddRect(coordBGStartPos, ImVec2(coordBGStartPos.x + coordBGSize, coordBGStartPos.y + coordBGSize), ImGui::GetColorU32(ImGuiCol_Border));

	glm::vec3 coordFrame[3]{
		{1.0f, 0.0f, 0.0f},
			{0.0f, 1.0f, 0.0f},
			{0.0f, 0.0f, 1.0f} };
	glm::mat4 view{ glm::transpose(glm::mat4{
			glm::vec4{camera.getU(), 0.0f},
			glm::vec4{camera.getV(), 1.0f},
			glm::vec4{camera.getW(), 0.0f},
			glm::vec4{glm::vec3{0.0f}, 1.0f}}) };
	glm::mat4 ortho{
		glm::vec4{0.8f, 0.0f, 0.0f, 0.0f},
			glm::vec4{0.0f, 0.8f, 0.0f, 0.0f},
			glm::vec4{0.0f, 0.0f, 0.8f, 0.0f},
			glm::vec4{0.0f, 0.0f, 0.0f, 1.0f} };
	glm::vec2 frameOrigin{ coordBGStartPos.x + coordBGSize * 0.5f,
		coordBGStartPos.y + coordBGSize * 0.5f };
	glm::vec3 coordFrameScreenSpace[3]{};
	for (int i{ 0 }; i < ARRAYSIZE(coordFrameScreenSpace); ++i)
	{
		coordFrameScreenSpace[i] = (ortho * view * glm::vec4{coordFrame[i], 1.0f}) * 0.5f + 0.5f;
		coordFrameScreenSpace[i] =
		{coordBGStartPos.x + coordFrameScreenSpace[i].x * coordBGSize,
			coordFrameScreenSpace[i].y,
			coordBGStartPos.y + coordBGSize * (1.0f - coordFrameScreenSpace[i].z)};
	}

	int drawOrder[3]{ 0, 1, 2 };
	const ImU32 colors[3]{ IM_COL32(255, 0, 0, 255), IM_COL32(0, 255, 0, 255), IM_COL32(0, 0, 255, 255) };
	const char* names[3]{ "X", "Y", "Z" };
	for (int i{ 0 }, n{ ARRAYSIZE(drawOrder) }; i < n - 1; ++i)
	{
		bool swapped{ false };
		for (int j{ 0 }; j < n - i - 1; j++)
		{
			if (coordFrameScreenSpace[drawOrder[j]].y
					<
					coordFrameScreenSpace[drawOrder[j + 1]].y)
			{
				int tmp{ drawOrder[j] };
				drawOrder[j] = drawOrder[j + 1];
				drawOrder[j + 1] = tmp;
				swapped = true;
			}
		}
		if (!swapped)
			break;
	}
	for (int i{ 0 }; i < ARRAYSIZE(drawOrder); ++i)
	{
		drawList->AddLine(ImVec2{frameOrigin.x, frameOrigin.y},
				ImVec2{coordFrameScreenSpace[drawOrder[i]].x, coordFrameScreenSpace[drawOrder[i]].z},
				colors[drawOrder[i]], 3.0f);
		drawList->AddText(ImVec2{coordFrameScreenSpace[drawOrder[i]].x, coordFrameScreenSpace[drawOrder[i]].z}, IM_COL32_WHITE, names[drawOrder[i]]);
	}

	ImGui::End();
}
void UI::recordInformationWindow(SceneData& scene, int currentSampleCount)
{
	ImGui::Begin("Information");

	ImGui::Text("Samples processed: %d", currentSampleCount);

	size_t triangleCount{ 0 };
	for (auto& md : scene.models)
		triangleCount += md.triangleCount;
	ImGui::Text("Triangle count: %llu", triangleCount);

	const char* memUnits[]{ "kB", "MB", "GB" };
	static int selectedMemUnit{ 1 };
	const char* previewVal{ memUnits[selectedMemUnit] };
	size_t freeMem{};
	size_t totalMem{};
	CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
	double unitDiv{ selectedMemUnit == 0 ? 1000.0 : (selectedMemUnit == 1 ? 1000.0 * 1000.0 : 1000.0 * 1000.0 * 1000.0) };
	double freeMemInUnits{ freeMem / unitDiv };
	double totalMemInUnits{ totalMem / unitDiv };
	int percentageConsumed{ static_cast<int>((totalMemInUnits - freeMemInUnits) / totalMemInUnits * 100.0) };
	ImGui::Text("Memory consumed: %f %s out of %f %s (%d %%)",
			totalMemInUnits - freeMemInUnits, memUnits[selectedMemUnit], totalMemInUnits, memUnits[selectedMemUnit], percentageConsumed);
	ImGui::SameLine();
	if (ImGui::BeginCombo("##", previewVal, ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_WidthFitPreview))
	{
		for (int i{ 0 }; i < ARRAYSIZE(memUnits); ++i)
		{
			const bool isSelected{ selectedMemUnit == i };
			if (ImGui::Selectable(memUnits[i], isSelected))
				selectedMemUnit = i;

			if (isSelected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	ImGui::End();
}

void UI::recordInterface(CommandBuffer& commands, Window& window, Camera& camera, RenderContext& rContext, SceneData& scene, GLuint renderResult, int currentSampleCount)
{
	startImGuiRecording();
	constexpr ImColor infoColor{ 0.99f, 0.33f, 0.29f };

	static float renderScale{ 0.33f };
	float renderWinWidth{};
	float renderWinHeight{};

	bool openImageRenderSettings{ false };
	recordDockspace(commands, window, openImageRenderSettings);

	recordImageRenderSettingsWindow(commands, rContext, renderResult, openImageRenderSettings);
	recordImageRenderWindow(commands, window, rContext, renderResult, currentSampleCount);

	if (m_imageRenderWindowIsOpen || m_imageRenderSettingsWindowIsOpen)
		ImGui::BeginDisabled();

	ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
	recordPreviewWindow(commands, rContext, renderWinWidth, renderWinHeight,
			renderResult, renderScale);
	ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
	recordRenderSettingsWindow(commands, camera, rContext, renderScale,
			renderWinWidth, renderWinHeight);

	ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
	ImGui::Begin("Scene settings");
	recordSceneGeneralSettings(commands, window, scene, camera, infoColor);
	recordSceneModelsSettings(commands, window, scene, infoColor);
	recordSceneLightsSettings(commands, scene, infoColor);
	recordSceneEnvironmentMapSettings(commands, window, scene, infoColor);
	ImGui::End();

	ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
	recordCoordinateFrameWindow(camera);
	ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
	recordInformationWindow(scene, currentSampleCount);

	if (m_imageRenderWindowIsOpen || m_imageRenderSettingsWindowIsOpen)
		ImGui::EndDisabled();
}
void UI::recordInput(CommandBuffer& commands, Window& window, Camera& camera, RenderContext& rContext)
{
	static bool first{ true };

	static double prevTime{ 0.0f };
	static double delta{};
	double newTime{ glfwGetTime() };
	delta = newTime - prevTime;
	prevTime = newTime;

	GLFWwindow* glfwwindow{ window.getGLFWwindow() };
	bool renderFocused{ m_previewWindowIsFocused };
	int state{};
	state = glfwGetKey(glfwwindow, GLFW_KEY_W);
	if (state == GLFW_PRESS && renderFocused)
		camera.addMoveDir(Camera::Direction::FORWARD);
	state = glfwGetKey(glfwwindow, GLFW_KEY_A);
	if (state == GLFW_PRESS && renderFocused)
		camera.addMoveDir(Camera::Direction::LEFT);
	state = glfwGetKey(glfwwindow, GLFW_KEY_S);
	if (state == GLFW_PRESS && renderFocused)
		camera.addMoveDir(Camera::Direction::BACKWARD);
	state = glfwGetKey(glfwwindow, GLFW_KEY_D);
	if (state == GLFW_PRESS && renderFocused)
		camera.addMoveDir(Camera::Direction::RIGHT);
	state = glfwGetKey(glfwwindow, GLFW_KEY_SPACE);
	if (state == GLFW_PRESS && renderFocused)
		camera.addMoveDir(Camera::Direction::UP);
	state = glfwGetKey(glfwwindow, GLFW_KEY_LEFT_SHIFT);
	if (state == GLFW_PRESS && renderFocused)
		camera.addMoveDir(Camera::Direction::DOWN);
	if (camera.move(delta))
		commands.pushCommand(Command{ .type = CommandType::CHANGE_CAMERA_POS });

	static int prevState{ GLFW_RELEASE };
	state = glfwGetMouseButton(glfwwindow, GLFW_MOUSE_BUTTON_LEFT);
	prevState = state;
	static double xposPrev{};
	static double yposPrev{};
	static double xpos{};
	static double ypos{};
	glfwGetCursorPos(glfwwindow, &xpos, &ypos);
	if (!first && m_cursorIsDraggingOverPreviewWindow)
	{
		double xd{ (xpos - xposPrev) };
		double yd{ (ypos - yposPrev) };
		camera.rotate(xd, yd);
		commands.pushCommand(Command{ .type = CommandType::CHANGE_CAMERA_ORIENTATION });
	}
	xposPrev = xpos;
	yposPrev = ypos;

	first = false;
}
