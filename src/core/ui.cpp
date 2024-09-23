#include "../core/ui.h"

#include "../core/command.h"
#include "../core/window.h"
#include "../core/camera.h"
#include "../core/scene.h"
#include "../core/render_context.h"

namespace ImGuiWidgets
{
	bool Knob(const char* label, float* p_value, float v_min, float v_max)
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
			if (*p_value < v_min) *p_value = v_min;
			if (*p_value > v_max) *p_value = v_max;
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

void UI::recordInterface(CommandBuffer& commands, Window& window, Camera& camera, RenderContext& rContext, SceneData& scene, GLuint renderResult, int currentSampleCount)
{
	startImGuiRecording();
	constexpr ImColor infoColor{ 0.99f, 0.33f, 0.29f };

	bool changed{ false };

	ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
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
		// if (ImGui::BeginMenuBar())
		// {
		// 	ImGui::EndMenuBar();
		// }
	}
	ImGui::End();     


	ImGui::Begin("Render");

	ImVec2 vMin{ ImGui::GetWindowContentRegionMin() };
	ImVec2 vMax{ ImGui::GetWindowContentRegionMax() };
	vMin.x += ImGui::GetWindowPos().x;
	vMin.y += ImGui::GetWindowPos().y;
	vMax.x += ImGui::GetWindowPos().x;
	vMax.y += ImGui::GetWindowPos().y;
	float renderWinWidth{ vMax.x - vMin.x };
	float renderWinHeight{ vMax.y - vMin.y };

	static float renderScale{ 0.2f };
	int renderWidth{ std::max(std::min(4096, static_cast<int>(renderWinWidth * renderScale)), 0) };
	int renderHeight{ std::max(std::min(4096, static_cast<int>(renderWinHeight * renderScale)), 0) };
	if (renderHeight != rContext.getRenderHeight() || renderWidth != rContext.getRenderWidth())
	{
		rContext.setRenderWidth(std::max(renderWidth, 1));
		rContext.setRenderHeight(std::max(renderHeight, 1));
		commands.pushCommand(Command{ .type = CommandType::CHANGE_RENDER_RESOLUTION });
	}

	if (renderWidth != 0 && renderHeight != 0 && !ImGui::IsWindowCollapsed())
	{
		ImGui::GetWindowDrawList()->AddImage(reinterpret_cast<ImTextureID>(renderResult), vMin, vMax, ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
		setCursorIsDraggingOverRenderWindow(ImGui::IsMouseDragging(ImGuiMouseButton_Left) && ImGui::IsWindowFocused());
	}

	ImGui::End();


	ImGui::Begin("Render settings");

	changed = ImGui::SliderFloat("Render scale", &renderScale, 0.005f, 1.0f, "%.4f");
	if (changed)
	{
		renderWidth = std::max(std::min(4096, static_cast<int>(renderWinWidth * renderScale)), 1);
		renderHeight = std::max(std::min(4096, static_cast<int>(renderWinHeight * renderScale)), 1);
		if (renderHeight != rContext.getRenderHeight() || renderWidth != rContext.getRenderWidth())
		{
			rContext.setRenderWidth(std::max(renderWidth, 1));
			rContext.setRenderHeight(std::max(renderHeight, 1));
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

	if (ImGui::TreeNode("Path depth"))
	{
		static int pathDepth{};
		pathDepth = rContext.getMaxPathDepth();
		changed = ImGui::InputInt("Max path depth", &pathDepth);
		pathDepth = std::max(1, std::min(65535, pathDepth));
		if (changed)
		{
			rContext.setMaxPathDepth(pathDepth);
			commands.pushCommand(Command{ .type = CommandType::CHANGE_PATH_LENGTH });
		}
		pathDepth = rContext.getMaxReflectedPathDepth();
		changed = ImGui::InputInt("Max reflected path depth", &pathDepth);
		pathDepth = std::max(1, std::min(65535, pathDepth));
		if (changed)
		{
			rContext.setMaxReflectedPathDepth(pathDepth);
			commands.pushCommand(Command{ .type = CommandType::CHANGE_PATH_LENGTH });
		}
		pathDepth = rContext.getMaxTransmittedPathDepth();
		changed = ImGui::InputInt("Max transmitted path depth", &pathDepth);
		pathDepth = std::max(1, std::min(65535, pathDepth));
		if (changed)
		{
			rContext.setMaxTransmittedPathDepth(pathDepth);
			commands.pushCommand(Command{ .type = CommandType::CHANGE_PATH_LENGTH });
		}
		ImGui::TreePop();
	}

	static bool checkbox{ camera.depthOfFieldEnabled() };
	bool dofChanged{ false };
	changed = ImGui::Checkbox("Depth of Field", &checkbox);
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

	constexpr float exposureParameterization{ 10.0f };
	static float exposure{ rContext.getImageExposure() * (1.0f / exposureParameterization) };
	changed = ImGui::SliderFloat("Image exposure", &exposure, -1.0f, 1.0f, "%.5f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
	if (changed)
	{
		rContext.setImageExposure(exposure * exposureParameterization);
		commands.pushCommand(Command{ .type = CommandType::CHANGE_IMAGE_EXPOSURE });
	}

	ImGui::End();


	ImGui::Begin("Scene settings");

	ImGui::SeparatorText("General");

	static float movingSpeed{ static_cast<float>(camera.getMovingSpeed()) };
	changed = ImGui::DragFloat("Moving speed", &movingSpeed, 0.5f, 0.01f, 1000.0f);
	if (changed) camera.setMovingSpeed(movingSpeed);

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
			else if (ImGui::IsMouseReleased(ImGuiMouseButton_Right) && ImGui::IsItemHovered())
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
			if (ImGui::DragFloat("Scale Uniform", &uniScale, 0.02f, 0.001f, 1000.0f))
			{
				xScale /= prev; yScale /= prev; zScale /= prev;
				xScale *= uniScale; yScale *= uniScale; zScale *= uniScale;
				changesMade = true;
			}
			if (ImGui::DragFloat("Scale X", &xScale, 0.02f, 0.001f, 1000.0f))
				changesMade = true;
			ImGui::SameLine();
			if (ImGui::DragFloat("Scale Y", &yScale, 0.02f, 0.001f, 1000.0f))
				changesMade = true;
			ImGui::SameLine();
			if (ImGui::DragFloat("Scale Z", &zScale, 0.02f, 0.001f, 1000.0f))
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

			md.transform = glm::mat4x3{
				curRot[0] * xScale,
				curRot[1] * yScale,
				curRot[2] * zScale,
				curPos};

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
				modelPayload = { .id = scene.models[selectedIndex].id };
				scene.models.erase(scene.models.begin() + selectedIndex);
				commands.pushCommand(Command{ .type = CommandType::REMOVE_MODEL, .payload = &modelPayload });
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
	}
	ImGui::EndChild();

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
		scene.diskLights.emplace_back(glm::vec3{0.0f}, 1.0f, glm::vec3{0.0f, -1.0f, 0.0f}, 0.1f,
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
		}

		if (ImGui::BeginPopup("Sphere light settings popup"))
		{
			SceneData::SphereLight& l{ scene.sphereLights[selectedIndex] };

			float v[3]{ l.getPosition().x, l.getPosition().y, l.getPosition().z };
			changed = ImGui::DragFloat3("Position", v, 0.5f);
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

			changed = ImGui::Button("Remove light", ImVec2(ImGui::GetItemRectSize().x, 0.0f));
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
		else if (ImGui::BeginPopup("Disk light settings popup"))
		{
			SceneData::DiskLight& l{ scene.diskLights[selectedIndex] };

			float v[3]{ l.getPosition().x, l.getPosition().y, l.getPosition().z };
			changed = ImGui::DragFloat3("Position", v, 0.5f);
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
			float xzL{ glm::length(glm::vec2(norm.x, norm.z)) };
			if (xzL > 0.0001f)
				phi = (norm.z > 0.0f ? 1.0f : -1.0f) * std::acos(norm.x / xzL);
			theta = std::acos(norm.y);
			changed = ImGui::DragFloat("Phi", &phi, 2.0f * glm::pi<float>() / 360.0f, glm::pi<float>(), glm::pi<float>());
			changed = ImGui::DragFloat("Theta", &theta, 2.0f * glm::pi<float>() / 360.0f, 0.0f, glm::pi<float>()) || changed;
			if (changed)
			{
				l.setNormal(glm::normalize(glm::vec3{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)}));
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

			changed = ImGui::Button("Remove light", ImVec2(ImGui::GetItemRectSize().x, 0.0f));
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

	ImGui::End();


	ImGui::Begin("Information");
	ImGui::Text("Current sample count: %d", currentSampleCount);
	ImGui::End();
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
	if (!first && cursorIsDraggingOverRenderWindow())
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
