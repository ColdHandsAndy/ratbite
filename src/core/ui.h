#pragma once

#include <vector>
#include <string_view>
#include <filesystem>

#include <glad/glad.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <nfd_glfw3.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include "../core/debug_macros.h"
#include "../core/command.h"
#include "../core/window.h"
#include "../core/camera.h"
#include "../core/scene.h"
#include "../core/render_context.h"

class UI
{
private:
	bool m_cursorIsDraggingOverRenderWindow{ false };
public:
	UI(GLFWwindow* window)
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io{ ImGui::GetIO() };
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		setTheme();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init();
		startImGuiRecording();
	}
	UI(UI&&) = default;
	UI(const UI&) = default;
	UI &operator=(UI&&) = default;
	UI &operator=(const UI&) = default;
	~UI() = default;

	void recordInterface(CommandBuffer& commands, Window& window, Camera& camera, RenderContext& rContext, SceneData& scene, GLuint renderResult, int currentSampleCount);
	void recordInput(CommandBuffer& commands, Window& window, Camera& camera, RenderContext& rContext);
	
	bool mouseIsCaptured() const { return ImGui::GetIO().WantCaptureMouse; }
	bool keyboardIsCaptured() const { return ImGui::GetIO().WantCaptureKeyboard; }

	void setCursorIsDraggingOverRenderWindow(bool isDraggingOver) { m_cursorIsDraggingOverRenderWindow = isDraggingOver; }
	bool cursorIsDraggingOverRenderWindow() const { return m_cursorIsDraggingOverRenderWindow; }

	std::vector<std::filesystem::path> getFilesFromFileDialogWindow(GLFWwindow* window, const char* defaultPath, const char* fileFilters)
	{
		nfdfilteritem_t filters{ .name = "Filters", .spec = fileFilters };

		const nfdpathset_t* outPaths{};
		nfdwindowhandle_t parentWindow{};
		NFD_GetNativeWindowFromGLFWWindow(window, &parentWindow);
		nfdopendialogu8args_t dialogArgs{
			.filterList = &filters,
			.filterCount = 1,
			.defaultPath = defaultPath,
			.parentWindow = parentWindow };
		nfdresult_t dialogResult{ NFD_OpenDialogMultipleU8_With(&outPaths, &dialogArgs) };

		std::vector<std::filesystem::path> resPaths{};

		if (dialogResult == NFD_OKAY)
		{
			nfdpathsetsize_t numPaths{};
			NFD_PathSet_GetCount(outPaths, &numPaths);
			resPaths.resize(numPaths);

			for (int i{ 0 }; i < numPaths; ++i)
			{
				nfdchar_t* path{};
				NFD_PathSet_GetPath(outPaths, i, &path);
				resPaths[i] = path;

				NFD_PathSet_FreePath(path);
			}

			NFD_PathSet_Free(outPaths);
		}
		else if (dialogResult == NFD_CANCEL)
		{

		}
		else
		{
			R_ERR_LOG(NFD_GetError());
		}

		return resPaths;
	}
	
	void startImGuiRecording()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}
	void renderInterface()
	{
		if (true)
		{
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}
		else
		{
			ImGui::EndFrame();
		}
	}

	void cleanup()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}
private:
	// Theme from https://github.com/ocornut/imgui/issues/707#issuecomment-917151020
	void setTheme()
	{
		ImVec4* colors = ImGui::GetStyle().Colors;
		colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
		colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
		colors[ImGuiCol_WindowBg]               = ImVec4(0.10f, 0.10f, 0.10f, 0.80f);
		colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_PopupBg]                = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
		colors[ImGuiCol_Border]                 = ImVec4(0.19f, 0.19f, 0.19f, 0.09f);
		colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
		colors[ImGuiCol_FrameBg]                = ImVec4(0.05f, 0.05f, 0.05f, 0.34f);
		colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.19f, 0.19f, 0.19f, 0.34f);
		colors[ImGuiCol_FrameBgActive]          = ImVec4(0.20f, 0.22f, 0.23f, 0.80f);
		colors[ImGuiCol_TitleBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.80f);
		colors[ImGuiCol_TitleBgActive]          = ImVec4(0.06f, 0.06f, 0.06f, 0.80f);
		colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 0.80f);
		colors[ImGuiCol_MenuBarBg]              = ImVec4(0.14f, 0.14f, 0.14f, 0.80f);
		colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.05f, 0.05f, 0.05f, 0.24f);
		colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
		colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
		colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
		colors[ImGuiCol_CheckMark]              = ImVec4(0.82f, 0.72f, 0.66f, 1.00f);
		colors[ImGuiCol_SliderGrab]             = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
		colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
		colors[ImGuiCol_Button]                 = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
		colors[ImGuiCol_ButtonHovered]          = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
		colors[ImGuiCol_ButtonActive]           = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
		colors[ImGuiCol_Header]                 = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
		colors[ImGuiCol_HeaderHovered]          = ImVec4(0.00f, 0.00f, 0.00f, 0.36f);
		colors[ImGuiCol_HeaderActive]           = ImVec4(0.20f, 0.22f, 0.23f, 0.33f);
		colors[ImGuiCol_Separator]              = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
		colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
		colors[ImGuiCol_SeparatorActive]        = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
		colors[ImGuiCol_ResizeGrip]             = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
		colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
		colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
		colors[ImGuiCol_Tab]                    = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
		colors[ImGuiCol_TabHovered]             = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);

		colors[ImGuiCol_TabSelected]            = ImVec4(0.20f, 0.20f, 0.20f, 0.36f);
		colors[ImGuiCol_TabSelectedOverline]    = ImVec4(0.55f, 0.29f, 0.23f, 1.00f);
		colors[ImGuiCol_TabDimmed]              = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
		colors[ImGuiCol_TabDimmedSelected]      = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
		colors[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(0.33f, 0.17f, 0.14f, 1.00f);

		colors[ImGuiCol_PlotLines]              = ImVec4(0.55f, 0.29f, 0.23f, 1.00f);
		colors[ImGuiCol_PlotLinesHovered]       = ImVec4(0.55f, 0.29f, 0.23f, 1.00f);
		colors[ImGuiCol_PlotHistogram]          = ImVec4(0.55f, 0.29f, 0.23f, 1.00f);
		colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(0.55f, 0.29f, 0.23f, 1.00f);
		colors[ImGuiCol_TableHeaderBg]          = ImVec4(0.00f, 0.00f, 0.00f, 0.22f);
		colors[ImGuiCol_TableBorderStrong]      = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
		colors[ImGuiCol_TableBorderLight]       = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
		colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
		colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
		colors[ImGuiCol_DragDropTarget]         = ImVec4(0.82f, 0.72f, 0.66f, 1.00f);
		colors[ImGuiCol_NavHighlight]           = ImVec4(0.55f, 0.29f, 0.23f, 1.00f);
		colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(0.55f, 0.29f, 0.23f, 0.70f);
		colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.55f, 0.29f, 0.23f, 0.20f);
		colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.55f, 0.29f, 0.23f, 0.35f);
		colors[ImGuiCol_DockingPreview]         = ImVec4(0.30f, 0.32f, 0.33f, 0.80f);
		colors[ImGuiCol_DockingEmptyBg]         = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);

		ImGuiStyle& style = ImGui::GetStyle();
		style.WindowPadding                     = ImVec2(8.00f, 8.00f);
		style.FramePadding                      = ImVec2(5.00f, 5.00f);
		style.CellPadding                       = ImVec2(6.00f, 6.00f);
		style.ItemSpacing                       = ImVec2(6.00f, 6.00f);
		style.ItemInnerSpacing                  = ImVec2(6.00f, 6.00f);
		style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
		style.IndentSpacing                     = 25;
		style.ScrollbarSize                     = 15;
		style.GrabMinSize                       = 10;
		style.WindowBorderSize                  = 1;
		style.ChildBorderSize                   = 1;
		style.PopupBorderSize                   = 1;
		style.FrameBorderSize                   = 1;
		style.TabBorderSize                     = 1;
		style.WindowRounding                    = 0;
		style.ChildRounding                     = 0;
		style.FrameRounding                     = 0;
		style.PopupRounding                     = 4;
		style.ScrollbarRounding                 = 9;
		style.GrabRounding                      = 3;
		style.LogSliderDeadzone                 = 4;
		style.TabRounding                       = 4;
	}
};
