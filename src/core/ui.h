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
	bool m_previewWindowIsFocused{ false };
	bool m_cursorIsDraggingOverPreviewWindow{ false };
	bool m_imageRenderWindowIsOpen{ false };
	bool m_imageRenderSettingsWindowIsOpen{ false };
public:
	UI(GLFWwindow* window)
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io{ ImGui::GetIO() };
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		io.ConfigWindowsMoveFromTitleBarOnly = true;
		setTheme();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init();

		// Record empty ImGui frame so window resize callback doesn't throw an error
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::EndFrame();
		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backupContext{ glfwGetCurrentContext() };
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backupContext);
		}
	}
	~UI() = default;
	UI(UI&&) = delete;
	UI& operator=(UI&&) = delete;
	UI(const UI&) = delete;
	UI& operator=(const UI&) = delete;

	std::string getFileFromFileDialogWindow(GLFWwindow* window, const char* defaultPath, const char* fileFilters)
	{
		std::string resPath{};
		nfdfilteritem_t filters{ .name = "Filters", .spec = fileFilters };
		nfdwindowhandle_t parentWindow{};
		NFD_GetNativeWindowFromGLFWWindow(window, &parentWindow);
		nfdopendialogu8args_t dialogArgs{
			.filterList = &filters,
			.filterCount = 1,
			.defaultPath = defaultPath,
			.parentWindow = parentWindow };

		nfdu8char_t* outPath{};
		nfdresult_t dialogResult{ NFD_OpenDialogU8_With(&outPath, &dialogArgs) };
		if (dialogResult == NFD_OKAY)
		{
			resPath = outPath;
			NFD_PathSet_FreePath(outPath);
		}
		else if (dialogResult == NFD_CANCEL)
		{
		}
		else
		{
			R_ERR_LOG(NFD_GetError());
		}
		return resPath;
	}
	std::vector<std::filesystem::path> getFilesFromFileDialogWindow(GLFWwindow* window, const char* defaultPath, const char* fileFilters) const
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
	std::string saveFilesWithFileDialogWindow(GLFWwindow* window, const char* defaultPath, const char* defaultName, const char* fileFilters) const
	{
		std::string resPath{};
		nfdfilteritem_t filters{ .name = "Filters", .spec = fileFilters };
		nfdwindowhandle_t parentWindow{};
		NFD_GetNativeWindowFromGLFWWindow(window, &parentWindow);
		nfdsavedialogu8args_t dialogArgs{
			.filterList = &filters,
			.filterCount = 1,
			.defaultPath = defaultPath,
			.defaultName = defaultName,
			.parentWindow = parentWindow };
		nfdu8char_t* outPath{};
		nfdresult_t dialogResult{ NFD_SaveDialogU8_With(&outPath, &dialogArgs) };
		if (dialogResult == NFD_OKAY)
		{
			resPath = outPath;
			NFD_PathSet_FreePath(outPath);
		}
		else if (dialogResult == NFD_CANCEL)
		{
		}
		else
		{
			R_ERR_LOG(NFD_GetError());
		}
		return resPath;
	}

	void recordInterface(CommandBuffer& commands, Window& window, Camera& camera, RenderContext& rContext, SceneData& scene, GLuint renderResult, int currentSampleCount);
	void recordInput(CommandBuffer& commands, Window& window, Camera& camera, RenderContext& rContext);
	
	bool mouseIsCaptured() const { return ImGui::GetIO().WantCaptureMouse; }
	bool keyboardIsCaptured() const { return ImGui::GetIO().WantCaptureKeyboard; }

	void drawInterface(bool noUpdates)
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

		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backupContext{ glfwGetCurrentContext() };
			if (!noUpdates) ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backupContext);
		}
	}

	void cleanup()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}
private:
	void startImGuiRecording()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	void recordDockspace(CommandBuffer& commands, Window& window, bool& openImageRenderSettings);
	void recordImageRenderSettingsWindow(CommandBuffer& commands, RenderContext& rContext, GLuint renderResult, bool openImageRenderSettings);
	void recordImageRenderWindow(CommandBuffer& commands, Window& window, RenderContext& rContext, GLuint renderResult, int currentSampleCount);
	void recordPreviewWindow(CommandBuffer& commands, RenderContext& rContext,
			float& renderWinWidth, float& renderWinHeight,
			GLuint renderResult, float renderScale);
	void recordRenderSettingsWindow(CommandBuffer& commands, Camera& camera, RenderContext& rContext, float& renderScale, float renderWinWidth, float renderWinHeight);
	void recordSceneGeneralSettings(CommandBuffer& commands, Window& window, SceneData& scene, Camera& camera, const ImVec4& infoColor);
	void recordSceneModelsSettings(CommandBuffer& commands, Window& window, SceneData& scene, const ImVec4& infoColor);
	void recordSceneLightsSettings(CommandBuffer& commands, SceneData& scene, const ImVec4& infoColor);
	void recordSceneEnvironmentMapSettings(CommandBuffer& commands, Window& window, SceneData& scene, const ImVec4& infoColor);
	void recordCoordinateFrameWindow(Camera& camera);
	void recordInformationWindow(SceneData& scene, int currentSampleCount);

private:
	void setTheme()
	{
		ImVec4* colors = ImGui::GetStyle().Colors;
		colors[ImGuiCol_Text]                   = ImVec4(0.831f, 0.847f, 0.878f, 1.000f);
		colors[ImGuiCol_TextDisabled]           = ImVec4(0.831f, 0.847f, 0.878f, 0.502f);

		colors[ImGuiCol_WindowBg]               = ImVec4(0.173f, 0.192f, 0.235f, 1.000f);
		colors[ImGuiCol_ChildBg]                = ImVec4(0.000f, 0.000f, 0.000f, 0.159f);
		colors[ImGuiCol_PopupBg]                = ImVec4(0.173f, 0.192f, 0.235f, 1.000f);

		colors[ImGuiCol_Border]                 = ImVec4(0.816f, 0.722f, 0.659f, 0.706f);
		colors[ImGuiCol_BorderShadow]           = ImVec4(0.000f, 0.000f, 0.000f, 0.000f);

		colors[ImGuiCol_FrameBg]                = ImVec4(0.106f, 0.114f, 0.137f, 0.502f);
		colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.553f, 0.286f, 0.227f, 0.078f);
		colors[ImGuiCol_FrameBgActive]          = ImVec4(0.553f, 0.286f, 0.227f, 0.216f);

		colors[ImGuiCol_TitleBg]                = ImVec4(0.129f, 0.129f, 0.157f, 1.000f);
		colors[ImGuiCol_TitleBgActive]          = ImVec4(0.129f, 0.129f, 0.157f, 1.000f);
		colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.129f, 0.129f, 0.157f, 1.000f);
		colors[ImGuiCol_MenuBarBg]              = ImVec4(0.129f, 0.129f, 0.157f, 1.000f);

		colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.020f, 0.020f, 0.020f, 0.000f);
		colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.533f, 0.533f, 0.533f, 1.000f);
		colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.333f, 0.333f, 0.333f, 1.000f);
		colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.600f, 0.600f, 0.600f, 1.000f);

		colors[ImGuiCol_CheckMark]              = ImVec4(0.816f, 0.722f, 0.659f, 1.000f);

		colors[ImGuiCol_SliderGrab]             = ImVec4(0.553f, 0.286f, 0.227f, 0.863f);
		colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.553f, 0.286f, 0.227f, 0.949f);

		colors[ImGuiCol_Button]                 = ImVec4(0.353f, 0.413f, 0.442f, 0.502f);
		colors[ImGuiCol_ButtonHovered]          = ImVec4(0.153f, 0.173f, 0.212f, 1.000f);
		colors[ImGuiCol_ButtonActive]           = ImVec4(0.553f, 0.286f, 0.227f, 1.000f);

		colors[ImGuiCol_Header]                 = ImVec4(0.229f, 0.259f, 0.318f, 1.000f);
		colors[ImGuiCol_HeaderHovered]          = ImVec4(0.553f, 0.286f, 0.227f, 0.725f);
		colors[ImGuiCol_HeaderActive]           = ImVec4(0.553f, 0.286f, 0.227f, 1.000f);

		colors[ImGuiCol_Separator]              = ImVec4(0.875f, 0.827f, 0.765f, 0.729f);
		colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.875f, 0.827f, 0.765f, 1.000f);
		colors[ImGuiCol_SeparatorActive]        = ImVec4(0.973f, 0.929f, 0.890f, 1.000f);

		colors[ImGuiCol_ResizeGrip]             = ImVec4(0.106f, 0.114f, 0.137f, 1.000f);
		colors[ImGuiCol_ResizeGripHovered]      = ImVec4(1.000f, 0.929f, 0.863f, 0.784f);
		colors[ImGuiCol_ResizeGripActive]       = ImVec4(1.000f, 0.929f, 0.863f, 1.000f);

		colors[ImGuiCol_Tab]                    = ImVec4(0.553f, 0.286f, 0.227f, 0.392f);
		colors[ImGuiCol_TabHovered]             = ImVec4(0.608f, 0.314f, 0.251f, 1.000f);
		colors[ImGuiCol_TabSelected]            = ImVec4(0.553f, 0.286f, 0.227f, 1.000f);
		colors[ImGuiCol_TabSelectedOverline]    = ImVec4(1.000f, 0.929f, 0.863f, 1.000f);
		colors[ImGuiCol_TabDimmed]              = ImVec4(0.553f, 0.286f, 0.227f, 0.353f);
		colors[ImGuiCol_TabDimmedSelected]      = ImVec4(0.553f, 0.286f, 0.227f, 0.706f);
		colors[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(1.000f, 0.929f, 0.863f, 0.500f);

		colors[ImGuiCol_DockingPreview]         = ImVec4(0.553f, 0.286f, 0.227f, 0.627f);
		colors[ImGuiCol_DockingEmptyBg]         = ImVec4(0.200f, 0.200f, 0.200f, 1.000f);

		colors[ImGuiCol_PlotLines]              = ImVec4(0.608f, 0.608f, 0.608f, 1.000f);
		colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.000f, 0.427f, 0.349f, 1.000f);
		colors[ImGuiCol_PlotHistogram]          = ImVec4(0.898f, 0.698f, 0.000f, 1.000f);
		colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.000f, 0.600f, 0.000f, 1.000f);

		colors[ImGuiCol_TableHeaderBg]          = ImVec4(0.106f, 0.114f, 0.137f, 1.000f);
		colors[ImGuiCol_TableBorderStrong]      = ImVec4(0.204f, 0.231f, 0.282f, 1.000f);
		colors[ImGuiCol_TableBorderLight]       = ImVec4(0.204f, 0.231f, 0.282f, 0.502f);
		colors[ImGuiCol_TableRowBg]             = ImVec4(0.000f, 0.000f, 0.000f, 0.000f);
		colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.000f, 1.000f, 1.000f, 0.039f);

		colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.204f, 0.231f, 0.282f, 1.000f);
		colors[ImGuiCol_DragDropTarget]         = ImVec4(1.000f, 1.000f, 0.000f, 0.900f);

		colors[ImGuiCol_NavHighlight]           = ImVec4(0.553f, 0.286f, 0.227f, 1.000f);
		colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(0.204f, 0.231f, 0.282f, 0.753f);
		colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.106f, 0.114f, 0.137f, 0.753f);
		colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.106f, 0.114f, 0.137f, 0.753f);


		ImGuiStyle& style = ImGui::GetStyle();
		style.WindowPadding                     = ImVec2(3.00f, 3.00f);
		style.FramePadding                      = ImVec2(6.00f, 3.00f);
		style.ItemSpacing                       = ImVec2(8.00f, 4.00f);
		style.ItemInnerSpacing                  = ImVec2(4.00f, 4.00f);
		style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
		style.IndentSpacing                     = 8;
		style.ScrollbarSize                     = 13;
		style.GrabMinSize                       = 14;

		style.WindowBorderSize                  = 1;
		style.ChildBorderSize                   = 0;
		style.PopupBorderSize                   = 1;
		style.FrameBorderSize                   = 0;
		style.TabBorderSize                     = 0;
		style.TabBarBorderSize                  = 1;
		style.TabBarOverlineSize                = 2;

		style.WindowRounding                    = 0;
		style.ChildRounding                     = 3;
		style.FrameRounding                     = 0;
		style.PopupRounding                     = 0;
		style.ScrollbarRounding                 = 12;
		style.GrabRounding                      = 3;
		style.TabRounding                       = 0;

		style.CellPadding                       = ImVec2(4.00f, 2.00f);
		style.TableAngledHeadersAngle           = 35.0f;
		style.TableAngledHeadersTextAlign       = ImVec2(0.50f, 0.00f);

		style.SeparatorTextBorderSize           = 3;
		style.SeparatorTextAlign                = ImVec2(0.00f, 0.50f);
		style.SeparatorTextPadding              = ImVec2(20.0f, 3.0f);

		style.LogSliderDeadzone                 = 4;
	}
};
