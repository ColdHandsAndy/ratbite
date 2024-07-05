#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "debug_macros.h"
#include "callbacks.h"

class Window;
class RenderingInterface;
void draw(Window* window, const RenderingInterface* rInterface);

class Window
{
private:
	int m_width{};
	int m_height{};
	float m_invWidth{};
	float m_invHeight{};

	bool m_sizeChanged{ false };

	GLFWwindow* m_glfwWindow{};

	struct UserDataGLFW
	{
		Window* window{};
		const RenderingInterface* rInterface{};
		void (*draw)(Window* window, const RenderingInterface* rInterface){};
	} m_userPointer{};
public:
	Window(int width, int height) : m_width{ width }, m_height{ height }, m_invWidth{ 1.0f / width }, m_invHeight{ 1.0f / height } 
	{
		R_ASSERT_LOG(width > 0 && height > 0, "Window resolution is nonsensical\n");

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		m_glfwWindow = glfwCreateWindow(m_width, m_height, "ratbite", NULL, NULL);
		R_ASSERT(m_glfwWindow != nullptr);
		glfwMakeContextCurrent(m_glfwWindow);
		glfwSetWindowUserPointer(m_glfwWindow, &m_userPointer);
		m_userPointer.window = this;
		m_userPointer.draw = draw;
		glfwSwapInterval(1);
		glfwSetFramebufferSizeCallback(m_glfwWindow, glfwFramebufferSizeCallback);
		glfwSetWindowRefreshCallback(m_glfwWindow, glfwWindowRefreshCallback);
		R_ASSERT_LOG(gladLoadGLLoader((GLADloadproc) glfwGetProcAddress), "GLAD failed to load OpenGL funcitons");

#ifdef _DEBUG
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(openGLLogCallback, 0);
#endif

		glViewport(0, 0, m_width, m_height);
	}
	Window() = delete;
	Window(Window&&) = delete;
	Window(const Window&) = delete;
	Window &operator=(Window&&) = delete;
	Window &operator=(const Window&) = delete;
	~Window() = default;

	void attachRenderingInterface(const RenderingInterface* renderingInterface)
	{
		m_userPointer.rInterface = renderingInterface;
	}

	void resize(int newWidth, int newHeight)
	{
		m_width = newWidth;
		m_height = newHeight;
		m_invWidth = 1.0f / newWidth;
		m_invHeight = 1.0f / newHeight;

		glViewport(0, 0, newWidth, newHeight);
	}

	GLFWwindow* getGLFWwindow() { return m_glfwWindow; }

	int getWidth() const { return m_width; }
	int getHeight() const { return m_height; }
	int getInvWidth() const { return m_invWidth; }
	int getInvHeight() const { return m_invHeight; }
private:
	static void glfwWindowRefreshCallback(GLFWwindow* window)
	{
		UserDataGLFW* userData{ reinterpret_cast<UserDataGLFW*>(glfwGetWindowUserPointer(window)) };
		userData->draw(userData->window, userData->rInterface);
	}
	static void glfwFramebufferSizeCallback(GLFWwindow* window, int width, int height)
	{
		Window* win{ reinterpret_cast<UserDataGLFW*>(glfwGetWindowUserPointer(window))->window };
		win->resize(width, height);
	}
};