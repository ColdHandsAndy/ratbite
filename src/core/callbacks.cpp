#include "callbacks.h"

#include <cstdint>
#include <iostream>
#include <format>

void optixLogCallback(unsigned int level, const char* tag, const char* message, void*) 
{
	std::cerr << std::format("Optix Log: [  Level - {}  ] [  Tag - {}  ]:\n\t{}\n", level, tag, message);
}

void GLAPIENTRY openGLLogCallback(GLenum source, 
								  GLenum type, 
								  GLuint id,
								  GLenum severity, 
								  GLsizei length,
								  const GLchar *message, 
								  const void *userParam)
{
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204)
		return;

	std::string srcstr{};
	switch (source)
	{
		case GL_DEBUG_SOURCE_API:
			srcstr = "Source - API";
			break;
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
			srcstr = "Source - Window System";
			break;
		case GL_DEBUG_SOURCE_SHADER_COMPILER:
			srcstr = "Source - Shader Compiler";
			break;
		case GL_DEBUG_SOURCE_THIRD_PARTY:
			srcstr = "Source - Third Party";
			break;
		case GL_DEBUG_SOURCE_APPLICATION:
			srcstr = "Source - Application";
			break;
		case GL_DEBUG_SOURCE_OTHER:
			srcstr = "Source - Other";
			break;
	};

	std::string typestr{};
	switch (type)
	{
		case GL_DEBUG_TYPE_ERROR:
			typestr = "Type - Error";
			break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			typestr = "Type - Deprecated Behaviour";
			break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			typestr = "Type - Undefined Behaviour";
			break;
		case GL_DEBUG_TYPE_PORTABILITY:
			typestr = "Type - Portability";
			break;
		case GL_DEBUG_TYPE_PERFORMANCE:
			typestr = "Type - Performance";
			break;
		case GL_DEBUG_TYPE_MARKER:
			typestr = "Type - Marker";
			break;
		case GL_DEBUG_TYPE_PUSH_GROUP:
			typestr = "Type - Push Group";
			break;
		case GL_DEBUG_TYPE_POP_GROUP:
			typestr = "Type - Pop Group";
			break;
		case GL_DEBUG_TYPE_OTHER:
			typestr = "Type - Other";
			break;
	};

	std::string severitystr{};
	switch (severity)
	{
		case GL_DEBUG_SEVERITY_HIGH:
			severitystr = "Severity - High";
			break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			severitystr = "Severity - Medium";
			break;
		case GL_DEBUG_SEVERITY_LOW:
			severitystr = "Severity - Low";
			break;
		case GL_DEBUG_SEVERITY_NOTIFICATION:
			severitystr = "Severity - Notification";
			break;
	};

	std::cerr << std::format("OpenGL Log: [  {}  ] [  {}  ] [  {}  ]:\n\t{}\n", srcstr, typestr, severitystr, message);
}
void checkGLShaderCompileErrors(uint32_t id)
{
	int success;
	char infoLog[2048];

	glGetShaderiv(id, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(id, 1024, NULL, infoLog);
		std::cerr << std::format("OpenGL shader compilation error.\n\tLog: {}\n", infoLog);
	}
}
void checkGLProgramLinkingErrors(uint32_t id)
{
	int success;
	char infoLog[2048];

	glGetProgramiv(id, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(id, 1024, NULL, infoLog);
		std::cerr << std::format("OpenGL program linking error.\n\tLog: {}\n", infoLog);
	}
}
