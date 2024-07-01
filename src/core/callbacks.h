#pragma once

#include <cstdint>
#include <glad/glad.h>

void optixLogCallback(unsigned int level, const char* tag, const char* message, void*);
void GLAPIENTRY openGLLogCallback(GLenum source, 
								  GLenum type, 
								  GLuint id,
								  GLenum severity, 
								  GLsizei length,
								  const GLchar *message, 
								  const void *userParam);
void checkGLShaderCompileErrors(uint32_t id);
void checkGLProgramLinkingErrors(uint32_t id);
