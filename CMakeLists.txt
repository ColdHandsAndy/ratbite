cmake_minimum_required(VERSION 3.30)

# Build into a separate directory
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "ERROR: Build should not be in the same directory as the source code.")
endif()


# Set the OptiX SDK path on your system
set(OPTIX_SDK_PATH $ENV{OPTIX_SDK_PATH})

# Change for your needs
set(CMAKE_DEFAULT_BUILD_TYPE "Debug")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER "cl")
set(CMAKE_CXX_COMPILER "cl")
set(CMAKE_CUDA_COMPILER "nvcc")
set(CMAKE_CUDA_ARCHITECTURES "61")


# Find CUDA Toolkit
find_package(CUDAToolkit)


# Main application
project(ratbite LANGUAGES CXX)

link_directories(${PROJECT_NAME} "${CMAKE_SOURCE_DIR}/dependencies/lib")

file(GLOB_RECURSE source_files CONFIGURE_DEPENDS
	"${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE lib_source_files CONFIGURE_DEPENDS
	"${CMAKE_SOURCE_DIR}/dependencies/src/*.cpp")
add_executable(ratbite ${source_files} ${lib_source_files})
set_property(TARGET ratbite PROPERTY CXX_STANDARD 20)
set_property(TARGET ratbite PROPERTY CXX_STANDARD_REQUIRED ON)
target_include_directories(ratbite PUBLIC
	"${CMAKE_SOURCE_DIR}/dependencies/include"
	"${CMAKE_SOURCE_DIR}/dependencies/src/imgui"
	"${CMAKE_SOURCE_DIR}/dependencies/src/glad/include"
	"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
	"$ENV{OPTIX_SDK_PATH}/include")
target_link_libraries(ratbite PUBLIC opengl32)
target_link_libraries(ratbite PUBLIC glfw3)
target_link_libraries(ratbite PUBLIC nfd)
target_link_libraries(ratbite PUBLIC CUDA::cudart)
target_link_libraries(ratbite PUBLIC CUDA::cuda_driver)


# CUDA and OptiX kernels
project(ratbiteCUDA LANGUAGES CUDA)

set(cuda_ptx_source_files "${CMAKE_SOURCE_DIR}/src/kernels/pathtrace.cu")
list(APPEND cuda_ptx_source_files "${CMAKE_SOURCE_DIR}/src/kernels/image.cu")
list(APPEND cuda_ptx_source_files "${CMAKE_SOURCE_DIR}/src/kernels/cdf.cu")
add_library(ptx OBJECT ${cuda_ptx_source_files})
set_property(TARGET ptx PROPERTY CUDA_STANDARD 20)
set_property(TARGET ptx PROPERTY CUDA_STANDARD_REQUIRED ON)
set_property(TARGET ptx PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET ptx PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(ptx PUBLIC
	"${CMAKE_SOURCE_DIR}/dependencies/include"
	"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
	"${OPTIX_SDK_PATH}/include")
target_link_libraries(ptx PUBLIC CUDA::cudart)
target_compile_options(ptx PUBLIC $<$<CONFIG:Debug>:${CMAKE_CUDA_FLAGS} -lineinfo>)

# Copy kernels to the target dir
add_custom_target(copyPTX ALL)
add_dependencies(copyPTX ptx)
add_custom_command(TARGET copyPTX POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:ptx> $<TARGET_FILE_DIR:ratbite>
                   COMMAND_EXPAND_LISTS)


# Copy data files to the target dir
add_custom_command(TARGET ratbite POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/bin/albedoConductorLUT.bin
	$<TARGET_FILE_DIR:ratbite>/bin)
add_custom_command(TARGET ratbite POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/bin/albedoDielectricInnerLUT.bin
	$<TARGET_FILE_DIR:ratbite>/bin)
add_custom_command(TARGET ratbite POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/bin/albedoDielectricOuterLUT.bin
	$<TARGET_FILE_DIR:ratbite>/bin)
add_custom_command(TARGET ratbite POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/bin/albedoDielectricReflectiveInnerLUT.bin
	$<TARGET_FILE_DIR:ratbite>/bin)
add_custom_command(TARGET ratbite POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/bin/albedoDielectricReflectiveOuterLUT.bin
	$<TARGET_FILE_DIR:ratbite>/bin)
add_custom_command(TARGET ratbite POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/bin/sheenLUT.bin
	$<TARGET_FILE_DIR:ratbite>/bin)

add_custom_command(TARGET ratbite POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/docs/pics/logo_icon_64.png $<TARGET_FILE_DIR:ratbite>)

# Copy compile commands for an editor
if (${CMAKE_EXPORT_COMPILE_COMMANDS})
    add_custom_command(TARGET ratbite
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR})
endif()
