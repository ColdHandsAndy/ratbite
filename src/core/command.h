#pragma once

#include <cstdint>
#include <stack>

#include "../core/scene.h"

enum class CommandType : int
{
	NO_COMMAND,

	CHANGE_RENDER_MODE,

	CHANGE_SAMPLE_COUNT,
	CHANGE_PATH_LENGTH,
	CHANGE_IMAGE_EXPOSURE,
	CHANGE_DEPTH_OF_FIELD_SETTINGS,

	CHANGE_RENDER_RESOLUTION,

	CHANGE_CAMERA_POS,
	CHANGE_CAMERA_ORIENTATION,

	ADD_LIGHT,
	CHANGE_LIGHT_POSITION,
	CHANGE_LIGHT_SIZE,
	CHANGE_LIGHT_ORIENTATION,
	CHANGE_LIGHT_POWER,
	CHANGE_LIGHT_EMISSION_SPECTRUM,
	REMOVE_LIGHT,

	ADD_MODEL,
	CHANGE_MODEL_MATERIAL,
	CHANGE_MODEL_TRANSFORM,
	REMOVE_MODEL,

	DESC
};

namespace CommandPayloads
{
	struct Light
	{
		LightType type{};
		uint32_t id{};
		uint32_t index{};
		SpectralData::SpectralDataType oldEmissionType{};
	};
	struct Model
	{
		uint32_t id{};
		uint32_t index{};
		SceneData::BxDF matType{};
		SpectralData::SpectralDataType oldIORType{};
		SpectralData::SpectralDataType oldACType{};
	};
}

struct Command
{
	const CommandType type{ CommandType::NO_COMMAND };
	const void* payload{ nullptr };
};

class CommandBuffer
{
private:
	std::stack<Command> m_commands{};
public:
	CommandBuffer() = default;
	CommandBuffer(CommandBuffer&&) = default;
	CommandBuffer(const CommandBuffer&) = default;
	CommandBuffer& operator=(CommandBuffer&&) = default;
	CommandBuffer& operator=(const CommandBuffer&) = default;
	~CommandBuffer() = default;

	bool empty() const { return m_commands.empty(); }

	void pushCommand(const Command& command)
	{
		m_commands.push(command);
	}
	Command pullCommand()
	{
		Command tmp{ m_commands.top() };
		m_commands.pop();
		return tmp;
	}
	void clear() { m_commands = {}; }
};
