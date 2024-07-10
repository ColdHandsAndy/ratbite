#pragma once

#include <glm/vec3.hpp>
#include <glm/common.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "../core/debug_macros.h"

class Camera
{
private:
	glm::dvec3 m_pos{};
	glm::dvec3 m_u{};
	glm::dvec3 m_v{};
	glm::dvec3 m_w{};
	glm::dvec3 m_upWorld{};

	double m_speed{ 280.0 };
	double m_rotationSpeed{ 0.002 };

	glm::ivec3 m_step{ 0 };

	bool m_positionChanged{ false };
	bool m_orientationChanged{ false };
public:
	Camera(const glm::dvec3& position, const glm::dvec3& viewDirection, const glm::dvec3& upDirection) : m_pos{ position }, m_upWorld{ upDirection }
	{
		m_w = glm::normalize(viewDirection);
		m_u = glm::normalize(glm::cross(glm::normalize(upDirection), m_w));
		m_v = glm::cross(m_w, m_u);
	}
	Camera() = delete;
	~Camera() = default;

	enum class Direction
	{
		RIGHT,
		LEFT,
		UP,
		DOWN,
		FORWARD,
		BACKWARD,
		DESC
	};
	void addMoveDir(Direction dir)
	{
		switch (dir)
		{
			case Direction::FORWARD:
				m_step.z += 1;
				break;
			case Direction::BACKWARD:
				m_step.z -= 1;
				break;
			case Direction::RIGHT:
				m_step.x += 1;
				break;
			case Direction::LEFT:
				m_step.x -= 1;
				break;
			case Direction::UP:
				m_step.y += 1;
				break;
			case Direction::DOWN:
				m_step.y -= 1;
				break;
			default:
				R_ASSERT_LOG(true, "Unknown direction passed.");
				break;
		}
	}
	void move(double delta)
	{
		if (m_step.x == 0 && m_step.y == 0 && m_step.z == 0)
			return;
		glm::dvec3 dirStep{ glm::normalize(glm::dvec3{m_step}) * m_speed * delta };
		m_pos += m_u * dirStep.x;
		m_pos += m_upWorld * dirStep.y;
		m_pos += m_w * dirStep.z;
		m_step = glm::ivec3{};
		
		m_positionChanged = true;
	}
	void rotate(double xp, double yp)
	{
		m_w = glm::dvec3{glm::rotate(xp * m_rotationSpeed, m_upWorld) * glm::dvec4{m_w, 1.0}};
		m_u = glm::normalize(glm::cross(glm::normalize(m_upWorld), m_w));
		m_w = glm::dvec3{glm::rotate(yp * m_rotationSpeed, m_u) * glm::dvec4{m_w, 1.0}};
		m_u = glm::normalize(glm::cross(glm::normalize(m_upWorld), m_w));
		m_v = glm::cross(m_w, m_u);

		m_orientationChanged = true;
	}
	const glm::dvec3& getPosition() const { return m_pos; }
	const glm::dvec3& getU() const { return m_u; }
	const glm::dvec3& getV() const { return m_v; }
	const glm::dvec3& getW() const { return m_w; }
	const double getMovingSpeed() const { return m_speed; }

	const bool changesMade() const { return m_positionChanged || m_orientationChanged; }
private:
	void acceptChanges() { m_positionChanged = false; m_orientationChanged = false; }
	bool positionChanged() const { return m_positionChanged; }
	bool orientationChanged() const { return m_orientationChanged; }

	friend class RenderingInterface;
};
