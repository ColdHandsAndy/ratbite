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

	double m_fieldOfViewRad{ glm::radians(70.0f) };

	double m_speed{ 3.0 };
	double m_rotationSpeed{ 0.002 };

	glm::ivec3 m_step{ 0 };

	bool m_depthOfFieldEnabled{ false };
	double m_focusDistance{ 1.0 };
	double m_aperture{ 0.01 };

public:
	Camera(const glm::dvec3& position, const glm::dvec3& viewDirection, const glm::dvec3& upDirection) : m_pos{ position }, m_upWorld{ glm::normalize(upDirection) }
	{
		m_v = glm::normalize(viewDirection);
		m_u = glm::normalize(glm::cross(m_v, glm::normalize(upDirection)));
		m_w = glm::cross(m_u, m_v);
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
				m_step.y += 1;
				break;
			case Direction::BACKWARD:
				m_step.y -= 1;
				break;
			case Direction::RIGHT:
				m_step.x += 1;
				break;
			case Direction::LEFT:
				m_step.x -= 1;
				break;
			case Direction::UP:
				m_step.z += 1;
				break;
			case Direction::DOWN:
				m_step.z -= 1;
				break;
			default:
				R_ASSERT_LOG(true, "Unknown direction passed.");
				break;
		}
	}
	bool move(double delta)
	{
		if (m_step.x == 0 && m_step.y == 0 && m_step.z == 0)
			return false;
		glm::dvec3 dirStep{ glm::normalize(glm::dvec3{m_step}) * m_speed * delta };
		m_pos += m_u * dirStep.x;
		m_pos += m_upWorld * dirStep.z;
		m_pos += m_v * dirStep.y;
		m_step = glm::ivec3{ 0 };
		return true;
	}
	void rotate(double xp, double yp)
	{
		m_v = glm::dvec3{glm::rotate(-xp * m_rotationSpeed, m_upWorld) * glm::dvec4{m_v, 1.0}};
		m_u = glm::normalize(glm::cross(m_v, m_upWorld));
		glm::dvec3 newV{ glm::dvec3{glm::rotate(-yp * m_rotationSpeed, m_u) * glm::dvec4{m_v, 1.0}} };
		glm::dvec3 newU{ glm::normalize(glm::cross(newV, m_upWorld)) };
		if (glm::dot(newU, m_u) > 0.0)
		{
			m_v = newV;
			m_u = newU;
		}
		m_w = glm::cross(m_u, m_v);
	}
	void setFieldOfView(double radians) { m_fieldOfViewRad = radians; }
	void setMovingSpeed(double speed) { m_speed = speed; }
	void setRotationSpeed(double speed) { m_rotationSpeed = speed; }
	void setFocusDistance(double fd) { m_focusDistance = std::max(0.0, std::min(65504.0, fd)); }
	void setAperture(double a) { m_aperture = std::max(0.0, std::min(65504.0, a)); }
	void setDepthOfField(bool enabled) { m_depthOfFieldEnabled = enabled; }
	const glm::dvec3& getPosition() const { return m_pos; }
	const glm::dvec3& getU() const { return m_u; }
	const glm::dvec3& getV() const { return m_v; }
	const glm::dvec3& getW() const { return m_w; }
	const double getFieldOfView() const { return m_fieldOfViewRad; }
	const double getMovingSpeed() const { return m_speed; }
	const double getRotationSpeed() const { return m_rotationSpeed; }
	const double getFocusDistance() const { return m_focusDistance; }
	const double getAperture() const { return m_aperture; }
	const bool depthOfFieldEnabled() const { return m_depthOfFieldEnabled; }
};
