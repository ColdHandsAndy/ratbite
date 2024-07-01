#pragma once

#include <glm/vec3.hpp>
#include <glm/common.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/dual_quaternion.hpp>

class Camera
{
private:
	glm::dualquat m_cameraFromWorld{};
	glm::vec3 m_pos{};
	glm::vec3 m_u{};
	glm::vec3 m_v{};
	glm::vec3 m_w{};
public:
	Camera(const glm::vec3& position, const glm::vec3& viewDirection, const glm::vec3& upDirection) : m_pos{ position }
	{
		m_w = glm::normalize(viewDirection);
		m_u = glm::normalize(glm::cross(glm::normalize(upDirection), m_w));
		m_v = glm::cross(m_w, m_u);
		glm::vec3 trans{ glm::transpose(glm::mat3{ m_u, m_v, m_w }) * (-position) };
		m_cameraFromWorld = glm::dualquat_cast(glm::mat3x4(glm::vec4(m_u, trans.x), glm::vec4(m_v, trans.y), glm::vec4(m_w, trans.z)));
	}
	Camera() = delete;
	~Camera() = default;

	const glm::vec3& getPosition() const { return m_pos; }
	const glm::vec3& getU() const { return m_u; }
	const glm::vec3& getV() const { return m_v; }
	const glm::vec3& getW() const { return m_w; }
};
