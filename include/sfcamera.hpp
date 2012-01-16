////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /include/sfcamera.hpp
// Author:            Martin Kirst
// Creation Date:     2012.01.14
// Description:
//
// Declaration of a SFML based camera.
// Provides simple camera and viewing features like moving around with keys or getting
// the projection matrix.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////


#ifndef SFCAMERA_HPP_
#define SFCAMERA_HPP_

#include <glm/glm.hpp>
#include "globals.hpp"


////////////////////////////////////////////////////////////////////////////////
// Class Declaration
////////////////////////////////////////////////////////////////////////////////


class SFCamera {
public:
	// Constructors and Destructor
	////////////////////////////////////////////////////////////////////////////////
	SFCamera(const float& posX = Globals::CAM_POSX, const float& posY = Globals::CAM_POSY, const float& posZ = Globals::CAM_POSZ,
		const float& heading = Globals::CAM_HEADING, const float& pitch = Globals::CAM_PITCH,
		const float& fov = Globals::CAM_FOV, const float& zNear = Globals::CAM_ZNEAR, const float& zFar = Globals::CAM_ZFAR);
	~SFCamera();

	// Accessors
	////////////////////////////////////////////////////////////////////////////////
	const glm::vec3& GetPosition() const;
	const glm::vec3& GetUp() const;
	const glm::vec3& GetCenter() const;
	const float& GetHeading() const;
	const float& GetPitch() const;
	const float& GetFov() const;
	const float& GetZNear() const;
	const float& GetZFar() const;
	const glm::mat4& GetView() const;
	const glm::mat4& GetProjection() const;
	void SetPosition(const float& posX, const float& posY, const float& posZ);
	void SetHeading(const float& heading);
	void SetPitch(const float& pitch);
	void SetFov(const float& fov);
	void SetZNear(const float& zNear);
	void SetZFar(const float& zFar);

	// Public Methods
	////////////////////////////////////////////////////////////////////////////////
	void Update();

private:
	// Private Methods
	////////////////////////////////////////////////////////////////////////////////
	void CalculateProjection();

	// Variables
	////////////////////////////////////////////////////////////////////////////////
	glm::vec3 position;
	glm::vec3 up;
	glm::vec3 center;
	float heading;
	float pitch;
	float fov;
	float zNear;
	float zFar;
	glm::mat4 view;
	glm::mat4 projection;
	bool mouseActive;
	sf::Vector2i mouseActivePosition;
	unsigned int elapsedTime;
};


#endif // SFCAMERA_HPP_
