////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /src/sfcamera.cpp
// Author:            Martin Kirst
// Creation Date:     2012.01.14
// Description:
//
// Implementation and source code file of a SFML based camera.
// Provides simple camera and viewing features like moving around with keys or getting
// the projection matrix.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////


#include <glm/gtc/matrix_transform.hpp>
#include "program.hpp"
#include "sfcamera.hpp"


////////////////////////////////////////////////////////////////////////////////
// Definition: Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
SFCamera::SFCamera(const float& posX, const float& posY, const float& posZ, const float& heading, const float& pitch,
	const float& fov, const float& zNear, const float& zFar)
	: position(posX, posY, posZ),
	up(0.f, 1.f, 0.f),
	center(0.f, 0.f, 1.f),
	projection(glm::perspectiveFov<float>(fov, Globals::RENDER_VIEWPORT_WIDTH, Globals::RENDER_VIEWPORT_HEIGHT, zNear, zFar))
{
	this->heading = heading;
	this->pitch = pitch;
	this->fov = fov;
	this->zNear = zNear;
	this->zFar = zFar;
	view = glm::lookAt(this->position, center, up);
	mouseActive = false;
}


////////////////////////////////////////////////////////////////////////////////
SFCamera::~SFCamera() {
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Accessors
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
const glm::vec3& SFCamera::GetPosition() const {
	return position;
}


////////////////////////////////////////////////////////////////////////////////
const glm::vec3& SFCamera::GetUp() const {
	return up;
}


////////////////////////////////////////////////////////////////////////////////
const glm::vec3& SFCamera::GetCenter() const {
	return center;
}


////////////////////////////////////////////////////////////////////////////////
const float& SFCamera::GetHeading() const {
	return heading;
}


////////////////////////////////////////////////////////////////////////////////
const float& SFCamera::GetPitch() const {
	return pitch;
}


////////////////////////////////////////////////////////////////////////////////
const float& SFCamera::GetFov() const {
	return fov;
}


////////////////////////////////////////////////////////////////////////////////
const float& SFCamera::GetZNear() const {
	return zNear;
}


////////////////////////////////////////////////////////////////////////////////
const float& SFCamera::GetZFar() const {
	return zFar;
}


////////////////////////////////////////////////////////////////////////////////
const glm::mat4& SFCamera::GetView() const {
	return view;
}


////////////////////////////////////////////////////////////////////////////////
const glm::mat4& SFCamera::GetProjection() const {
	return projection;
}


////////////////////////////////////////////////////////////////////////////////
void SFCamera::SetPosition(const float& posX, const float& posY, const float& posZ) {
	position = glm::vec3(posX, posY, posZ);
}


////////////////////////////////////////////////////////////////////////////////
void SFCamera::SetHeading(const float& heading) {
	this->heading = heading;
}


////////////////////////////////////////////////////////////////////////////////
void SFCamera::SetPitch(const float& pitch) {
	this->pitch = pitch;
}


////////////////////////////////////////////////////////////////////////////////
void SFCamera::SetFov(const float& fov) {
	this->fov = fov;
	CalculateProjection();
}


////////////////////////////////////////////////////////////////////////////////
void SFCamera::SetZNear(const float& zNear) {
	this->zNear = zNear;
	CalculateProjection();
}


////////////////////////////////////////////////////////////////////////////////
void SFCamera::SetZFar(const float& zFar) {
	this->zFar = zFar;
	CalculateProjection();
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Public Methods
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void SFCamera::Update() {
	// calculate the current program performance
	float performance = Program::GetElapsedTime() / 1000.f;

	// handle mouse inputs
	if (sf::Mouse::IsButtonPressed(Globals::INPUT_CAM_ROTATION) && !mouseActive) {
		mouseActive = true;
		mouseActivePosition = sf::Mouse::GetPosition();
	}
	else if (sf::Mouse::IsButtonPressed(Globals::INPUT_CAM_ROTATION) && mouseActive) {
		sf::Vector2i mouseDiff = sf::Mouse::GetPosition() - mouseActivePosition;
		heading += mouseDiff.x * Globals::CAM_SENSITIVITY * performance;
		pitch += mouseDiff.y * Globals::CAM_SENSITIVITY * performance;
		sf::Mouse::SetPosition(mouseActivePosition);
	}
	else if (mouseActive)
		mouseActive = false;

	// calculate rotation transformation
	glm::mat4 rotation = glm::rotate(glm::mat4(1.f), heading, glm::normalize(up));
	glm::vec4 homoCenter = rotation * glm::vec4(glm::normalize(center), 1.f);
	glm::vec3 viewCenter(homoCenter.x, homoCenter.y, homoCenter.z);
	glm::vec3 viewSide = glm::cross(glm::normalize(viewCenter), glm::normalize(up));
	rotation = glm::rotate(rotation, pitch, glm::normalize(viewSide));
	homoCenter = rotation * glm::vec4(glm::normalize(center), 1.f);
	viewCenter = glm::vec3(homoCenter.x, homoCenter.y, homoCenter.z);
	viewSide = glm::cross(glm::normalize(viewCenter), glm::normalize(up));

	// handle keyboard inputs
	if (sf::Keyboard::IsKeyPressed(Globals::INPUT_CAM_FORE))
		position += viewCenter * Globals::CAM_VELOCITY * performance;
	if (sf::Keyboard::IsKeyPressed(Globals::INPUT_CAM_LEFT))
		position -= viewSide * Globals::CAM_VELOCITY * performance;
	if (sf::Keyboard::IsKeyPressed(Globals::INPUT_CAM_BACK))
		position -= viewCenter * Globals::CAM_VELOCITY * performance;
	if (sf::Keyboard::IsKeyPressed(Globals::INPUT_CAM_RIGHT))
		position += viewSide * Globals::CAM_VELOCITY * performance;

	// calculate the view matrix
	view = glm::lookAt(position, center, up);
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Private Methods
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void SFCamera::CalculateProjection() {
	projection = glm::perspectiveFov<float>(fov, Globals::RENDER_VIEWPORT_WIDTH, Globals::RENDER_VIEWPORT_HEIGHT, zNear, zFar);
}
