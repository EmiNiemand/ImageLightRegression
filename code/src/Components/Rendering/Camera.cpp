#include "Components/Rendering/Camera.h"
#include "Application.h"
#include "Managers/RenderingManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"

Camera::Camera(Object *parent, int id) : Component(parent, id) {
    projection = glm::perspective(glm::radians(fov),
                                  (float)Application::GetInstance()->resolution.first /
                                  (float)Application::GetInstance()->resolution.second,
                                  zNear, zFar);
}

Camera::~Camera() = default;

void Camera::OnUpdate() {
    Component::OnUpdate();

    RenderingManager::GetInstance()->UpdateView();
}

void Camera::SetFOV(float inFOV) {
    fov = inFOV;
    projection = glm::perspective(glm::radians(fov),
                                  (float)Application::GetInstance()->resolution.first /
                                  (float)Application::GetInstance()->resolution.second,
                                  zNear, zFar);
}

void Camera::SetZNear(float inZNear) {
    zNear = inZNear;
    projection = glm::perspective(glm::radians(fov),
                                  (float)Application::GetInstance()->resolution.first /
                                  (float)Application::GetInstance()->resolution.second,
                                  zNear, zFar);
}

void Camera::SetZFar(float inZFar) {
    zFar = inZFar;
    projection = glm::perspective(glm::radians(fov),
                                  (float)Application::GetInstance()->resolution.first /
                                  (float)Application::GetInstance()->resolution.second,
                                  zNear, zFar);
}

void Camera::SetActiveCamera(Object* inCameraObject) {
    if (inCameraObject->GetComponentByClass<Camera>() == nullptr) return;
    activeCamera = inCameraObject;
    RenderingManager::GetInstance()->UpdateView();
    RenderingManager::GetInstance()->UpdateProjection();
}

Object *Camera::GetActiveCamera() {
    return activeCamera;
}

glm::mat4 Camera::GetViewMatrix() {
    glm::vec3 position = parent->transform->GetGlobalPosition();
    glm::vec3 front = parent->transform->GetForward();
    glm::vec3 up = parent->transform->GetUp();

    return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::GetProjectionMatrix() {
    return projection;
}
