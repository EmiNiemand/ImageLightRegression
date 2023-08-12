#include "Components/Rendering/Camera.h"
#include "Managers/RenderingManager.h"
#include "Managers/UIManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"

Camera::Camera(Object *parent, int id) : Component(parent, id) {}

Camera::~Camera() = default;

void Camera::OnUpdate() {
    Component::OnUpdate();

    RenderingManager::GetInstance()->UpdateView();
    RenderingManager::GetInstance()->UpdateProjection();
    UIManager::GetInstance()->UpdateProjection();
}

void Camera::SetFOV(float inFOV) {
    fov = inFOV;
    OnUpdate();
}

void Camera::SetZNear(float inZNear) {
    zNear = inZNear;
    OnUpdate();
}

void Camera::SetZFar(float inZFar) {
    zFar = inZFar;
    OnUpdate();
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

glm::mat4 Camera::GetViewMatrix() const {
    glm::vec3 position = parent->transform->GetGlobalPosition();
    glm::vec3 forward = parent->transform->GetForward();
    glm::vec3 up = parent->transform->GetUp();

    return glm::lookAt(position, position + forward, up);
}

glm::mat4 Camera::GetProjectionMatrix() const {
    return glm::perspective(glm::radians(fov), 16.0f / 9.0f, zNear, zFar);
}
