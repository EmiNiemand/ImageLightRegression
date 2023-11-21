#include "Components/Rendering/Camera.h"
#include "Managers/RenderingManager.h"
#include "Managers/InputManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Macros.h"

Camera::Camera(Object* parent, int id) : Component(parent, id) {}

Camera::~Camera() = default;

void Camera::OnCreate() {
    Component::OnCreate();

    if ((!renderingCamera && parent != editorCamera) || editorCamera == renderingCamera) {
        renderingCamera = parent;
    }
}

void Camera::OnDestroy() {
    Component::OnDestroy();

    if (parent == renderingCamera) {
        for (auto& component : Application::GetInstance()->components) {
            if (dynamic_cast<Camera*>(component.second) != nullptr && component.second->parent != editorCamera) {
                renderingCamera = component.second->parent;
                break;
            }
        }
    }

    if (parent == renderingCamera) renderingCamera = nullptr;

    if (renderingCamera) {
        activeCamera = renderingCamera;
    }
    else if (editorCamera) {
        activeCamera = editorCamera;
        renderingCamera = editorCamera;
    }
    else {
        activeCamera = nullptr;
    }
}

void Camera::OnUpdate() {
    Component::OnUpdate();

    if (parent == activeCamera && parent == renderingCamera && !parent->GetEnabled() || !enabled) {
        activeCamera = editorCamera;
    }

    RenderingManager::GetInstance()->UpdateView();
    RenderingManager::GetInstance()->UpdateProjection();
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
    if (inZFar < zNear) return;
    zFar = inZFar;
    OnUpdate();
}

float Camera::GetFOV() const {
    return fov;
}

float Camera::GetZNear() const {
    return zNear;
}

float Camera::GetZFar() const {
    return zFar;
}

void Camera::ChangeActiveCamera() {
    if (activeCamera == editorCamera && renderingCamera->GetEnabled() &&
        renderingCamera->GetComponentByClass<Camera>()->enabled) {
        SetActiveCamera(renderingCamera);
    }
    else if (activeCamera == renderingCamera) {
        SetActiveCamera(editorCamera);
    }
    else {
        SetActiveCamera(previouslyActiveCamera);
    }
}

void Camera::SetActiveCamera(Object* inCamera) {
    if (!(renderingCamera && editorCamera)) return;

    if (activeCamera == editorCamera || activeCamera == renderingCamera) previouslyActiveCamera = activeCamera;
    activeCamera = inCamera;

    activeCamera->GetComponentByClass<Camera>()->OnUpdate();
}

void Camera::SetRenderingCamera(Object *inCameraObject) {
    if (inCameraObject == editorCamera) return;
    if (activeCamera == renderingCamera) activeCamera = inCameraObject;
    renderingCamera = inCameraObject;
}

Object *Camera::GetActiveCamera() {
    return activeCamera;
}

Object *Camera::GetEditorCamera() {
    return editorCamera;
}

Object *Camera::GetRenderingCamera() {
    return renderingCamera;
}

Object *Camera::GetPreviouslyActiveCamera() {
    return previouslyActiveCamera;
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

void Camera::Save(nlohmann::json &json) {
    Component::Save(json);

    /// TODO: add saving and loading rendering camera and active camera something with id idk
    json["ComponentType"] = "Camera";
    json["FOV"] = fov;
    json["zNear"] = zNear;
    json["zFar"] = zFar;
}

void Camera::Load(nlohmann::json &json) {
    Component::Load(json);

    fov = json["FOV"];
    zNear = json["zNear"];
    zFar = json["zFar"];

    OnUpdate();
}
