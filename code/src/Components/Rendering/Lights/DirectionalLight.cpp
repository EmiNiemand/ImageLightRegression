#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Managers/RenderingManager.h"
#include "Rendering/ObjectRenderer.h"

DirectionalLight::DirectionalLight(Object* parent, int id) : Component(parent, id) {}

DirectionalLight::~DirectionalLight() = default;

void DirectionalLight::OnCreate() {
    Component::OnCreate();

    auto& lights = RenderingManager::GetInstance()->objectRenderer->directionalLights;

    for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
        if (lights[i] == nullptr) {
            lights[i] = this;
            RenderingManager::GetInstance()->objectRenderer->UpdateLight(id);
            break;
        }
    }

}

void DirectionalLight::OnDestroy() {
    Component::OnDestroy();
    RenderingManager::GetInstance()->objectRenderer->RemoveLight(id);
}

void DirectionalLight::OnUpdate() {
    Component::OnUpdate();
    RenderingManager::GetInstance()->objectRenderer->UpdateLight(id);
}

#pragma region Getters
const glm::vec3 &DirectionalLight::GetAmbient() const {
    return ambient;
}

const glm::vec3 &DirectionalLight::GetDiffuse() const {
    return diffuse;
}

const glm::vec3 &DirectionalLight::GetSpecular() const {
    return specular;
}

const glm::vec3 &DirectionalLight::GetColor() const {
    return color;
}
#pragma endregion

#pragma region Setters
void DirectionalLight::SetAmbient(const glm::vec3 &inAmbient) {
    DirectionalLight::ambient = inAmbient;
    OnUpdate();
}

void DirectionalLight::SetDiffuse(const glm::vec3 &inDiffuse) {
    DirectionalLight::diffuse = inDiffuse;
    OnUpdate();
}

void DirectionalLight::SetSpecular(const glm::vec3 &inSpecular) {
    DirectionalLight::specular = inSpecular;
    OnUpdate();
}

void DirectionalLight::SetColor(const glm::vec3 &inColor) {
    DirectionalLight::color = inColor;
    OnUpdate();
}
#pragma endregion