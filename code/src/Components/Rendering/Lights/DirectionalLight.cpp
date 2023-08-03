#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Managers/RenderingManager.h"

DirectionalLight::DirectionalLight(Object* parent, int id) : Component(parent, id) {}

DirectionalLight::~DirectionalLight() = default;

void DirectionalLight::OnCreate() {
    Component::OnCreate();

    bool isAdded = false;
    int number = 0;

    for (auto&& directionalLight : RenderingManager::GetInstance()->directionalLights) {
        if (directionalLight.second == nullptr) {
            RenderingManager::GetInstance()->directionalLights.find(number)->second = this;
            isAdded = true;
        }
        number++;
    }
    if (!isAdded) RenderingManager::GetInstance()->directionalLights.insert({number, this});
    RenderingManager::GetInstance()->UpdateLight(id);
}

void DirectionalLight::OnDestroy() {
    Component::OnDestroy();
    RenderingManager::GetInstance()->RemoveLight(id);
}

void DirectionalLight::OnUpdate() {
    Component::OnUpdate();
    RenderingManager::GetInstance()->UpdateLight(id);
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