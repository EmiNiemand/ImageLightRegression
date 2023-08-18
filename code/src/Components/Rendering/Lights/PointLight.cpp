#include "Components/Rendering/Lights/PointLight.h"
#include "Managers/RenderingManager.h"
#include "Rendering/ObjectRenderer.h"

PointLight::PointLight(Object* parent, int id) : Component(parent, id) {}

PointLight::~PointLight() = default;

void PointLight::OnCreate() {
    Component::OnCreate();

    auto& lights = RenderingManager::GetInstance()->objectRenderer->pointLights;

    for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
        if (lights[i] == nullptr) {
            lights[i] = this;
            RenderingManager::GetInstance()->objectRenderer->UpdateLight(id);
            break;
        }
    }

}

void PointLight::OnDestroy() {
    Component::OnDestroy();
    RenderingManager::GetInstance()->objectRenderer->RemoveLight(id);
}

void PointLight::OnUpdate() {
    Component::OnUpdate();
    RenderingManager::GetInstance()->objectRenderer->UpdateLight(id);
}

#pragma region Getters
float PointLight::GetConstant() const {
    return constant;
}

float PointLight::GetLinear() const {
    return linear;
}

float PointLight::GetQuadratic() const {
    return quadratic;
}

const glm::vec3 &PointLight::GetAmbient() const {
    return ambient;
}

const glm::vec3 &PointLight::GetDiffuse() const {
    return diffuse;
}

const glm::vec3 &PointLight::GetSpecular() const {
    return specular;
}

const glm::vec3 &PointLight::GetColor() const {
    return color;
}
#pragma endregion

#pragma region Setters
void PointLight::SetConstant(float inConstant) {
    PointLight::constant = inConstant;
    OnUpdate();
}

void PointLight::SetLinear(float inLinear) {
    PointLight::linear = inLinear;
    OnUpdate();
}

void PointLight::SetQuadratic(float inQuadratic) {
    PointLight::quadratic = inQuadratic;
    OnUpdate();
}

void PointLight::SetAmbient(const glm::vec3 &inAmbient) {
    PointLight::ambient = inAmbient;
    OnUpdate();
}

void PointLight::SetDiffuse(const glm::vec3 &inDiffuse) {
    PointLight::diffuse = inDiffuse;
    OnUpdate();
}

void PointLight::SetSpecular(const glm::vec3 &inSpecular) {
    PointLight::specular = inSpecular;
    OnUpdate();
}

void PointLight::SetColor(const glm::vec3 &inColor) {
    PointLight::color = inColor;
    OnUpdate();
}
#pragma endregion