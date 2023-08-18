#include "Components/Rendering/Lights/Spotlight.h"
#include "Managers/RenderingManager.h"
#include "Rendering/ObjectRenderer.h"

SpotLight::SpotLight(Object* parent, int id) : Component(parent, id) {}

SpotLight::~SpotLight() = default;

void SpotLight::OnCreate() {
    Component::OnCreate();

    auto& lights = RenderingManager::GetInstance()->objectRenderer->spotLights;

    for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
        if (lights[i] == nullptr) {
            lights[i] = this;
            RenderingManager::GetInstance()->objectRenderer->UpdateLight(id);
            break;
        }
    }

}

void SpotLight::OnDestroy() {
    Component::OnDestroy();
    RenderingManager::GetInstance()->objectRenderer->RemoveLight(id);
}

void SpotLight::OnUpdate() {
    Component::OnUpdate();
    RenderingManager::GetInstance()->objectRenderer->UpdateLight(id);
}

#pragma region Getters
float SpotLight::GetCutOff() const {
    return cutOff;
}

float SpotLight::GetOuterCutOff() const {
    return outerCutOff;
}

float SpotLight::GetConstant() const {
    return constant;
}

float SpotLight::GetLinear() const {
    return linear;
}

float SpotLight::GetQuadratic() const {
    return quadratic;
}

const glm::vec3 &SpotLight::GetAmbient() const {
    return ambient;
}

const glm::vec3 &SpotLight::GetDiffuse() const {
    return diffuse;
}

const glm::vec3 &SpotLight::GetSpecular() const {
    return specular;
}

const glm::vec3 &SpotLight::GetColor() const {
    return color;
}
#pragma endregion

#pragma region Setters
void SpotLight::SetCutOff(float inCutOff) {
    cutOff = inCutOff;
    OnUpdate();
}

void SpotLight::SetOuterCutOff(float inOuterCutOff) {
    outerCutOff = inOuterCutOff;
    OnUpdate();
}

void SpotLight::SetConstant(float inConstant) {
    constant = inConstant;
    OnUpdate();
}

void SpotLight::SetLinear(float inLinear) {
    linear = inLinear;
    OnUpdate();
}

void SpotLight::SetQuadratic(float inQuadratic) {
    SpotLight::quadratic = inQuadratic;
    OnUpdate();
}

void SpotLight::SetAmbient(const glm::vec3 &inAmbient) {
    SpotLight::ambient = inAmbient;
    OnUpdate();
}

void SpotLight::SetDiffuse(const glm::vec3 &inDiffuse) {
    SpotLight::diffuse = inDiffuse;
    OnUpdate();
}

void SpotLight::SetSpecular(const glm::vec3 &inSpecular) {
    SpotLight::specular = inSpecular;
    OnUpdate();
}


void SpotLight::SetColor(const glm::vec3 &inColor) {
    SpotLight::color = inColor;
    OnUpdate();
}
#pragma endregion