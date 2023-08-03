#include "Components/Rendering/Lights/PointLight.h"
#include "Managers/RenderingManager.h"

PointLight::PointLight(Object* parent, int id) : Component(parent, id) {}

PointLight::~PointLight() = default;

void PointLight::OnCreate() {
    Component::OnCreate();

    bool isAdded = false;
    int number = 0;

    for (auto& pointLight : RenderingManager::GetInstance()->pointLights) {
        if (pointLight.second == nullptr) {
            RenderingManager::GetInstance()->pointLights.find(number)->second = this;
            isAdded = true;
        }
        number++;
    }
    if (!isAdded) RenderingManager::GetInstance()->pointLights.insert({number, this});
    RenderingManager::GetInstance()->UpdateLight(id);
}

void PointLight::OnDestroy() {
    Component::OnDestroy();
    RenderingManager::GetInstance()->RemoveLight(id);
}

void PointLight::OnUpdate() {
    Component::OnUpdate();
    RenderingManager::GetInstance()->UpdateLight(id);
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