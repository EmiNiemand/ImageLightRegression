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

void PointLight::Save(nlohmann::json &json) {
    Component::Save(json);

    json["ComponentType"] = "PointLight";

    json["Ambient"] = nlohmann::json::array();
    json["Ambient"].push_back(ambient.x);
    json["Ambient"].push_back(ambient.y);
    json["Ambient"].push_back(ambient.z);

    json["Diffuse"] = nlohmann::json::array();
    json["Diffuse"].push_back(diffuse.x);
    json["Diffuse"].push_back(diffuse.y);
    json["Diffuse"].push_back(diffuse.z);

    json["Specular"] = nlohmann::json::array();
    json["Specular"].push_back(specular.x);
    json["Specular"].push_back(specular.y);
    json["Specular"].push_back(specular.z);

    json["Color"] = nlohmann::json::array();
    json["Color"].push_back(color.x);
    json["Color"].push_back(color.y);
    json["Color"].push_back(color.z);

    json["Constant"] = constant;
    json["Linear"] = linear;
    json["Quadratic"] = quadratic;
}

void PointLight::Load(nlohmann::json &json) {
    Component::Load(json);

    ambient.x = json["Ambient"][0];
    ambient.y = json["Ambient"][1];
    ambient.z = json["Ambient"][2];

    diffuse.x = json["Diffuse"][0];
    diffuse.y = json["Diffuse"][1];
    diffuse.z = json["Diffuse"][2];

    specular.x = json["Specular"][0];
    specular.y = json["Specular"][1];
    specular.z = json["Specular"][2];

    color.x = json["Color"][0];
    color.y = json["Color"][1];
    color.z = json["Color"][2];

    constant = json["Constant"];
    linear = json["Linear"];
    quadratic = json["Quadratic"];

    OnUpdate();
}

#pragma endregion