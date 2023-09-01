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

void SpotLight::Save(nlohmann::json &json) {
    Component::Save(json);

    json["ComponentType"] = "SpotLight";

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

    json["CutOff"] = cutOff;
    json["OuterCutOff"] = outerCutOff;

    json["Constant"] = constant;
    json["Linear"] = linear;
    json["Quadratic"] = quadratic;
}

void SpotLight::Load(nlohmann::json &json) {
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

    cutOff = json["CutOff"];
    outerCutOff = json["OuterCutOff"];

    constant = json["Constant"];
    linear = json["Linear"];
    quadratic = json["Quadratic"];

    OnUpdate();
}

#pragma endregion