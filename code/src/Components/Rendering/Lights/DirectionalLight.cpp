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

void DirectionalLight::Save(nlohmann::json &json) {
    Component::Save(json);

    json["ComponentType"] = "DirectionalLight";

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
}

void DirectionalLight::Load(nlohmann::json &json) {
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

    OnUpdate();
}

