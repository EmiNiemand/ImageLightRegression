#include "Components/Rendering/Renderer.h"
#include "Managers/ResourceManager.h"
#include "Managers/RenderingManager.h"
#include "Managers/EditorManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Resources/Shader.h"
#include "Resources/Model.h"

Renderer::Renderer(Object *parent, int id) : Component(parent, id) {}

Renderer::~Renderer() = default;

void Renderer::OnDestroy() {
    Component::OnDestroy();
    if (model) {
        ResourceManager::UnloadResource(model->GetPath());
    }
}

void Renderer::Update() {
    if (!enabled) return;
    Component::Update();

    AddToDraw();
}

void Renderer::Draw(Shader* shader) {
    if(model == nullptr) return;

    shader->Activate();
    shader->SetMat4("model", parent->transform->GetModelMatrix());
    shader->SetVec2("texStrech", texScale);
    shader->SetVec3("material.color", material.color);
    shader->SetFloat("material.shininess", material.shininess);
    shader->SetFloat("material.reflection", material.reflection);
    shader->SetFloat("material.refraction", material.refraction);

    model->Draw(shader);
}

void Renderer::LoadModel(const std::string& path) {
    if (model) ResourceManager::UnloadResource(model->GetPath());
    model = ResourceManager::LoadResource<Model>(path);
}

void Renderer::AddToDraw() {
    RenderingManager::GetInstance()->AddToDrawBuffer(this);
}

void Renderer::Save(nlohmann::json &json) {
    Component::Save(json);

    json["ComponentType"] = "Renderer";
    json["Material"] = nlohmann::json::array();
    json["Material"].push_back(material.color.x);
    json["Material"].push_back(material.color.y);
    json["Material"].push_back(material.color.z);
    json["Material"].push_back(material.shininess);
    json["Material"].push_back(material.reflection);
    json["Material"].push_back(material.refraction);

    json["TextureScale"] = nlohmann::json::array();
    json["TextureScale"].push_back(texScale.x);
    json["TextureScale"].push_back(texScale.y);

    json["Model"] = model->GetPath();
    json["DrawShadows"] = drawShadows;
}

void Renderer::Load(nlohmann::json &json) {
    Component::Load(json);

    material.color.x = json["Material"][0];
    material.color.y = json["Material"][1];
    material.color.z = json["Material"][2];
    material.shininess = json["Material"][3];
    material.reflection = json["Material"][4];
    material.refraction = json["Material"][5];

    texScale.x = json["TextureScale"][0];
    texScale.y = json["TextureScale"][1];

    LoadModel(json["Model"]);
    drawShadows = json["DrawShadows"];
}
