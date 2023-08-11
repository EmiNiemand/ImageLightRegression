#include "Components/Rendering/Renderer.h"
#include "Managers/ResourceManager.h"
#include "Managers/RenderingManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Resources/Shader.h"
#include "Resources/Model.h"

Renderer::Renderer(Object *parent, int id) : Component(parent, id) {}

Renderer::~Renderer() = default;

void Renderer::OnDestroy() {
    Component::OnDestroy();
    ResourceManager::UnloadResource(model->GetPath());
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
    shader->SetVec2("texStrech", textScale);
    shader->SetVec3("material.color", material.color);
    shader->SetFloat("material.shininess", material.shininess);
    shader->SetFloat("material.reflection", material.reflection);
    shader->SetFloat("material.refraction", material.refraction);

    model->Draw(shader);
}

void Renderer::LoadModel(const std::string& path) {
    model = ResourceManager::LoadResource<Model>(path);
}

void Renderer::AddToDraw() {
    RenderingManager::GetInstance()->AddToDrawBuffer(this);
}
