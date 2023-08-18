#include "Components/Rendering/Skybox.h"
#include "Managers/ResourceManager.h"
#include "Managers/RenderingManager.h"
#include "Rendering/SkyboxRenderer.h"
#include "Core/Object.h"
#include "Resources/CubeMap.h"
#include "Resources/Shader.h"


Skybox::Skybox(Object *parent, int id) : Component(parent, id) {
    cubeMap = ResourceManager::LoadResource<CubeMap>("resources/Resources/CubeMap.json");

    SkyboxRenderer* skyboxRenderer = RenderingManager::GetInstance()->skyboxRenderer;

    if (skyboxRenderer != nullptr && !skyboxRenderer->GetActiveSkybox()) {
        SetActive();
    }
}

Skybox::~Skybox() = default;

void Skybox::OnDestroy() {
    Component::OnDestroy();
    SkyboxRenderer* skyboxRenderer = RenderingManager::GetInstance()->skyboxRenderer;
    if (parent == skyboxRenderer->GetActiveSkybox()) skyboxRenderer->SetActiveSkybox(nullptr);
    ResourceManager::UnloadResource(cubeMap->GetPath());
}

CubeMap *Skybox::GetCubeMap() const {
    return cubeMap;
}

void Skybox::SetActive() {
    RenderingManager::GetInstance()->skyboxRenderer->SetActiveSkybox(parent);
}
