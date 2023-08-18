#include "Managers/RenderingManager.h"
#include "Managers/ResourceManager.h"
#include "Rendering/ShadowRenderer.h"
#include "Rendering/ObjectRenderer.h"
#include "Rendering/SkyboxRenderer.h"
#include "Resources/Shader.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Components/Rendering/Renderer.h"
#include "Components/Rendering/Camera.h"

RenderingManager::RenderingManager() = default;

RenderingManager::~RenderingManager() = default;

RenderingManager* RenderingManager::GetInstance() {
    if (renderingManager == nullptr) {
        renderingManager = new RenderingManager();
    }
    return renderingManager;
}

void RenderingManager::Startup() {
    shadowRenderer = new ShadowRenderer();
    objectRenderer = new ObjectRenderer();
    skyboxRenderer = new SkyboxRenderer();
}

void RenderingManager::Shutdown() {
    delete shadowRenderer;
    delete objectRenderer;
    delete skyboxRenderer;
    delete renderingManager;
}

void RenderingManager::Draw(Shader* inShader) {
    inShader->Activate();
    inShader->SetMat4("lightSpaceMatrix", RenderingManager::GetInstance()->shadowRenderer->lightSpaceMatrix);

    for (int i = 0; i < drawBuffer.size(); ++i) {
        drawBuffer[i]->Draw(inShader);
    }
}

void RenderingManager::ClearBuffer() {
    drawBuffer.clear();
}

void RenderingManager::AddToDrawBuffer(Renderer* renderer) {
    drawBuffer.push_back(renderer);
}

const std::vector<Renderer*>& RenderingManager::GetDrawBuffer() const {
    return drawBuffer;
}

void RenderingManager::UpdateProjection() const {
    glm::mat4 projection = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetProjectionMatrix();

    objectRenderer->shader->Activate();
    objectRenderer->shader->SetMat4("projection", projection);

    skyboxRenderer->cubeMapShader->Activate();
    skyboxRenderer->cubeMapShader->SetMat4("projection", projection);
}

void RenderingManager::UpdateView() const {
    glm::mat4 view = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetViewMatrix();

    objectRenderer->shader->Activate();
    objectRenderer->shader->SetMat4("view", view);
    objectRenderer->shader->SetVec3("viewPosition", Camera::GetActiveCamera()->transform->GetGlobalPosition());

    skyboxRenderer->cubeMapShader->Activate();
    skyboxRenderer->cubeMapShader->SetMat4("view", view);
}
