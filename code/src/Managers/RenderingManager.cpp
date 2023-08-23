#include "Managers/RenderingManager.h"
#include "Managers/ResourceManager.h"
#include "Rendering/ShadowRenderer.h"
#include "Rendering/ObjectRenderer.h"
#include "Rendering/SkyboxRenderer.h"
#include "Rendering/UIRenderer.h"
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
    uiRenderer = new UIRenderer();
}

void RenderingManager::Shutdown() {
    delete shadowRenderer;
    delete objectRenderer;
    delete skyboxRenderer;
    delete uiRenderer;
    delete renderingManager;
}

void RenderingManager::Draw(Shader* inShader) {
    inShader->Activate();
    for (int i = 0; i < 4; ++i) {
        glActiveTexture(GL_TEXTURE5 + i);
        glBindTexture(GL_TEXTURE_2D, RenderingManager::GetInstance()->shadowRenderer->depthMaps[i]);
        inShader->SetMat4("directionalLightSpaceMatrices[" + std::to_string(i) + "]",
                          RenderingManager::GetInstance()->shadowRenderer->directionalLightSpaceMatrices[i]);


    }
    for (int i = 4; i < 8; ++i) {
        glActiveTexture(GL_TEXTURE5 + i);
        glBindTexture(GL_TEXTURE_2D, RenderingManager::GetInstance()->shadowRenderer->depthMaps[i]);
        inShader->SetMat4("spotLightSpaceMatrices[" + std::to_string(i - 4) + "]",
                          RenderingManager::GetInstance()->shadowRenderer->spotLightSpaceMatrices[i - 4]);


    }
    for (int i = 8; i < 12; ++i) {
        glActiveTexture(GL_TEXTURE5 + i);
        glBindTexture(GL_TEXTURE_CUBE_MAP, RenderingManager::GetInstance()->shadowRenderer->depthMaps[i]);
        inShader->SetVec3("pointLightPositions[" + std::to_string(i - 8) + "]",
                          RenderingManager::GetInstance()->shadowRenderer->pointLightPositions[i - 8]);
    }

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
    objectRenderer->shader->SetFloat("farPlane", Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetZFar());

    skyboxRenderer->cubeMapShader->Activate();
    skyboxRenderer->cubeMapShader->SetMat4("projection", projection);

    uiRenderer->imageShader->Activate();
    uiRenderer->imageShader->SetMat4("projection", projection);
}

void RenderingManager::UpdateView() const {
    glm::mat4 view = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetViewMatrix();

    objectRenderer->shader->Activate();
    objectRenderer->shader->SetMat4("view", view);
    objectRenderer->shader->SetVec3("viewPosition", Camera::GetActiveCamera()->transform->GetGlobalPosition());

    skyboxRenderer->cubeMapShader->Activate();
    skyboxRenderer->cubeMapShader->SetMat4("view", view);
}
