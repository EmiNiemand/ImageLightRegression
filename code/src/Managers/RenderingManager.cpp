#include "Managers/RenderingManager.h"
#include "Managers/ResourceManager.h"
#include "Managers/EditorManager.h"
#include "Rendering/ShadowRenderer.h"
#include "Rendering/ObjectRenderer.h"
#include "Rendering/SkyboxRenderer.h"
#include "Rendering/UIRenderer.h"
#include "Rendering/PostProcessRenderer.h"
#include "Resources/Shader.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Components/Rendering/Renderer.h"
#include "Components/Rendering/Camera.h"
#include "Components/Rendering/Skybox.h"
#include "Components/Rendering/UI/Image.h"

RenderingManager::RenderingManager() = default;

RenderingManager::~RenderingManager() = default;

RenderingManager* RenderingManager::GetInstance() {
    if (renderingManager == nullptr) {
        renderingManager = new RenderingManager();
    }
    return renderingManager;
}

void RenderingManager::Startup() {
    selectedObjectShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/SelectedObjectShader.json");

    shadowRenderer = new ShadowRenderer();
    objectRenderer = new ObjectRenderer();
    skyboxRenderer = new SkyboxRenderer();
    uiRenderer = new UIRenderer();
    postProcessRenderer = new PostProcessRenderer();
}

void RenderingManager::Shutdown() {
    delete shadowRenderer;
    delete objectRenderer;
    delete skyboxRenderer;
    delete uiRenderer;
    delete postProcessRenderer;

    selectedObjectShader->Delete();
    ResourceManager::UnloadResource(selectedObjectShader->GetPath());

    delete renderingManager;
}

void RenderingManager::Draw(Shader* inShader) {
    inShader->Activate();

    for (int i = 0; i < drawBuffer.size(); ++i) {
        drawBuffer[i]->Draw(inShader);
    }
}

void RenderingManager::DrawFrame() {
    RenderingManager::GetInstance()->shadowRenderer->PrepareShadowMap();

    glBindFramebuffer(GL_FRAMEBUFFER, objectRenderer->fbo);
    glViewport(0, 0, Application::viewports[0].resolution.x, Application::viewports[0].resolution.y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Shader* shader = objectRenderer->shader;

    for (int i = 0; i < 4; ++i) {
        glActiveTexture(GL_TEXTURE5 + i);
        glBindTexture(GL_TEXTURE_2D, shadowRenderer->depthMaps[i]);
        shader->SetMat4("directionalLightSpaceMatrices[" + std::to_string(i) + "]",
                          shadowRenderer->directionalLightSpaceMatrices[i]);


    }
    for (int i = 4; i < 8; ++i) {
        glActiveTexture(GL_TEXTURE5 + i);
        glBindTexture(GL_TEXTURE_2D, shadowRenderer->depthMaps[i]);
        shader->SetMat4("spotLightSpaceMatrices[" + std::to_string(i - 4) + "]",
                        shadowRenderer->spotLightSpaceMatrices[i - 4]);


    }
    for (int i = 8; i < 12; ++i) {
        glActiveTexture(GL_TEXTURE5 + i);
        glBindTexture(GL_TEXTURE_CUBE_MAP, shadowRenderer->depthMaps[i]);
        shader->SetVec3("pointLightPositions[" + std::to_string(i - 8) + "]",
                        shadowRenderer->pointLightPositions[i - 8]);
    }
    skyboxRenderer->Draw();
    Draw(shader);


    glBindFramebuffer(GL_FRAMEBUFFER, objectRenderer->fbo2);
    glViewport(0, 0, Application::viewports[0].resolution.x, Application::viewports[0].resolution.y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Object* selectedObject = EditorManager::GetInstance()->selectedNode;
    if (selectedObject != nullptr) {
        Renderer* renderer = selectedObject->GetComponentByClass<Renderer>();
        Skybox* skybox = selectedObject->GetComponentByClass<Skybox>();
        if (renderer != nullptr) {
            selectedObjectShader->Activate();
            glDisable(GL_DEPTH_TEST);
            selectedObjectShader->SetInt("isSkybox", 0);
            renderer->Draw(selectedObjectShader);
            glEnable(GL_DEPTH_TEST);
        }
        else if (skybox != nullptr) {
            selectedObjectShader->Activate();
            selectedObjectShader->SetInt("isSkybox", 1);
            glDepthFunc(GL_LEQUAL);

            glBindVertexArray(skyboxRenderer->vao);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);

            glDepthFunc(GL_LESS);
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glViewport(Application::viewports[0].position.x, Application::viewports[0].position.y, Application::viewports[0].resolution.x, Application::viewports[0].resolution.y);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    postProcessRenderer->postProcessShader->Activate();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, objectRenderer->screenTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, objectRenderer->selectedObjectTexture);

    postProcessRenderer->postProcessShader->SetIVec2("screenPosition", Application::viewports[0].position);
    postProcessRenderer->Draw();

    glEnable(GL_DEPTH_TEST);

    Application* application = Application::GetInstance();

    glViewport(Application::viewports[1].position.x, Application::viewports[1].position.y,
               Application::viewports[1].resolution.x, Application::viewports[1].resolution.y);
    application->loadedImage->GetComponentByClass<Image>()->Draw(uiRenderer->imageShader);

    if (application->isStarted) {
        glViewport(Application::viewports[2].position.x, Application::viewports[2].position.y,
                   Application::viewports[2].resolution.x, Application::viewports[2].resolution.y);
        application->renderedImage->GetComponentByClass<Image>()->Draw(uiRenderer->imageShader);

        glViewport(Application::viewports[3].position.x, Application::viewports[3].position.y,
                   Application::viewports[3].resolution.x, Application::viewports[3].resolution.y);
        application->differenceImage->GetComponentByClass<Image>()->Draw(uiRenderer->imageShader);
    }

    RenderingManager::GetInstance()->ClearBuffer();
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

    selectedObjectShader->Activate();
    selectedObjectShader->SetMat4("projection", projection);
}

void RenderingManager::UpdateView() const {
    glm::mat4 view = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetViewMatrix();

    objectRenderer->shader->Activate();
    objectRenderer->shader->SetMat4("view", view);
    objectRenderer->shader->SetVec3("viewPosition", Camera::GetActiveCamera()->transform->GetGlobalPosition());

    skyboxRenderer->cubeMapShader->Activate();
    skyboxRenderer->cubeMapShader->SetMat4("view", view);

    selectedObjectShader->Activate();
    selectedObjectShader->SetMat4("view", view);
}

void RenderingManager::OnWindowResize() const {
    objectRenderer->PrepareBuffers();
}
