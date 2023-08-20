#include "Rendering/ShadowRenderer.h"
#include "Managers/RenderingManager.h"
#include "Managers/ResourceManager.h"
#include "Rendering/ObjectRenderer.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Components/Rendering/Camera.h"
#include "Components/Rendering/Renderer.h"
#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Components/Rendering/Lights/PointLight.h"
#include "Components/Rendering/Lights/Spotlight.h"
#include "Resources/Shader.h"

ShadowRenderer::ShadowRenderer() {
    shadowShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/ShadowMapShader.json");

    glGenFramebuffers(1, &depthMapFBO);
    // create depth texture
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadowResolution, shadowResolution, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    // attach depth texture as FBO's depth buffer
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

ShadowRenderer::~ShadowRenderer() {
    shadowShader->Delete();

    ResourceManager::UnloadResource(shadowShader->GetPath());
}

void ShadowRenderer::PrepareShadowMap() {
    glm::mat4 lightProjection, lightView;

    float nearPlane = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetZNear();
    float farPlane = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetZFar();

    glViewport(0, 0, shadowResolution, shadowResolution);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);

    // render scene from light's point of view
    shadowShader->Activate();

    for (auto& light : RenderingManager::GetInstance()->objectRenderer->directionalLights) {
        if (!light) continue;

        Transform* lightTransform = light->parent->transform;
        lightProjection = glm::ortho(-100.0f, 100.0f, -100.0f, 100.0f, nearPlane, farPlane);

        lightView = glm::lookAt(lightTransform->GetGlobalPosition(), lightTransform->GetForward(), lightTransform->GetUp());
        lightSpaceMatrix = lightProjection * lightView;

        shadowShader->SetMat4("lightSpaceMatrix", lightSpaceMatrix);

        for (int i = 0; i < RenderingManager::GetInstance()->GetDrawBuffer().size(); ++i) {
            Renderer* renderer = RenderingManager::GetInstance()->GetDrawBuffer()[i];
            if (renderer->drawShadows) renderer->Draw(shadowShader);
        }
    }

    for (auto& light : RenderingManager::GetInstance()->objectRenderer->pointLights) {
        if (!light) continue;

        Transform* lightTransform = light->parent->transform;
        lightProjection = glm::ortho(-100.0f, 100.0f, -100.0f, 100.0f, nearPlane, farPlane);

        lightView = glm::lookAt(lightTransform->GetGlobalPosition(), lightTransform->GetForward(), lightTransform->GetUp());
        lightSpaceMatrix = lightProjection * lightView;

        shadowShader->SetMat4("lightSpaceMatrix", lightSpaceMatrix);

        for (int i = 0; i < RenderingManager::GetInstance()->GetDrawBuffer().size(); ++i) {
            Renderer* renderer = RenderingManager::GetInstance()->GetDrawBuffer()[i];
            if (renderer->drawShadows) renderer->Draw(shadowShader);
        }
    }

    for (auto& light : RenderingManager::GetInstance()->objectRenderer->spotLights) {
        if (!light) continue;

        Transform* lightTransform = light->parent->transform;
        Camera* camera = Camera::GetActiveCamera()->GetComponentByClass<Camera>();
        lightProjection = glm::perspective(std::acos(light->GetOuterCutOff()), 1.0f, camera->GetZNear(), camera->GetZFar());

        lightView = glm::lookAt(lightTransform->GetGlobalPosition(), lightTransform->GetGlobalPosition() + lightTransform->GetForward(), lightTransform->GetUp());
        lightSpaceMatrix = lightProjection * lightView;

        shadowShader->SetMat4("lightSpaceMatrix", lightSpaceMatrix);

        for (int i = 0; i < RenderingManager::GetInstance()->GetDrawBuffer().size(); ++i) {
            Renderer* renderer = RenderingManager::GetInstance()->GetDrawBuffer()[i];
            if (renderer->drawShadows) renderer->Draw(shadowShader);
        }
    }

    glDisable(GL_CULL_FACE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
