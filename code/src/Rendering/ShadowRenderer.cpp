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
    dnslShadowShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/DNSLShadowMapShader.json");
    plShadowShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/PLShadowMapShader.json");

    for (int i = 0; i < 8; ++i) {
        glGenFramebuffers(1, &depthMapFBOs[i]);
        // create depth texture
        glGenTextures(1, &depthMaps[i]);
        glBindTexture(GL_TEXTURE_2D, depthMaps[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 4096, 4096, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        // attach depth texture as FBO's depth buffer
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBOs[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMaps[i], 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    for (int i = 8; i < 12; ++i) {
        glGenFramebuffers(1, &depthMapFBOs[i]);
        glGenTextures(1, &depthMaps[i]);
        glBindTexture(GL_TEXTURE_CUBE_MAP, depthMaps[i]);
        for (unsigned int j = 0; j < 6; ++j)
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, 0, GL_DEPTH_COMPONENT, 2048, 2048, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        // attach depth texture as FBO's depth buffer
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBOs[i]);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthMaps[i], 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

ShadowRenderer::~ShadowRenderer() {
    dnslShadowShader->Delete();
    plShadowShader->Delete();

    ResourceManager::UnloadResource(dnslShadowShader->GetPath());
    ResourceManager::UnloadResource(plShadowShader->GetPath());
}

void ShadowRenderer::PrepareShadowMap() {
    glm::mat4 lightProjection, lightView;

    float nearPlane = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetZNear();
    float farPlane = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetZFar();

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);

    glViewport(0, 0, 4096, 4096);

    // render scene from light's point of view
    dnslShadowShader->Activate();

    int depthMapIndex = 0;
    int lightIndex = 0;

    for (auto& light : RenderingManager::GetInstance()->objectRenderer->directionalLights) {
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBOs[depthMapIndex]);
        glClear(GL_DEPTH_BUFFER_BIT);

        if (light) {
            Transform* lightTransform = light->parent->transform;
            lightProjection = glm::ortho(-100.0f, 100.0f, -100.0f, 100.0f, nearPlane, farPlane);

            lightView = glm::lookAt(lightTransform->GetGlobalPosition(), lightTransform->GetGlobalPosition() + lightTransform->GetForward(), lightTransform->GetUp());
            directionalLightSpaceMatrices[lightIndex] = lightProjection * lightView;

            dnslShadowShader->SetMat4("lightSpaceMatrix", directionalLightSpaceMatrices[lightIndex]);

            for (int i = 0; i < RenderingManager::GetInstance()->GetDrawBuffer().size(); ++i) {
                Renderer* renderer = RenderingManager::GetInstance()->GetDrawBuffer()[i];
                if (renderer->drawShadows) renderer->Draw(dnslShadowShader);
            }
        }

        ++depthMapIndex;
        ++lightIndex;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    lightIndex = 0;

    for (auto& light : RenderingManager::GetInstance()->objectRenderer->spotLights) {
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBOs[depthMapIndex]);
        glClear(GL_DEPTH_BUFFER_BIT);

        if (light) {
            Transform* lightTransform = light->parent->transform;
            Camera* camera = Camera::GetActiveCamera()->GetComponentByClass<Camera>();
            lightProjection = glm::perspective(std::acos(light->GetOuterCutOff()), 1.0f, camera->GetZNear(), camera->GetZFar());

            lightView = glm::lookAt(lightTransform->GetGlobalPosition(), lightTransform->GetGlobalPosition() + lightTransform->GetForward(), lightTransform->GetUp());
            spotLightSpaceMatrices[lightIndex] = lightProjection * lightView;

            dnslShadowShader->SetMat4("lightSpaceMatrix", spotLightSpaceMatrices[lightIndex]);

            for (int i = 0; i < RenderingManager::GetInstance()->GetDrawBuffer().size(); ++i) {
                Renderer* renderer = RenderingManager::GetInstance()->GetDrawBuffer()[i];
                if (renderer->drawShadows) renderer->Draw(dnslShadowShader);
            }
        }

        ++depthMapIndex;
        ++lightIndex;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    glViewport(0, 0, 2048, 2048);
    lightIndex = 0;

    plShadowShader->Activate();

    for (auto& light : RenderingManager::GetInstance()->objectRenderer->pointLights) {
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBOs[depthMapIndex]);
        glClear(GL_DEPTH_BUFFER_BIT);

        if (light) {
            Transform* lightTransform = light->parent->transform;
            Camera* camera = Camera::GetActiveCamera()->GetComponentByClass<Camera>();
            lightProjection = glm::perspective(glm::radians(90.0f), 1.0f, camera->GetZNear(), camera->GetZFar());

            glm::vec3 lightPosition = lightTransform->GetGlobalPosition();

            pointLightPositions[lightIndex] = lightPosition;

            glm::mat4 shadowTransforms[6];
            shadowTransforms[0] = lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
            shadowTransforms[1] = lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
            shadowTransforms[2] = lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            shadowTransforms[3] = lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
            shadowTransforms[4] = lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
            shadowTransforms[5] = lightProjection * glm::lookAt(lightPosition, lightPosition + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f));

            for (int i = 0; i < 6; ++i) {
                plShadowShader->SetMat4("shadowMatrices[" + std::to_string(i) + "]", shadowTransforms[i]);
            }

            plShadowShader->SetFloat("farPlane", camera->GetZFar());
            plShadowShader->SetVec3("lightPosition", lightPosition);

            for (int i = 0; i < RenderingManager::GetInstance()->GetDrawBuffer().size(); ++i) {
                Renderer* renderer = RenderingManager::GetInstance()->GetDrawBuffer()[i];
                if (renderer->drawShadows) renderer->Draw(plShadowShader);
            }
        }

        ++depthMapIndex;
        ++lightIndex;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    glDisable(GL_CULL_FACE);
}
