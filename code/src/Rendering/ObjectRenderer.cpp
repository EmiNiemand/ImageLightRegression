#include "Rendering/ObjectRenderer.h"
#include "Managers/ResourceManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Components/Rendering/Lights/PointLight.h"
#include "Components/Rendering/Lights/Spotlight.h"
#include "Resources/Shader.h"
#include "Application.h"

ObjectRenderer::ObjectRenderer() {
    shader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/BasicShader.json");

    PrepareBuffers();

    shader->Activate();
    shader->SetInt("cubeMapTexture", 4);

    shader->Activate();
    for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
        shader->SetInt("directionalLightShadowMapTexture[" + std::to_string(i) + "]", 5 + i);
    }
    for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
        shader->SetInt("spotLightShadowMapTexture[" + std::to_string(i) + "]", 9 + i);
    }
    for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
        shader->SetInt("pointLightShadowMapTexture[" + std::to_string(i) + "]", 13 + i);
    }
}

ObjectRenderer::~ObjectRenderer() {
    shader->Delete();

    glDeleteRenderbuffers(1, &rbo);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo2);
    glDeleteFramebuffers(1, &fbo2);
    glDeleteRenderbuffers(1, &rbo3);
    glDeleteFramebuffers(1, &fbo3);
    ResourceManager::UnloadResource(shader->GetPath());
}

void ObjectRenderer::PrepareBuffers() {
    int width = Application::viewports[0].resolution.x;
    int height = Application::viewports[0].resolution.y;

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    // create a color attachment texture
    glGenTextures(1, &screenTexture);
    glBindTexture(GL_TEXTURE_2D, screenTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screenTexture, 0);

    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        spdlog::error("ERROR::FRAMEBUFFER:: Framebuffer is not complete!");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenFramebuffers(1, &fbo2);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo2);
    // create a color attachment texture
    glGenTextures(1, &selectedObjectTexture);
    glBindTexture(GL_TEXTURE_2D, selectedObjectTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, selectedObjectTexture, 0);

    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    glGenRenderbuffers(1, &rbo2);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo2);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo2);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        spdlog::error("ERROR::FRAMEBUFFER:: Framebuffer is not complete!");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenFramebuffers(1, &fbo3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo3);
    // create a color attachment texture
    glGenTextures(1, &renderingCameraTexture);
    glBindTexture(GL_TEXTURE_2D, renderingCameraTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderingCameraTexture, 0);

    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    glGenRenderbuffers(1, &rbo3);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo3);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo3);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        spdlog::error("ERROR::FRAMEBUFFER:: Framebuffer is not complete!");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ObjectRenderer::UpdateLight(int componentId) {
    Component* light = Application::GetInstance()->components.at(componentId);

    if (dynamic_cast<DirectionalLight*>(light) != nullptr) {
        for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
            if (directionalLights[i] != nullptr && directionalLights[i]->id == componentId) {
                UpdateDirectionalLight(i, shader);
                return;
            }
        }
    }
    else if (dynamic_cast<PointLight*>(light) != nullptr) {
        for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
            if (pointLights[i] != nullptr && pointLights[i]->id == componentId) {
                UpdatePointLight(i, shader);
                return;
            }
        }
    }
    else if (dynamic_cast<SpotLight*>(light) != nullptr) {
        for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
            if (spotLights[i] != nullptr && spotLights[i]->id == componentId) {
                UpdateSpotLight(i, shader);
                return;
            }
        }
    }
}

void ObjectRenderer::RemoveLight(int componentId) {
    Component* light = Application::GetInstance()->components.at(componentId);

    if (dynamic_cast<DirectionalLight*>(light) != nullptr) {
        for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
            if (directionalLights[i] != nullptr && directionalLights[i]->id == componentId) {
                RemoveDirectionalLight(i, shader);
                return;
            }
        }
    }
    else if (dynamic_cast<PointLight*>(light) != nullptr) {
        for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
            if (pointLights[i] != nullptr && pointLights[i]->id == componentId) {
                RemovePointLight(i, shader);
                return;
            }
        }
    }
    else if (dynamic_cast<SpotLight*>(light) != nullptr) {
        for (int i = 0; i < NUMBER_OF_LIGHTS; ++i) {
            if (spotLights[i] != nullptr && spotLights[i]->id == componentId) {
                RemoveSpotLight(i, shader);
                return;
            }
        }
    }
}

void ObjectRenderer::UpdatePointLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    PointLight* pointLight = pointLights[lightNumber];
    std::string light = "pointLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light + ".isActive", pointLight->GetEnabled());
    lightShader->SetVec3(light + ".position", pointLight->parent->transform->GetLocalPosition());
    lightShader->SetFloat(light + ".constant", pointLight->GetConstant());
    lightShader->SetFloat(light + ".linear", pointLight->GetLinear());
    lightShader->SetFloat(light + ".quadratic", pointLight->GetQuadratic());
    lightShader->SetVec3(light + ".ambient", pointLight->GetAmbient());
    lightShader->SetVec3(light + ".diffuse", pointLight->GetDiffuse());
    lightShader->SetVec3(light + ".specular", pointLight->GetSpecular());
    lightShader->SetVec3(light + ".color", pointLight->GetColor());
}

void ObjectRenderer::UpdateDirectionalLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    DirectionalLight* directionalLight = directionalLights[lightNumber];
    std::string light = "directionalLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light + ".isActive", directionalLight->GetEnabled());
    lightShader->SetVec3(light + ".direction", directionalLight->parent->transform->GetForward());
    lightShader->SetVec3(light + ".ambient", directionalLight->GetAmbient());
    lightShader->SetVec3(light + ".diffuse", directionalLight->GetDiffuse());
    lightShader->SetVec3(light + ".specular", directionalLight->GetSpecular());
    lightShader->SetVec3(light + ".color", directionalLight->GetColor());
}

void ObjectRenderer::UpdateSpotLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    SpotLight* spotLight = spotLights[lightNumber];
    std::string light = "spotLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light + ".isActive", spotLight->GetEnabled());
    lightShader->SetVec3(light + ".position", spotLight->parent->transform->GetLocalPosition());
    lightShader->SetVec3(light + ".direction", spotLight->parent->transform->GetForward());
    lightShader->SetFloat(light + ".cutOff", spotLight->GetCutOff());
    lightShader->SetFloat(light + ".outerCutOff", spotLight->GetOuterCutOff());
    lightShader->SetFloat(light + ".constant", spotLight->GetConstant());
    lightShader->SetFloat(light + ".linear", spotLight->GetLinear());
    lightShader->SetFloat(light + ".quadratic", spotLight->GetQuadratic());
    lightShader->SetVec3(light + ".ambient", spotLight->GetAmbient());
    lightShader->SetVec3(light + ".diffuse", spotLight->GetDiffuse());
    lightShader->SetVec3(light + ".specular", spotLight->GetSpecular());
    lightShader->SetVec3(light + ".color", spotLight->GetColor());
}

void ObjectRenderer::RemovePointLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    std::string light = "pointLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light + ".isActive", false);
    lightShader->SetVec3(light + ".position", {0, 0, 0});
    lightShader->SetFloat(light + ".constant", 0);
    lightShader->SetFloat(light + ".linear", 0);
    lightShader->SetFloat(light + ".quadratic", 0);
    lightShader->SetVec3(light + ".ambient", {0, 0, 0});
    lightShader->SetVec3(light + ".diffuse", {0, 0, 0});
    lightShader->SetVec3(light + ".specular", {0, 0, 0});
    lightShader->SetVec3(light + ".color", {0, 0, 0});

    pointLights[lightNumber] = nullptr;
}

void ObjectRenderer::RemoveDirectionalLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    std::string light = "directionalLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light +".isActive", false);
    lightShader->SetVec3(light + ".direction", {0, 0, 0});
    lightShader->SetVec3(light + ".ambient", {0, 0, 0});
    lightShader->SetVec3(light + ".diffuse", {0, 0, 0});
    lightShader->SetVec3(light + ".specular", {0, 0, 0});
    lightShader->SetVec3(light + ".color", {0, 0, 0});

    directionalLights[lightNumber] = nullptr;
}

void ObjectRenderer::RemoveSpotLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    std::string light = "spotLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light +".isActive", false);
    lightShader->SetVec3(light + ".position", {0, 0, 0});
    lightShader->SetVec3(light + ".direction", {0, 0, 0});
    lightShader->SetFloat(light + ".cutOff", 0);
    lightShader->SetFloat(light + ".outerCutOff", 0);
    lightShader->SetFloat(light + ".constant", 0);
    lightShader->SetFloat(light + ".linear", 0);
    lightShader->SetFloat(light + ".quadratic", 0);
    lightShader->SetVec3(light + ".ambient", {0, 0, 0});
    lightShader->SetVec3(light + ".diffuse", {0, 0, 0});
    lightShader->SetVec3(light + ".specular", {0, 0, 0});
    lightShader->SetVec3(light + ".color", {0, 0, 0});

    spotLights[lightNumber] = nullptr;
}
