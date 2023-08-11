#include "Managers/RenderingManager.h"
#include "Managers/ResourceManager.h"
#include "Resources/Shader.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Components/Rendering/Renderer.h"
#include "Components/Rendering/Camera.h"
#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Components/Rendering/Lights/PointLight.h"
#include "Components/Rendering/Lights/Spotlight.h"

RenderingManager::RenderingManager() {
    shader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/BasicShader.json");
    cubeMapShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/CubeMapShader.json");
}

RenderingManager::~RenderingManager() {
    delete renderingManager;
}

RenderingManager* RenderingManager::GetInstance() {
    if (renderingManager == nullptr) {
        renderingManager = new RenderingManager();
    }
    return renderingManager;
}

void RenderingManager::Shutdown() const {
    shader->Delete();
    cubeMapShader->Delete();
}

void RenderingManager::Draw(Shader* inShader) {
    for (int i = 0; i < bufferIterator; ++i) {
        drawBuffer[i]->Draw(inShader);
    }
    ClearBuffer();
}

void RenderingManager::AddToDrawBuffer(Renderer* renderer) {
    drawBuffer[bufferIterator] = renderer;
    ++bufferIterator;
}

void RenderingManager::UpdateProjection() const {
    glm::mat4 projection = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetProjectionMatrix();

    shader->Activate();
    shader->SetMat4("projection", projection);

    cubeMapShader->Activate();
    cubeMapShader->SetMat4("projection", projection);
}

void RenderingManager::UpdateView() const {
    glm::mat4 view = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetViewMatrix();

    shader->Activate();
    shader->SetMat4("view", view);
    shader->SetVec3("viewPos", Camera::GetActiveCamera()->transform->GetGlobalPosition());

    cubeMapShader->Activate();
    cubeMapShader->SetMat4("view", view);
}

void RenderingManager::UpdateLight(int componentId) {

    for (int i = 0; i < spotLights.size(); ++i) {
        if (spotLights.at(i) != nullptr && spotLights.at(i)->id == componentId) {
            UpdateSpotLight(i, shader);
            return;
        }
    }
    for (int i = 0; i < directionalLights.size(); ++i) {
        if (directionalLights.at(i) != nullptr && directionalLights.at(i)->id == componentId) {
            UpdateDirectionalLight(i, shader);
            return;
        }
    }
    for (int i = 0; i < pointLights.size(); ++i) {
        if (pointLights.at(i) != nullptr && pointLights.at(i)->id == componentId) {
            UpdatePointLight(i, shader);
            return;
        }
    }
}

void RenderingManager::RemoveLight(int componentId) {
    for (int i = 0; i < spotLights.size(); ++i) {
        if (spotLights.at(i) != nullptr && spotLights.at(i)->id == componentId) {
            RemoveSpotLight(i, shader);
            return;
        }
    }
    for (int i = 0; i < directionalLights.size(); ++i) {
        if (directionalLights.at(i) != nullptr && directionalLights.at(i)->id == componentId) {
            RemoveDirectionalLight(i, shader);
            return;
        }
    }
    for (int i = 0; i < pointLights.size(); ++i) {
        if (pointLights.at(i) != nullptr && pointLights.at(i)->id == componentId) {
            RemovePointLight(i, shader);
            return;
        }
    }
}

void RenderingManager::UpdatePointLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    PointLight* pointLight = pointLights.find(lightNumber)->second;
    std::string light = "pointLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light + ".isActive", pointLight->enabled);
    lightShader->SetVec3(light + ".position", pointLight->parent->transform->GetLocalPosition());
    lightShader->SetFloat(light + ".constant", pointLight->GetConstant());
    lightShader->SetFloat(light + ".linear", pointLight->GetLinear());
    lightShader->SetFloat(light + ".quadratic", pointLight->GetQuadratic());
    lightShader->SetVec3(light + ".ambient", pointLight->GetAmbient());
    lightShader->SetVec3(light + ".diffuse", pointLight->GetDiffuse());
    lightShader->SetVec3(light + ".specular", pointLight->GetSpecular());
    lightShader->SetVec3(light + ".color", pointLight->GetColor());
}

void RenderingManager::UpdateDirectionalLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    DirectionalLight* directionalLight = directionalLights.find(lightNumber)->second;
    std::string light = "directionalLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light + ".isActive", directionalLight->enabled);
    lightShader->SetVec3(light + ".direction", directionalLight->parent->transform->GetForward());
    lightShader->SetVec3(light + ".ambient", directionalLight->GetAmbient());
    lightShader->SetVec3(light + ".diffuse", directionalLight->GetDiffuse());
    lightShader->SetVec3(light + ".specular", directionalLight->GetSpecular());
    lightShader->SetVec3(light + ".color", directionalLight->GetColor());
}

void RenderingManager::UpdateSpotLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    SpotLight* spotLight = spotLights.find(lightNumber)->second;
    std::string light = "spotLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light + ".isActive", spotLight->enabled);
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

void RenderingManager::RemovePointLight(int lightNumber, Shader* lightShader) {
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

    pointLights.find(lightNumber)->second = nullptr;
}

void RenderingManager::RemoveDirectionalLight(int lightNumber, Shader* lightShader) {
    lightShader->Activate();
    std::string light = "directionalLights[" + std::to_string(lightNumber) + "]";
    lightShader->SetBool(light +".isActive", false);
    lightShader->SetVec3(light + ".direction", {0, 0, 0});
    lightShader->SetVec3(light + ".ambient", {0, 0, 0});
    lightShader->SetVec3(light + ".diffuse", {0, 0, 0});
    lightShader->SetVec3(light + ".specular", {0, 0, 0});
    lightShader->SetVec3(light + ".color", {0, 0, 0});

    directionalLights.find(lightNumber)->second = nullptr;
}

void RenderingManager::RemoveSpotLight(int lightNumber, Shader* lightShader) {
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

    spotLights.find(lightNumber)->second = nullptr;
}

void RenderingManager::ClearBuffer() {
    for (int i = 0; i < bufferIterator; ++i) {
        drawBuffer[i] = nullptr;
    }
    bufferIterator = 0;
}