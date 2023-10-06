#include "Managers/NeuralNetworkManager.h"
#include "Managers/RenderingManager.h"
#include "Rendering/ObjectRenderer.h"
#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Components/Rendering/Lights/PointLight.h"
#include "Components/Rendering/Lights/SpotLight.h"
#include "Components/Transform.h"
#include "Core/Object.h"

NeuralNetworkManager::NeuralNetworkManager() = default;

NeuralNetworkManager::~NeuralNetworkManager() = default;

NeuralNetworkManager* NeuralNetworkManager::GetInstance() {
    if (neuralNetworkManager == nullptr) {
        neuralNetworkManager = new NeuralNetworkManager();
    }
    return neuralNetworkManager;
}

void NeuralNetworkManager::Startup() {
    neurons.reserve(20);
    weights.reserve(20);
    lambdas.reserve(20);

    RenderingManager* renderingManager = RenderingManager::GetInstance();
    ObjectRenderer* renderer = renderingManager->objectRenderer;

    //TODO: Add loading saved values here, if exist skip creating new one

    for (int i = 0; i < 4; ++i) {
        DirectionalLight* light = renderer->directionalLights[i];
        if (light != nullptr) {
            DivideDirectionalLightToNeurons(light);
        }
    }

    for (int i = 0; i < 4; ++i) {
        PointLight* light = renderer->pointLights[i];
        if (light != nullptr) {
            DividePointLightToNeurons(light);
        }
    }

    for (int i = 0; i < 4; ++i) {
        SpotLight* light = renderer->spotLights[i];
        if (light != nullptr) {
            DivideSpotLightToNeurons(light);
        }
    }
}

void NeuralNetworkManager::Update() {
    // TODO: rng weights
    // TODO: calculate new light values
    // TODO: set light values
    // TODO: check if new weights gives better values
    // TODO: save or not values
}

void NeuralNetworkManager::Shutdown() {
    // TODO: add saving model values here
}

void NeuralNetworkManager::SetLightValues() {

}

void NeuralNetworkManager::DivideDirectionalLightToNeurons(DirectionalLight *light) {
    glm::vec3 position = light->parent->transform->GetGlobalPosition();
    neurons.push_back(position.x);
    neurons.push_back(position.y);
    neurons.push_back(position.z);

    glm::vec3 ambient = light->GetAmbient();
    neurons.push_back(ambient.x);
    neurons.push_back(ambient.y);
    neurons.push_back(ambient.z);

    glm::vec3 diffuse = light->GetDiffuse();
    neurons.push_back(diffuse.x);
    neurons.push_back(diffuse.y);
    neurons.push_back(diffuse.z);

    glm::vec3 specular = light->GetSpecular();
    neurons.push_back(specular.x);
    neurons.push_back(specular.y);
    neurons.push_back(specular.z);

    glm::vec3 color = light->GetColor();
    neurons.push_back(color.x);
    neurons.push_back(color.y);
    neurons.push_back(color.z);
}

void NeuralNetworkManager::DividePointLightToNeurons(PointLight *light) {
    glm::vec3 position = light->parent->transform->GetGlobalPosition();
    neurons.push_back(position.x);
    neurons.push_back(position.y);
    neurons.push_back(position.z);

    glm::vec3 ambient = light->GetAmbient();
    neurons.push_back(ambient.x);
    neurons.push_back(ambient.y);
    neurons.push_back(ambient.z);

    glm::vec3 diffuse = light->GetDiffuse();
    neurons.push_back(diffuse.x);
    neurons.push_back(diffuse.y);
    neurons.push_back(diffuse.z);

    glm::vec3 specular = light->GetSpecular();
    neurons.push_back(specular.x);
    neurons.push_back(specular.y);
    neurons.push_back(specular.z);

    glm::vec3 color = light->GetColor();
    neurons.push_back(color.x);
    neurons.push_back(color.y);
    neurons.push_back(color.z);

    neurons.push_back(light->GetConstant());
    neurons.push_back(light->GetLinear());
    neurons.push_back(light->GetQuadratic());

}

void NeuralNetworkManager::DivideSpotLightToNeurons(SpotLight *light) {
    glm::vec3 position = light->parent->transform->GetGlobalPosition();
    neurons.push_back(position.x);
    neurons.push_back(position.y);
    neurons.push_back(position.z);

    glm::vec3 ambient = light->GetAmbient();
    neurons.push_back(ambient.x);
    neurons.push_back(ambient.y);
    neurons.push_back(ambient.z);

    glm::vec3 diffuse = light->GetDiffuse();
    neurons.push_back(diffuse.x);
    neurons.push_back(diffuse.y);
    neurons.push_back(diffuse.z);

    glm::vec3 specular = light->GetSpecular();
    neurons.push_back(specular.x);
    neurons.push_back(specular.y);
    neurons.push_back(specular.z);

    glm::vec3 color = light->GetColor();
    neurons.push_back(color.x);
    neurons.push_back(color.y);
    neurons.push_back(color.z);

    neurons.push_back(light->GetConstant());
    neurons.push_back(light->GetLinear());
    neurons.push_back(light->GetQuadratic());

    neurons.push_back(light->GetCutOff());
    neurons.push_back(light->GetOuterCutOff());
}
