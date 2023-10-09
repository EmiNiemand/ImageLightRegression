#include "Managers/NeuralNetworkManager.h"
#include "Managers/RenderingManager.h"
#include "Managers/EditorManager.h"
#include "Rendering/ObjectRenderer.h"
#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Components/Rendering/Lights/PointLight.h"
#include "Components/Rendering/Lights/SpotLight.h"
#include "Components/Rendering/UI/Image.h"
#include "Components/Transform.h"
#include "Core/Object.h"
#include "Macros.h"

#include "effolkronium/random.hpp"

#include <glad/glad.h>

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
    outputs.reserve(20);
    weights.reserve(400);
    weightsCopy.reserve(400);
    biases.reserve(20);
}

void NeuralNetworkManager::Shutdown() {
    // TODO: add saving model values here
    delete loadedImage;

    delete neuralNetworkManager;
}

void NeuralNetworkManager::InitializeNetwork() {
    weightsCopy.clear();
    weights.clear();
    neurons.clear();
    outputs.clear();
    biases.clear();

    int width = Application::viewports[0].resolution.x;
    int height = Application::viewports[0].resolution.y;

    loadedImage = new char[width * height * 3];

    // Get texture image
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, EditorManager::GetInstance()->loadedImage->GetComponentByClass<Image>()->GetTextureID());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, loadedImage);

    //TODO: Add loading saved values here, if exist skip creating new one

    DivideLightValues();

    for (int i = 0; i < neurons.size() * neurons.size(); ++i) {
        weightsCopy.push_back(effolkronium::random_static::get(0.0f, 0.1f));
    }


    weights = weightsCopy;

    // TODO: check if messing with biases will give better results
    for (int i = 0; i < neurons.size(); ++i) {
//        biases.push_back(effolkronium::random_static::get(0.0f, 1.0f));
        biases.push_back(0.0f);
    }
}

void NeuralNetworkManager::Finalize() {
    weights = weightsCopy;
    CalculateOutputs();
    SetLightValues();
}

void NeuralNetworkManager::PreRenderUpdate() {
    if (!Application::GetInstance()->isStarted) return;
    CalculateWeights();
    CalculateOutputs();
    SetLightValues();
}

void NeuralNetworkManager::PostRenderUpdate() {
    if (!Application::GetInstance()->isStarted) return;
    if (CheckOutputValues()) {
        weightsCopy = weights;
    }
    else {
        ++iteration;
    }
}

float NeuralNetworkManager::CalculateAverage(char* data, int size) {
    float* averages = new float[size];

    for (int i = 0; i < size * 3; i+=3) {
        averages[i / 3] = (fabs((float)loadedImage[i] - (float)data[i]) / (((float)loadedImage[i] + (float)data[i]) / 2.0f) +
                           fabs((float)loadedImage[i + 1] - (float)data[i + 1]) / (((float)loadedImage[i + 1] + (float)data[i + 1]) / 2.0f) +
                           fabs((float)loadedImage[i + 2] - (float)data[i + 2]) / (((float)loadedImage[i + 2] + (float)data[i + 2]) / 2.0f)) / 3.0f;
    }

    float average = 0;

    for (int i = 0; i < size; ++i) {
        average += averages[i];
    }

    delete[] averages;

    return 1 - average /(float)size;
}

void NeuralNetworkManager::CalculateWeights() {
    weights.clear();
    for (int i = 0; i < weightsCopy.size(); ++i) {
        weights.push_back(weightsCopy[i] + effolkronium::random_static::get(-0.01f, 0.01f));
    }
}

void NeuralNetworkManager::CalculateOutputs() {
    outputs.clear();
    float neuron = 0;
    for (int i = 0; i < neurons.size(); ++i) {
        for (int j = 0; j < neurons.size(); ++j) {
            neuron += neurons[j] * weights[i * neurons.size() + j];
        }
        neuron += biases[i];
        outputs.push_back(neuron);
        neuron = 0;
    }
}

bool NeuralNetworkManager::CheckOutputValues() {
    int width = Application::viewports[0].resolution.x;
    int height = Application::viewports[0].resolution.y;

    int size = width * height;

    char* data = new char[size * 3];

    // Get texture image
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderingManager::GetInstance()->objectRenderer->screenTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

    float average = CalculateAverage(data, size);

    delete[] data;

    if (average > previousAccuracy) {
        previousAccuracy = average;

        return true;
    }

    return false;
}

void NeuralNetworkManager::SetLightValues() {
    RenderingManager* renderingManager = RenderingManager::GetInstance();
    ObjectRenderer* renderer = renderingManager->objectRenderer;

    for (int i = 0; i < 4; ++i) {
        DirectionalLight* light = renderer->directionalLights[i];
        if (light != nullptr) {
            CombineNeuronsToDirectionalLight(light, i * 15);
        }
    }

    for (int i = 0; i < 4; ++i) {
        PointLight* light = renderer->pointLights[i];
        if (light != nullptr) {
            CombineNeuronsToPointLight(light, directionalLightsNumber * 15 + i * 18);
        }
    }

    for (int i = 0; i < 4; ++i) {
        SpotLight* light = renderer->spotLights[i];
        if (light != nullptr) {
            CombineNeuronsToSpotLight(light, directionalLightsNumber * 15 + pointLightsNumber * 18 + i * 20);
        }
    }
}

void NeuralNetworkManager::DivideLightValues() {
    RenderingManager* renderingManager = RenderingManager::GetInstance();
    ObjectRenderer* renderer = renderingManager->objectRenderer;

    for (int i = 0; i < 4; ++i) {
        DirectionalLight* light = renderer->directionalLights[i];
        if (light != nullptr) {
            DivideDirectionalLightToNeurons(light);
            ++directionalLightsNumber;
        }
    }

    for (int i = 0; i < 4; ++i) {
        PointLight* light = renderer->pointLights[i];
        if (light != nullptr) {
            DividePointLightToNeurons(light);
            ++pointLightsNumber;
        }
    }

    for (int i = 0; i < 4; ++i) {
        SpotLight* light = renderer->spotLights[i];
        if (light != nullptr) {
            DivideSpotLightToNeurons(light);
            ++spotLightsNumber;
        }
    }
}

#pragma region Light divding and combining to/from neurons
void NeuralNetworkManager::CombineNeuronsToDirectionalLight(DirectionalLight *light, int index) {
    light->parent->transform->SetLocalPosition(glm::vec3(outputs[index], outputs[index + 1], outputs[index + 2]));
    light->SetAmbient(glm::vec3(outputs[index + 3], outputs[index + 4], outputs[index + 5]));
    light->SetDiffuse(glm::vec3(outputs[index + 6], outputs[index + 7], outputs[index + 8]));
    light->SetSpecular(glm::vec3(outputs[index + 9], outputs[index + 10], outputs[index + 11]));
    light->SetColor(glm::vec3(outputs[index + 12], outputs[index + 13], outputs[index + 14]));
}

void NeuralNetworkManager::CombineNeuronsToPointLight(PointLight *light, int index) {
    light->parent->transform->SetLocalPosition(glm::vec3(outputs[index], outputs[index + 1], outputs[index + 2]));
    light->SetAmbient(glm::vec3(outputs[index + 3], outputs[index + 4], outputs[index + 5]));
    light->SetDiffuse(glm::vec3(outputs[index + 6], outputs[index + 7], outputs[index + 8]));
    light->SetSpecular(glm::vec3(outputs[index + 9], outputs[index + 10], outputs[index + 11]));
    light->SetColor(glm::vec3(outputs[index + 12], outputs[index + 13], outputs[index + 14]));
    light->SetConstant(outputs[index + 15]);
    light->SetLinear(outputs[index + 16]);
    light->SetQuadratic(outputs[index + 17]);
}

void NeuralNetworkManager::CombineNeuronsToSpotLight(SpotLight *light, int index) {
    light->parent->transform->SetLocalPosition(glm::vec3(outputs[index], outputs[index + 1], outputs[index + 2]));
    light->SetAmbient(glm::vec3(outputs[index + 3], outputs[index + 4], outputs[index + 5]));
    light->SetDiffuse(glm::vec3(outputs[index + 6], outputs[index + 7], outputs[index + 8]));
    light->SetSpecular(glm::vec3(outputs[index + 9], outputs[index + 10], outputs[index + 11]));
    light->SetColor(glm::vec3(outputs[index + 12], outputs[index + 13], outputs[index + 14]));
    light->SetConstant(outputs[index + 15]);
    light->SetLinear(outputs[index + 16]);
    light->SetQuadratic(outputs[index + 17]);
    light->SetCutOff(outputs[index + 18]);
    light->SetOuterCutOff(outputs[index + 19]);
}

void NeuralNetworkManager::DivideDirectionalLightToNeurons(DirectionalLight *light) {
    glm::vec3 position = light->parent->transform->GetLocalPosition();
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
    glm::vec3 position = light->parent->transform->GetLocalPosition();
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
    glm::vec3 position = light->parent->transform->GetLocalPosition();
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
#pragma endregion