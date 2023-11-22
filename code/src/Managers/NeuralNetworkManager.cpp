#include "Managers/NeuralNetworkManager.h"
#include "Managers/RenderingManager.h"
#include "Managers/EditorManager.h"
#include "Rendering/ObjectRenderer.h"
#include "Resources/Texture.h"
#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Components/Rendering/Lights/PointLight.h"
#include "Components/Rendering/Lights/SpotLight.h"
#include "Components/Rendering/UI/Image.h"
#include "Components/Rendering/Camera.h"
#include "Components/Transform.h"
#include "Core/Object.h"
#include "Macros.h"
#include "CUM.h"

#include "effolkronium/random.hpp"
#include "stb_image_write.h"

#include <glad/glad.h>

#define M_PI 3.14159265358979323846
#define RNG(min, max) effolkronium::random_static::get(min, max)

NeuralNetworkManager::NeuralNetworkManager() = default;

NeuralNetworkManager::~NeuralNetworkManager() = default;

NeuralNetworkManager* NeuralNetworkManager::GetInstance() {
    if (neuralNetworkManager == nullptr) {
        neuralNetworkManager = new NeuralNetworkManager();
    }
    return neuralNetworkManager;
}

void NeuralNetworkManager::Startup() {
    /// Set outputSize for now to 2 which are Euler's angles between camera and light on the Dome
    outputSize = 2;

    Load();
}

void NeuralNetworkManager::Run() {
    if (finalize) {
        FinalizeNetwork();
        finalize = false;
    }
}

void NeuralNetworkManager::Shutdown() {
    if (state == NetworkState::Processing || state == NetworkState::Training) {
        Application::GetInstance()->isStarted = false;
        FinalizeNetwork();
    }

    if (thread != nullptr) {
        if (thread->joinable()) thread->join();
        delete thread;
    }

    for (int i = 0; i < weights.size(); ++i) {
        delete weights[i];
    }
    weights.clear();

    for (int i = 0; i < biases.size(); ++i) {
        delete biases[i];
    }
    biases.clear();

    delete loadedData;

    delete neuralNetworkManager;
}

void NeuralNetworkManager::InitializeNetwork(NetworkTask task) {
    layers.reserve(16);
    poolingLayers.reserve(4);

    currentTask = task;

    if (RenderingManager::GetInstance()->objectRenderer->pointLights[0] == nullptr) return;

    if (task == NetworkTask::TrainNetwork) {
        Train(50, 50, 0.0001);
    }
    else if (task == NetworkTask::ProcessImage) {
        ProcessImage();
    }
}

void NeuralNetworkManager::FinalizeNetwork() {
    if (currentTask == NetworkTask::TrainNetwork) {
        Save();
    }

    for (int i = 0; i < layers.size(); ++i) {
        delete layers[i];
    }
    layers.clear();

    for (int i = 0; i < poolingLayers.size(); ++i) {
        delete poolingLayers[i];
    }
    poolingLayers.clear();

    currentTask = None;
    state = Idle;
}

void NeuralNetworkManager::ProcessImage() {
    state = Processing;
    RenderingManager* renderingManager = RenderingManager::GetInstance();

    if (renderingManager->objectRenderer->pointLights[0] == nullptr) FinalizeNetwork();

    Forward();

    glm::vec3 cameraPosition = Camera::GetRenderingCamera()->transform->GetGlobalPosition();
    float* cameraSphericalCoords = CUM::CartesianToSphericalCoordinates(cameraPosition);
    glm::vec3 lightPosition = CUM::SphericalToCartesianCoordinates(layers[15]->maps[0] + cameraSphericalCoords[0],
        layers[15]->maps[1] + cameraSphericalCoords[1], glm::length(cameraPosition));

    delete[] cameraSphericalCoords;

    renderingManager->objectRenderer->pointLights[0]->parent->transform->SetLocalPosition(lightPosition);
}

void NeuralNetworkManager::Train(int epoch, int trainingSize, float learningStep) {
    if (thread != nullptr) {
        if (thread->joinable()) thread->join();
        delete thread;
    }

    thread = new std::thread(&NeuralNetworkManager::ThreadTrain, epoch, trainingSize, learningStep);
}

void NeuralNetworkManager::ThreadTrain(int epoch, int trainingSize, float learningStep) {
    NeuralNetworkManager* manager = NeuralNetworkManager::GetInstance();
    manager->state = NetworkState::Training;

    RenderingManager* renderingManager = RenderingManager::GetInstance();

    // Save texture values
    Texture* texture = EditorManager::GetInstance()->loadedImage->GetComponentByClass<Image>()->GetTexture();
    unsigned int prevImage = texture->GetID();
    glm::ivec2 prevImageResolution = texture->GetResolution();

    // Set camera texture as new loaded image which is used in network forward method
    int width = Application::viewports[0].resolution.x;
    int height = Application::viewports[0].resolution.y;
    texture->SetID(renderingManager->objectRenderer->renderingCameraTexture);
    texture->SetResolution(glm::ivec2(width, height));

    // Generate data set
    int dataSetSize = trainingSize * manager->outputSize;
    float* dataSet = new float[dataSetSize];

    glm::vec3* cameraPositions = new glm::vec3[trainingSize];
    glm::vec3* lightPositions = new glm::vec3[trainingSize];

    for (int i = 0; i < trainingSize; ++i) {
        cameraPositions[i] = glm::normalize(glm::vec3(RNG(-1.0f, 1.0f), RNG(0.0f, 1.0f), RNG(-1.0f, 1.0f))) * 20.0f;
        lightPositions[i] = glm::normalize(glm::vec3(RNG(-1.0f, 1.0f), RNG(0.0f, 1.0f), RNG(-1.0f, 1.0f))) * 20.0f;

        float* cameraSphericalCoords = CUM::CartesianToSphericalCoordinates(cameraPositions[i]);
        float* lightSphericalCoords = CUM::CartesianToSphericalCoordinates(lightPositions[i]);

        float phi = lightSphericalCoords[0] - cameraSphericalCoords[0];
        float theta = lightSphericalCoords[1] - cameraSphericalCoords[1];

        if (phi < 0) phi += (float)(2 * M_PI);
        if (theta < 0) theta += (float)(2 * M_PI);

        dataSet[i * 2] = phi;
        dataSet[i * 2 + 1] = theta;

        delete[] cameraSphericalCoords;
        delete[] lightSphericalCoords;

        printf("TCamera: %f, %f, %f | TLight: %f, %f, %f  | TAngles: %f, %f \n", cameraPositions[i].x, cameraPositions[i].y,
               cameraPositions[i].z, lightPositions[i].x, lightPositions[i].y, lightPositions[i].z,
               dataSet[i * 2], dataSet[i * 2 + 1]);
    }

    float prevEpochAvgLoss = FLT_MAX;

    for (int i = 0; i < epoch; ++i) {
        float epochLoss = 0;

        for (int j = 0; j < trainingSize * manager->outputSize; j += manager->outputSize) {
            int idx = j / manager->outputSize;

            // Calculate camera looking direction and rotate it to look at point(0,0,0)
            glm::vec3 direction = glm::normalize(glm::vec3(0, 0, 0) - cameraPositions[idx]);

            float angleX = (float)(asin(direction.y) * 180.0f / M_PI);
            float angleY = (float)(-atan2(direction.x, -direction.z) * 180.0f / M_PI);
            Camera::GetRenderingCamera()->transform->SetLocalRotation(glm::vec3(angleX, angleY, 0));
            Camera::GetRenderingCamera()->transform->SetLocalPosition(cameraPositions[idx]);

            renderingManager->objectRenderer->pointLights[0]->parent->transform->SetLocalPosition(lightPositions[idx]);

            renderingManager->DrawOtherViewports();

            manager->Forward();

            float predictedValues[2] = {dataSet[j], dataSet[j + 1]};
            spdlog::info("Output: " + std::to_string(manager->layers[15]->maps[0]) + ", " +
                std::to_string(manager->layers[15]->maps[1]) + ", Target: " +
                std::to_string(predictedValues[0]) + ", " + std::to_string(predictedValues[1]));


            float loss = MSELossFunction(manager->layers[15]->maps, predictedValues, manager->outputSize);
            epochLoss += loss;

            manager->Backward(predictedValues, learningStep);

            for (int k = 0; k < manager->layers.size(); ++k) {
                delete manager->layers[k];
            }
            manager->layers.clear();

            for (int k = 0; k < manager->poolingLayers.size(); ++k) {
                delete manager->poolingLayers[k];
            }
            manager->poolingLayers.clear();

            if (!Application::GetInstance()->isStarted) {
                break;
            }
        }

        float averageEpochLoss = epochLoss / (float)trainingSize;

        spdlog::info("Epoch: " + std::to_string(i) + ", Loss: " + std::to_string(averageEpochLoss));

        if (!Application::GetInstance()->isStarted || prevEpochAvgLoss < averageEpochLoss) {
            break;
        }

        if (!isnan(averageEpochLoss)) {
            ThreadSave(false);
        }

        prevEpochAvgLoss = averageEpochLoss;
    }

    delete[] cameraPositions;
    delete[] lightPositions;
    delete[] dataSet;

    texture->SetID(prevImage);
    texture->SetResolution(prevImageResolution);

    if (Application::GetInstance()->isStarted) {
        Application::GetInstance()->isStarted = false;
        manager->finalize = true;
    }
}

void NeuralNetworkManager::Forward() {
    glm::ivec2 imageDim = glm::ivec2(224, 224);
    loadedData = GetLoadedImageWithSize(imageDim.x, imageDim.y);

#pragma region Classify
    // Group 1
    // Conv 1
    layers.emplace_back(ConvolutionLayer(loadedData, weights[0], {1, 1}, {1, 1}, biases[0]->maps));
    //ReLU
    ReLULayer(layers[0]);

    //Conv 2
    layers.emplace_back(ConvolutionLayer(layers[0], weights[1], {1, 1}, {1, 1}, biases[1]->maps));
    // ReLU
    ReLULayer(layers[1]);
    // Max Pooling [0]
    poolingLayers.emplace_back(PoolingLayer(layers[1], {2, 2}, {2, 2}));


    // Group 2
    // Conv 3
    layers.emplace_back(ConvolutionLayer(poolingLayers[0], weights[2], {1, 1}, {1, 1}, biases[2]->maps));
    // ReLU
    ReLULayer(layers[2]);

    // Conv 4
    layers.emplace_back(ConvolutionLayer(layers[2], weights[3], {1, 1}, {1, 1}, biases[3]->maps));
    // ReLU
    ReLULayer(layers[3]);
    // Max Pooling [1]
    poolingLayers.emplace_back(PoolingLayer(layers[3], {2, 2}, {2, 2}));


    // Group 3
    // Conv 5
    layers.emplace_back(ConvolutionLayer(poolingLayers[1], weights[4], {1, 1}, {1, 1}, biases[4]->maps));
    // ReLU
    ReLULayer(layers[4]);

    // Conv 6
    layers.emplace_back(ConvolutionLayer(layers[4], weights[5], {1, 1}, {1, 1}, biases[5]->maps));
    // ReLU
    ReLULayer(layers[5]);

    // Conv 7
    layers.emplace_back(ConvolutionLayer(layers[5], weights[6], {1, 1}, {1, 1}, biases[6]->maps));
    // ReLU
    ReLULayer(layers[6]);
    // Max Pooling [2]
    poolingLayers.emplace_back(PoolingLayer(layers[6], {2, 2}, {2, 2}));

    // Group 4
    // Conv 8
    layers.emplace_back(ConvolutionLayer(poolingLayers[2], weights[7], {1, 1}, {1, 1}, biases[7]->maps));
    // ReLU
    ReLULayer(layers[7]);

    // Conv 9
    layers.emplace_back(ConvolutionLayer(layers[7], weights[8], {1, 1}, {1, 1}, biases[8]->maps));
    // ReLU
    ReLULayer(layers[8]);

    // Conv 10
    layers.emplace_back(ConvolutionLayer(layers[8], weights[9], {1, 1}, {1, 1}, biases[9]->maps));
    // ReLU
    ReLULayer(layers[9]);
    // Max Pooling [3]
    poolingLayers.emplace_back(PoolingLayer(layers[9], {2, 2}, {2, 2}));


    // Group 5
    // Conv 11
    layers.emplace_back(ConvolutionLayer(poolingLayers[3], weights[10], {1, 1}, {1, 1}, biases[10]->maps));
    // ReLU
    ReLULayer(layers[10]);

    // Conv 12
    layers.emplace_back(ConvolutionLayer(layers[10], weights[11], {1, 1}, {1, 1}, biases[11]->maps));
    // ReLU
    ReLULayer(layers[11]);

    // Conv 13
    layers.emplace_back(ConvolutionLayer(layers[11], weights[12], {1, 1}, {1, 1}, biases[12]->maps));
    // ReLU
    ReLULayer(layers[12]);
    // Max Pooling [4]
    poolingLayers.emplace_back(PoolingLayer(layers[12], {2, 2}, {2, 2}));
#pragma endregion

    // Neurons of Hidden Layer 1
    layers.emplace_back(FullyConnectedLayer(poolingLayers[4], weights[13]->filters[0].maps, 25088, 4096, biases[13]->maps));

    // Neurons of Hidden Layer 2
    layers.emplace_back(FullyConnectedLayer(layers[13], weights[14]->filters[0].maps, 4096, 4096, biases[14]->maps));

    // Output neurons
    layers.emplace_back(FullyConnectedLayer(layers[14], weights[15]->filters[0].maps, 4096, outputSize, biases[15]->maps));
}

void NeuralNetworkManager::Backward(float* target, float learningRate) {
    float* outputGradients = new float[2];

    for (int i = 0; i < layers[15]->width; ++i) {
        outputGradients[i] = 2 * (layers[15]->maps[i] - target[i]);
    }

    printf("FCB ");
    FullyConnectedLayerBackward(layers[15], weights[15], biases[15], layers[14], outputGradients, learningRate);
    printf("FCB ");
    FullyConnectedLayerBackward(layers[14], weights[14], biases[14], layers[13], outputGradients, learningRate);
    printf("FCB ");
    FullyConnectedLayerBackward(layers[13], weights[13], biases[13], poolingLayers[4], outputGradients, learningRate);


    printf("MPB ");
    MaxPoolingBackward(poolingLayers[4], layers[12], outputGradients, ivec2(2, 2));
    printf("CLB ");
    ConvolutionLayerBackward(layers[12], weights[12], biases[12], layers[11], outputGradients, learningRate);
    printf("CLB ");
    ConvolutionLayerBackward(layers[11], weights[11], biases[11], layers[10], outputGradients, learningRate);
    printf("CLB ");
    ConvolutionLayerBackward(layers[10], weights[10], biases[10], poolingLayers[3], outputGradients, learningRate);

    printf("MPB ");
    MaxPoolingBackward(poolingLayers[3], layers[9], outputGradients, ivec2(2, 2));
    printf("CLB ");
    ConvolutionLayerBackward(layers[9], weights[9], biases[9], layers[8], outputGradients, learningRate);
    printf("CLB ");
    ConvolutionLayerBackward(layers[8], weights[8], biases[8], layers[7], outputGradients, learningRate);
    printf("CLB ");
    ConvolutionLayerBackward(layers[7], weights[7], biases[7], poolingLayers[2], outputGradients, learningRate);

    printf("MPB ");
    MaxPoolingBackward(poolingLayers[2], layers[6], outputGradients, ivec2(2, 2));
    printf("CLB ");
    ConvolutionLayerBackward(layers[6], weights[6], biases[6], layers[5], outputGradients, learningRate);
    printf("CLB ");
    ConvolutionLayerBackward(layers[5], weights[5], biases[5], layers[4], outputGradients, learningRate);
    printf("CLB ");
    ConvolutionLayerBackward(layers[4], weights[4], biases[4], poolingLayers[1], outputGradients, learningRate);

    printf("MPB ");
    MaxPoolingBackward(poolingLayers[1], layers[3], outputGradients, ivec2(2, 2));
    printf("CLB ");
    ConvolutionLayerBackward(layers[3], weights[3], biases[3], layers[2], outputGradients, learningRate);
    printf("CLB ");
    ConvolutionLayerBackward(layers[2], weights[2], biases[2], poolingLayers[0], outputGradients, learningRate);

    printf("MPB ");
    MaxPoolingBackward(poolingLayers[0], layers[1], outputGradients, ivec2(2, 2));
    printf("CLB ");
    ConvolutionLayerBackward(layers[1], weights[1], biases[1], layers[0], outputGradients, learningRate);
    printf("CLB ");
    ConvolutionLayerBackward(layers[0], weights[0], biases[0], loadedData, outputGradients, learningRate);

    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[i]->count; ++j) {
            for (int k = 0; k < weights[i]->filters[j].width * weights[i]->filters[j].height * weights[i]->filters[j].depth; ++k) {
                if (weights[i]->filters[j].maps[k] > 20.0f || weights[i]->filters[j].maps[k] < -20.0f || isnan(weights[i]->filters[j].maps[k]))
                    printf("%i, %i, %i, %f\n", i, j, k, weights[i]->filters[j].maps[k]);
            }
        }
    }

    printf("\n");

    delete[] outputGradients;
}

Layer* NeuralNetworkManager::GetLoadedImageWithSize(int outWidth, int outHeight) {
    const Texture* texture = EditorManager::GetInstance()->loadedImage->GetComponentByClass<Image>()->GetTexture();

    int inWidth = texture->GetResolution().x;
    int inHeight = texture->GetResolution().y;

    unsigned char* image = new unsigned char[inWidth * inHeight * 3];

    // Get texture image
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture->GetID());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

    unsigned char* resizedImage = CUM::ResizeImage(image, inWidth, inHeight, outWidth, outHeight);
    delete[] image;

    image = CUM::RotateImage(resizedImage, outWidth, outHeight, 3);
    delete[] resizedImage;

    Layer* output = new Layer();

    output->width = outWidth;
    output->height = outHeight;
    output->depth = 3;

    output->maps = new float[outWidth * outHeight * 3];

    int counter = 0;
    for (int i = 0; i < outWidth * outHeight * 3; i+=3) {
        output->maps[counter] = (float)image[i] / 255;
        output->maps[counter + outWidth] = (float)image[i + 1] / 255;
        output->maps[counter + 2 * outWidth] = (float)image[i + 2] / 255;
        ++counter;
    }

    delete[] image;

    return output;
}

#pragma region Save And Load
void NeuralNetworkManager::Load() {
    if (thread != nullptr) {
        if (thread->joinable()) thread->join();
        delete thread;
    }

    thread = new std::thread(&NeuralNetworkManager::ThreadLoad);
}

void NeuralNetworkManager::ThreadLoad() {
    NeuralNetworkManager* manager = NeuralNetworkManager::GetInstance();
    manager->state = LoadingSaving;

    for (int i = 0; i < manager->weights.size(); ++i) {
        delete manager->weights[i];
    }
    manager->weights.clear();

    for (int i = 0; i < manager->biases.size(); ++i) {
        delete manager->biases[i];
    }
    manager->biases.clear();

    manager->weights.reserve(16);
    manager->biases.reserve(16);

    manager->biases.push_back(new Layer(64, 1, 1));
    manager->biases.push_back(new Layer(64, 1, 1));
    manager->biases.push_back(new Layer(128, 1, 1));
    manager->biases.push_back(new Layer(128, 1, 1));
    manager->biases.push_back(new Layer(256, 1, 1));
    manager->biases.push_back(new Layer(256, 1, 1));
    manager->biases.push_back(new Layer(256, 1, 1));
    manager->biases.push_back(new Layer(512, 1, 1));
    manager->biases.push_back(new Layer(512, 1, 1));
    manager->biases.push_back(new Layer(512, 1, 1));
    manager->biases.push_back(new Layer(512, 1, 1));
    manager->biases.push_back(new Layer(512, 1, 1));
    manager->biases.push_back(new Layer(512, 1, 1));
    manager->biases.emplace_back(new Layer(4096, 1, 1));
    manager->biases.emplace_back(new Layer(4096, 1, 1));
    manager->biases.emplace_back(new Layer(2, 1, 1));

    FILE* stream;
    fopen_s(&stream, "resources/Resources/NeuralNetworkResources/Model.json", "rb");
    if (stream != nullptr) {
#pragma region Weights Inits
        manager->weights.emplace_back(new Group(64, 0, ivec3(3, 3, 3)));
        manager->weights.emplace_back(new Group(64, 0, ivec3(3, 3, 64)));
        manager->weights.emplace_back(new Group(128, 0, ivec3(3, 3, 64)));
        manager->weights.emplace_back(new Group(128, 0, ivec3(3, 3, 128)));
        manager->weights.emplace_back(new Group(256, 0, ivec3(3, 3, 128)));
        manager->weights.emplace_back(new Group(256, 0, ivec3(3, 3, 256)));
        manager->weights.emplace_back(new Group(256, 0, ivec3(3, 3, 256)));
        manager->weights.emplace_back(new Group(512, 0, ivec3(3, 3, 256)));
        manager->weights.emplace_back(new Group(512, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(512, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(512, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(512, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(512, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(1, 0, ivec3(102760448, 1, 1)));
        manager->weights.emplace_back(new Group(1, 0, ivec3(16777216, 1, 1)));
        manager->weights.emplace_back(new Group(1, 0, ivec3(4096 * manager->outputSize, 1, 1)));
#pragma endregion

        for (int w = 0; w < manager->weights.size(); ++w) {
            for (int i = 0; i < manager->weights[w]->count; ++i) {
                long mapSize = manager->weights[w]->filters[i].width * manager->weights[w]->filters[i].height *
                               manager->weights[w]->filters[i].depth;
                fread(manager->weights[w]->filters[i].maps, sizeof(float), mapSize, stream);
                fseek(stream, (long)(mapSize * sizeof(float)), SEEK_CUR);
            }
        }

        for (int b = 0; b < manager->biases.size(); ++b) {
            long mapSize = manager->biases[b]->width * manager->biases[b]->height *
                           manager->biases[b]->depth;
            fread(manager->biases[b]->maps, sizeof(float), mapSize, stream);
            fseek(stream, (long)(mapSize * sizeof(float)), SEEK_CUR);
        }

        fclose(stream);
    }
    else {
        manager->weights.emplace_back(new Group(64, RNG(0, 2137), ivec3(3, 3, 3), true));
        manager->weights.emplace_back(new Group(64, RNG(0, 2137), ivec3(3, 3, 64), true));
        manager->weights.emplace_back(new Group(128, RNG(0, 2137), ivec3(3, 3, 64), true));
        manager->weights.emplace_back(new Group(128, RNG(0, 2137), ivec3(3, 3, 128), true));
        manager->weights.emplace_back(new Group(256, RNG(0, 2137), ivec3(3, 3, 128), true));
        manager->weights.emplace_back(new Group(256, RNG(0, 2137), ivec3(3, 3, 256), true));
        manager->weights.emplace_back(new Group(256, RNG(0, 2137), ivec3(3, 3, 256), true));
        manager->weights.emplace_back(new Group(512, RNG(0, 2137), ivec3(3, 3, 256), true));
        manager->weights.emplace_back(new Group(512, RNG(0, 2137), ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, RNG(0, 2137), ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, RNG(0, 2137), ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, RNG(0, 2137), ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, RNG(0, 2137), ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(1, RNG(0, 2137), ivec3(102760448, 1, 1), true));
        manager->weights.emplace_back(new Group(1, RNG(0, 2137), ivec3(16777216, 1, 1), true));
        manager->weights.emplace_back(new Group(1, RNG(0, 2137), ivec3(4096 * manager->outputSize, 1, 1), true));
    }
    manager->state = Idle;
}

void NeuralNetworkManager::Save() {
    if (thread != nullptr) {
        if (thread->joinable()) thread->join();
        delete thread;
    }

    thread = new std::thread(&NeuralNetworkManager::ThreadSave, true);
}

void NeuralNetworkManager::ThreadSave(bool changeState) {
    NeuralNetworkManager* manager = NeuralNetworkManager::GetInstance();

    if(changeState) manager->state = LoadingSaving;

    FILE* stream;
    fopen_s(&stream, "resources/Resources/NeuralNetworkResources/Model.json", "wb");

    if (stream != nullptr) {
        for (int w = 0; w < manager->weights.size(); ++w) {
            for (int i = 0; i < manager->weights[w]->count; ++i) {
                long mapSize = manager->weights[w]->filters[i].width * manager->weights[w]->filters[i].height *
                               manager->weights[w]->filters[i].depth;
                fwrite(manager->weights[w]->filters[i].maps, sizeof(float), mapSize, stream);
                fseek(stream, (long)(mapSize * sizeof(float)), SEEK_CUR);
            }
        }

        for (int b = 0; b < manager->biases.size(); ++b) {
            long mapSize = manager->biases[b]->width * manager->biases[b]->height *
                           manager->biases[b]->depth;
            fwrite(manager->biases[b]->maps, sizeof(float), mapSize, stream);
            fseek(stream, (long)(mapSize * sizeof(float)), SEEK_CUR);
        }
        fclose(stream);
    }
    if(changeState) manager->state = Idle;
}
#pragma endregion