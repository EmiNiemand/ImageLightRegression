#include "Managers/NeuralNetworkManager.h"
#include "Managers/RenderingManager.h"
#include "Managers/EditorManager.h"
#include "Rendering/ObjectRenderer.h"
#include "Resources/Texture.h"
#include "NeuralNetwork/AdamOptimizer.h"
#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Components/Rendering/Lights/PointLight.h"
#include "Components/Rendering/Lights/SpotLight.h"
#include "Components/Rendering/UI/Image.h"
#include "Components/Rendering/Camera.h"
#include "Components/Transform.h"
#include "Core/Object.h"
#include "Macros.h"
#include "CUM.h"

#include "stb_image_write.h"
#include <spdlog/sinks/basic_file_sink.h>

NeuralNetworkManager::NeuralNetworkManager() = default;

NeuralNetworkManager::~NeuralNetworkManager() = default;

NeuralNetworkManager* NeuralNetworkManager::GetInstance() {
    if (neuralNetworkManager == nullptr) {
        neuralNetworkManager = new NeuralNetworkManager();
    }
    return neuralNetworkManager;
}

void NeuralNetworkManager::Startup() {
    AdamOptimizer::GetInstance()->Startup();

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

    DELETE_VECTOR_VALUES(weights)
    DELETE_VECTOR_VALUES(biases)

    AdamOptimizer::GetInstance()->Shutdown();

    delete loadedData;
    delete neuralNetworkManager;
}

void NeuralNetworkManager::InitializeNetwork(NetworkTask task) {
    layers.reserve(16);
    poolingLayers.reserve(4);

    currentTask = task;

    if (task == NetworkTask::TrainNetwork) {
        Train(10000, 1000, 10, 100, 0.0001, 0.0000000001);
    }
    else if (task == NetworkTask::ProcessImage) {
        ProcessImage();
    }
}

void NeuralNetworkManager::FinalizeNetwork() {
    if (currentTask == NetworkTask::TrainNetwork) {
        Save();
    }

    DELETE_VECTOR_VALUES(layers)
    DELETE_VECTOR_VALUES(poolingLayers)

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
    delete loadedData;
    loadedData = nullptr;

    renderingManager->objectRenderer->pointLights[0]->parent->transform->SetLocalPosition(lightPosition);
}

void NeuralNetworkManager::Train(int epoch, int trainingSize, int batchSize, int patience, float learningStep,
                                                                                            float minLearningRate) {
    if (thread != nullptr) {
        if (thread->joinable()) thread->join();
        delete thread;
    }

    thread = new std::thread(&NeuralNetworkManager::ThreadTrain, epoch, trainingSize, batchSize, patience, learningStep,
                                                                                                        minLearningRate);
}

void NeuralNetworkManager::ThreadTrain(int epoch, int trainingSize, int batchSize, int patience, float learningRate,
                                                                                            float minLearningRate) {
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

    // Save Camera initial values
    Transform* renderingCameraTransform = Camera::GetRenderingCamera()->transform;
    glm::vec3 cPosition = renderingCameraTransform->GetLocalPosition();
    glm::vec3 cRotation = renderingCameraTransform->GetLocalRotation();

    // Generate or read data set
    int dataSetSize = trainingSize * manager->outputSize;
    float* dataSet = new float[dataSetSize];
    glm::vec3* cameraPositions = new glm::vec3[trainingSize];
    glm::vec3* lightPositions = new glm::vec3[trainingSize];

    FillDataSet(dataSet, cameraPositions, lightPositions, dataSetSize, trainingSize);

    // Init variables
    std::vector<std::vector<Gradient*>> gradients;
    gradients.reserve(batchSize);

    float epochLoss;
    float bestEpochLoss = FLT_MAX;
    int patienceCounter = 0;

    // Training loop
    for (int i = 0; i < epoch; ++i) {
        epochLoss = 0;

        gradients.resize(batchSize);

        for (int j = 0; j < batchSize; ++j) {
            int idx = RNG(0, trainingSize - 1);

            // Calculate camera looking direction and rotate it to look at point(0,0,0)
            glm::vec3 direction = glm::normalize(glm::vec3(0, 0, 0) - cameraPositions[idx]);

            float angleX = (float)(asin(direction.y) * 180.0f / M_PI);
            float angleY = (float)(-atan2(direction.x, -direction.z) * 180.0f / M_PI);
            Camera::GetRenderingCamera()->transform->SetLocalRotation(glm::vec3(angleX, angleY, 0));
            Camera::GetRenderingCamera()->transform->SetLocalPosition(cameraPositions[idx]);

            renderingManager->objectRenderer->pointLights[0]->parent->transform->SetLocalPosition(lightPositions[idx]);
            renderingManager->DrawOtherViewports();

            // Wait until frame is rendered
            while (manager->waitForUpdate);

            manager->Forward();

            float predictedValues[2] = {dataSet[idx * 2], dataSet[idx * 2 + 1]};
            ILR_INFO_MSG("Output: " + STRING(manager->layers[15]->maps[0]) + ", " + STRING(manager->layers[15]->maps[1]) +
                         ", Target: " + STRING(predictedValues[0]) + ", " + STRING(predictedValues[1]));

            float loss = MSELossFunction(manager->layers[15]->maps, predictedValues, manager->outputSize);
            epochLoss += loss;

            manager->Backward(predictedValues, gradients[j]);

            DELETE_VECTOR_VALUES(manager->layers)
            DELETE_VECTOR_VALUES(manager->poolingLayers)

            delete manager->loadedData;
            manager->loadedData = nullptr;

            if (!Application::GetInstance()->isStarted) {
                break;
            }
        }

        if (!Application::GetInstance()->isStarted) {
            for (int g = 0; g < gradients.size(); ++g) {
                DELETE_VECTOR_VALUES(gradients[g])
            }
            gradients.clear();
            break;
        }


        float averageEpochLoss = epochLoss / (float)batchSize;

        ILR_WARN_MSG("**********************************");
        ILR_WARN_MSG("Epoch: " + STRING(i) + ", Loss: " + STRING(averageEpochLoss) + ", Rate: " + STRING(learningRate));
        ILR_WARN_MSG("**********************************");

        if (bestEpochLoss <= averageEpochLoss) {
            ++patienceCounter;
            if (patienceCounter == patience) {
                learningRate *= 0.1f;
                if (learningRate < minLearningRate) learningRate = minLearningRate;
                patienceCounter = 0;
            }
        }
        else if (bestEpochLoss > averageEpochLoss) {
            patienceCounter = 0;
            bestEpochLoss = averageEpochLoss;
        }

        MiniBatch(gradients, manager->weights, manager->biases, learningRate);
        ThreadSave(false);

        for (int g = 0; g < gradients.size(); ++g) {
            DELETE_VECTOR_VALUES(gradients[g])
        }
        gradients.clear();
    }

    delete[] cameraPositions;
    delete[] lightPositions;
    delete[] dataSet;

    // Set old values for texture and camera
    texture->SetID(prevImage);
    texture->SetResolution(prevImageResolution);
    renderingCameraTransform->SetLocalPosition(cPosition);
    renderingCameraTransform->SetLocalRotation(cRotation);

    if (Application::GetInstance()->isStarted) {
        Application::GetInstance()->isStarted = false;
        manager->finalize = true;
    }
}

void NeuralNetworkManager::FillDataSet(float *dataSet, glm::vec3* cameraPositions, glm::vec3* lightPositions, int dataSize,
                                                                                                    int trainingSize) {

    FILE* stream;
    fopen_s(&stream, "resources/Resources/NeuralNetworkResources/Data.json", "rb");
    if (stream != nullptr) {
        fread(dataSet, sizeof(float), dataSize, stream);
        fseek(stream, (long)(dataSize * sizeof(float)), SEEK_CUR);

        fread(cameraPositions, sizeof(glm::vec3), trainingSize, stream);
        fseek(stream, (long)(dataSize * sizeof(glm::vec3)), SEEK_CUR);

        fread(lightPositions, sizeof(glm::vec3), trainingSize, stream);
        fclose(stream);
    }
    else {
        std::shared_ptr<spdlog::logger> logger = spdlog::basic_logger_mt("logger", "resources/Resources/NeuralNetworkResources/DataLog.txt");
        logger->set_level(spdlog::level::info);
        logger->flush_on(spdlog::level::info);

        for (int i = 0; i < trainingSize; ++i) {
            cameraPositions[i] = glm::normalize(glm::vec3(RNG(-1.0f, 1.0f), RNG(0.0f, 1.0f), RNG(-1.0f, 1.0f))) * 20.0f;
            lightPositions[i] = glm::normalize(glm::vec3(RNG(-1.0f, 1.0f), RNG(0.0f, 1.0f), RNG(-1.0f, 1.0f))) * 20.0f;

            float* cameraSphericalCoords = CUM::CartesianToSphericalCoordinates(cameraPositions[i]);
            float* lightSphericalCoords = CUM::CartesianToSphericalCoordinates(lightPositions[i]);

            float phi = lightSphericalCoords[0] - cameraSphericalCoords[0];
            float theta = lightSphericalCoords[1] - cameraSphericalCoords[1];

            if (phi > (float)M_PI) phi = phi - 2.0f * (float)M_PI;
            if (phi < -(float)M_PI) phi = phi + 2.0f * (float)M_PI;

            dataSet[i * 2] = phi;
            dataSet[i * 2 + 1] = theta;

            logger->info("Camera: " + STRING_VEC3(cameraPositions[i]) + " Light: " + STRING_VEC3(lightPositions[i]) +
                         " Angles: " + STRING(phi) + ", " + STRING(theta));

            delete[] cameraSphericalCoords;
            delete[] lightSphericalCoords;
        }
        spdlog::drop("logger");
        logger.reset();

        fopen_s(&stream, "resources/Resources/NeuralNetworkResources/Data.json", "wb");

        fwrite(dataSet, sizeof(float), dataSize, stream);
        fseek(stream, (long)(dataSize * sizeof(float)), SEEK_CUR);

        fwrite(cameraPositions, sizeof(glm::vec3), trainingSize, stream);
        fseek(stream, (long)(dataSize * sizeof(glm::vec3)), SEEK_CUR);

        fwrite(lightPositions, sizeof(glm::vec3), trainingSize, stream);

        fclose(stream);
    }}

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

void NeuralNetworkManager::Backward(const float* target, std::vector<Gradient*>& gradients) {
    std::vector<float> outputGradients;
    outputGradients.reserve(2);

    float max = -FLT_MAX;
    for (int i = 0; i < layers[15]->width; ++i) {
        outputGradients.push_back(2 * (layers[15]->maps[i] - target[i]));
        if (outputGradients[i] > max) max = outputGradients[i];
    }

    printf("FCB ");
    gradients.push_back(FullyConnectedLayerBackward(layers[15], weights[15], layers[14], outputGradients));
    outputGradients.clear();
    printf("FCB ");
    gradients.push_back(FullyConnectedLayerBackward(layers[14], weights[14], layers[13], gradients[0]->inputsGradients));
    gradients[0]->inputsGradients.clear();
    printf("FCB ");
    gradients.push_back(FullyConnectedLayerBackward(layers[13], weights[13], poolingLayers[4], gradients[1]->inputsGradients));
    gradients[1]->inputsGradients.clear();

    printf("MPB ");
    MaxPoolingBackward(poolingLayers[4], layers[12], gradients[2]->inputsGradients, ivec2(2, 2), ivec2(2, 2));
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[12], weights[12], layers[11], gradients[2]->inputsGradients));
    gradients[2]->inputsGradients.clear();
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[11], weights[11], layers[10], gradients[3]->inputsGradients));
    gradients[3]->inputsGradients.clear();
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[10], weights[10], poolingLayers[3], gradients[4]->inputsGradients));
    gradients[4]->inputsGradients.clear();

    printf("MPB ");
    MaxPoolingBackward(poolingLayers[3], layers[9], gradients[5]->inputsGradients, ivec2(2, 2), ivec2(2, 2));
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[9], weights[9], layers[8], gradients[5]->inputsGradients));
    gradients[5]->inputsGradients.clear();
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[8], weights[8], layers[7], gradients[6]->inputsGradients));
    gradients[6]->inputsGradients.clear();
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[7], weights[7], poolingLayers[2], gradients[7]->inputsGradients));
    gradients[7]->inputsGradients.clear();

    printf("MPB ");
    MaxPoolingBackward(poolingLayers[2], layers[6], gradients[8]->inputsGradients, ivec2(2, 2), ivec2(2, 2));
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[6], weights[6], layers[5], gradients[8]->inputsGradients));
    gradients[8]->inputsGradients.clear();
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[5], weights[5], layers[4], gradients[9]->inputsGradients));
    gradients[9]->inputsGradients.clear();
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[4], weights[4], poolingLayers[1], gradients[10]->inputsGradients));
    gradients[10]->inputsGradients.clear();

    printf("MPB ");
    MaxPoolingBackward(poolingLayers[1], layers[3], gradients[11]->inputsGradients, ivec2(2, 2), ivec2(2, 2));
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[3], weights[3], layers[2], gradients[11]->inputsGradients));
    gradients[11]->inputsGradients.clear();
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[2], weights[2], poolingLayers[0], gradients[12]->inputsGradients));
    gradients[12]->inputsGradients.clear();

    printf("MPB ");
    MaxPoolingBackward(poolingLayers[0], layers[1], gradients[13]->inputsGradients, ivec2(2, 2), ivec2(2, 2));
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[1], weights[1], layers[0], gradients[13]->inputsGradients));
    gradients[13]->inputsGradients.clear();
    printf("CLB ");
    gradients.push_back(ConvolutionLayerBackward(layers[0], weights[0], loadedData, gradients[14]->inputsGradients));
    gradients[14]->inputsGradients.clear();

    printf("\n");
}

Layer* NeuralNetworkManager::GetLoadedImageWithSize(int outWidth, int outHeight) {
    int imageSize = RenderingManager::GetInstance()->currentlyRenderedImage.size();
    unsigned char* image = new unsigned char[imageSize];
    std::memcpy(image, &RenderingManager::GetInstance()->currentlyRenderedImage[0], imageSize);

    glm::ivec2 imageResolution = EditorManager::GetInstance()->loadedImage->GetComponentByClass<Image>()->GetTexture()->GetResolution();

    unsigned char* resizedImage = CUM::ResizeImage(image, imageResolution.x, imageResolution.y, outWidth, outHeight);
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
        output->maps[counter] = (float)image[i] / 127.5f - 1;
        output->maps[counter + outWidth * outHeight] = (float)image[i + 1] / 127.5f - 1;
        output->maps[counter + 2 * outWidth * outHeight] = (float)image[i + 2] / 127.5f - 1;
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
        manager->weights.emplace_back(new Group(64, 0, 0, ivec3(3, 3, 3)));
        manager->weights.emplace_back(new Group(64, 0, 0, ivec3(3, 3, 64)));
        manager->weights.emplace_back(new Group(128, 0, 0, ivec3(3, 3, 64)));
        manager->weights.emplace_back(new Group(128, 0, 0, ivec3(3, 3, 128)));
        manager->weights.emplace_back(new Group(256, 0, 0, ivec3(3, 3, 128)));
        manager->weights.emplace_back(new Group(256, 0, 0, ivec3(3, 3, 256)));
        manager->weights.emplace_back(new Group(256, 0, 0, ivec3(3, 3, 256)));
        manager->weights.emplace_back(new Group(512, 0, 0, ivec3(3, 3, 256)));
        manager->weights.emplace_back(new Group(512, 0, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(512, 0, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(512, 0, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(512, 0, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(512, 0, 0, ivec3(3, 3, 512)));
        manager->weights.emplace_back(new Group(1, 0, 0, ivec3(102760448, 1, 1)));
        manager->weights.emplace_back(new Group(1, 0, 0, ivec3(16777216, 1, 1)));
        manager->weights.emplace_back(new Group(1, 0, 0, ivec3(4096 * manager->outputSize, 1, 1)));
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
        manager->weights.emplace_back(new Group(64, 27, 3211264, ivec3(3, 3, 3), true));
        manager->weights.emplace_back(new Group(64, 3211264, 3211264, ivec3(3, 3, 64), true));
        manager->weights.emplace_back(new Group(128, 802816, 1605632, ivec3(3, 3, 64), true));
        manager->weights.emplace_back(new Group(128, 1605632, 1605632, ivec3(3, 3, 128), true));
        manager->weights.emplace_back(new Group(256, 401408, 802816, ivec3(3, 3, 128), true));
        manager->weights.emplace_back(new Group(256, 802816, 802816, ivec3(3, 3, 256), true));
        manager->weights.emplace_back(new Group(256, 802816, 802816, ivec3(3, 3, 256), true));
        manager->weights.emplace_back(new Group(512, 200704, 401408, ivec3(3, 3, 256), true));
        manager->weights.emplace_back(new Group(512, 401408, 401408, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, 401408, 401408, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, 100352, 100352, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, 100352, 100352, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, 100352, 100352, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(1, 25088, 4096, ivec3(102760448, 1, 1), true));
        manager->weights.emplace_back(new Group(1, 4096, 4096, ivec3(16777216, 1, 1), true));
        manager->weights.emplace_back(new Group(1, 4096, 2, ivec3(4096 * manager->outputSize, 1, 1), true));
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