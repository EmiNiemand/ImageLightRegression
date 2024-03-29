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

#include <chrono>

NeuralNetworkManager::NeuralNetworkManager() = default;

NeuralNetworkManager::~NeuralNetworkManager() = default;

NeuralNetworkManager* NeuralNetworkManager::GetInstance() {
    if (neuralNetworkManager == nullptr) {
        neuralNetworkManager = new NeuralNetworkManager();
    }
    return neuralNetworkManager;
}

NetworkState NeuralNetworkManager::GetState() const {
    return state;
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

    delete loadedImage;
    delete neuralNetworkManager;
}

void NeuralNetworkManager::InitializeNetwork(NetworkTask task) {
    layers.reserve(16);
    pooledLayers.reserve(4);

    currentTask = task;

    if (task == NetworkTask::TrainNetwork) {
        Train();
//        Test();
    }
    else if (task == NetworkTask::ProcessImage) {
        ProcessImage();
    }
}

void NeuralNetworkManager::FinalizeNetwork() {
    if (currentTask == NetworkTask::TrainNetwork) {
        Save();
    }

    AdamOptimizer::GetInstance()->Reset();

    DELETE_VECTOR_VALUES(layers)
    DELETE_VECTOR_VALUES(pooledLayers)

    currentTask = None;
    state = Idle;
}

void NeuralNetworkManager::ProcessImage() {
    state = Processing;
    RenderingManager* renderingManager = RenderingManager::GetInstance();
    if (renderingManager->objectRenderer->pointLights[0] == nullptr) FinalizeNetwork();

    Application* application = Application::GetInstance();
    bool frameValue;

    application->drawNewRenderedImage = true;
    RenderingManager::GetInstance()->DrawFrame();

    Forward(false);

    glm::vec3 cameraPosition = Camera::GetRenderingCamera()->transform->GetGlobalPosition();
    float* cameraSphericalCoords = CUM::CartesianCoordsToSphericalAngles(cameraPosition);
    glm::vec3 lightPosition = CUM::SphericalAnglesToCartesianCoordinates(layers[15]->maps[0] + cameraSphericalCoords[0],
                                                                         layers[15]->maps[1] + cameraSphericalCoords[1],
                                                                         glm::length(cameraPosition));

    delete[] cameraSphericalCoords;
    delete loadedImage;
    loadedImage = nullptr;

    renderingManager->objectRenderer->pointLights[0]->parent->transform->SetLocalPosition(lightPosition);
}

void NeuralNetworkManager::Train() {
    if (thread != nullptr) {
        if (thread->joinable()) thread->join();
        delete thread;
    }

    thread = new std::thread(&NeuralNetworkManager::ThreadTrain, (int)trainingParameters[0], (int)trainingParameters[1],
                (int)trainingParameters[2], (int)trainingParameters[3], trainingParameters[4], trainingParameters[5]);
}

void NeuralNetworkManager::ThreadTrain(int epoch, int trainingSize, int batchSize, int patience, float learningRate,
                                                                                            float minLearningRate) {
    // Get neuralNetworkManager(as manager) and renderingManager
    NeuralNetworkManager* manager = NeuralNetworkManager::GetInstance();
    manager->state = NetworkState::Training;

    RenderingManager* renderingManager = RenderingManager::GetInstance();
    Application* application = Application::GetInstance();

    AdamOptimizer* adamOptimizer = AdamOptimizer::GetInstance();
    adamOptimizer->learningRate = learningRate;

    // Save texture values
    Texture* texture = EditorManager::GetInstance()->loadedImage->GetComponentByClass<Image>()->GetTexture();
    unsigned int prevImage = texture->GetID();
    glm::ivec2 prevImageResolution = texture->GetResolution();

    // Save Camera initial values
    Transform* renderingCameraTransform = Camera::GetRenderingCamera()->transform;
    glm::vec3 cPosition = renderingCameraTransform->GetLocalPosition();
    glm::vec3 cRotation = renderingCameraTransform->GetLocalRotation();

    // Set camera texture as new loaded image which is used in network forward method
    int width = Application::viewports[0].resolution.x;
    int height = Application::viewports[0].resolution.y;
    texture->SetID(renderingManager->objectRenderer->renderingCameraTexture);
    texture->SetResolution(glm::ivec2(width, height));

    int batchesNumber = trainingSize / batchSize;

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

        adamOptimizer->IncrementTimeStep();

        for (int j = 0; j < batchesNumber; ++j) {
            gradients.resize(batchSize);
            for (int k = 0; k < batchSize; ++k) {
                int idx = k + j * batchSize;

                // Set light position
                renderingManager->objectRenderer->pointLights[0]->parent->transform->SetLocalPosition(lightPositions[idx]);
                // Set camera position
                Camera::GetRenderingCamera()->transform->SetLocalPosition(cameraPositions[idx]);

                // Calculate camera looking direction and rotate it to look at point(0,0,0)
                glm::vec3 direction = glm::normalize(glm::vec3(0, 0, 0) - cameraPositions[idx]);
                float angleX = (float)(asin(direction.y) * 180.0f / M_PI);
                float angleY = (float)(-atan2(direction.x, -direction.z) * 180.0f / M_PI);
                Camera::GetRenderingCamera()->transform->SetLocalRotation(glm::vec3(angleX, angleY, 0));

                bool frameValue;

                application->mutex.lock();
                application->frameSwitch = false;
                application->mutex.unlock();

                std::this_thread::sleep_for(std::chrono::milliseconds(10));

                while (true) {
                    application->mutex.lock();
                    if (application->frameSwitch) {
                        application->mutex.unlock();
                        break;
                    }
                    application->mutex.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }

                manager->Forward(true);

                float predictedValues[2] = {dataSet[idx * 2], dataSet[idx * 2 + 1]};
                ILR_INFO_MSG("Output: " + STRING(manager->layers[15]->maps[0]) + ", " + STRING(manager->layers[15]->maps[1]) +
                             ", Target: " + STRING(predictedValues[0]) + ", " + STRING(predictedValues[1]));

                float loss = MSELossFunction(manager->layers[15]->maps, predictedValues, manager->outputSize);
                epochLoss += loss;

                manager->Backward(predictedValues, gradients[k]);

                // Clear network layers and loaded image
                DELETE_VECTOR_VALUES(manager->layers)
                DELETE_VECTOR_VALUES(manager->pooledLayers)

                delete manager->loadedImage;
                manager->loadedImage = nullptr;

                if (!Application::GetInstance()->isStarted) {
                    break;
                }
            }

            UpdateNetwork(gradients, manager->weights, manager->biases);

            // Clear gradients
            for (int g = 0; g < gradients.size(); ++g) {
                DELETE_VECTOR_VALUES(gradients[g])
            }
            gradients.clear();

            if (!Application::GetInstance()->isStarted) {
                break;
            }
        }


        float averageEpochLoss = epochLoss / (float)batchesNumber;

        ILR_WARN_MSG("**********************************");
        ILR_WARN_MSG("Epoch: " + STRING(i) + ", Loss: " + STRING(averageEpochLoss) + ", Rate: " + STRING(learningRate));
        ILR_WARN_MSG("**********************************");

        if (bestEpochLoss <= averageEpochLoss) {
            ++patienceCounter;
            if (patienceCounter == patience) {
                learningRate *= 0.1f;
                adamOptimizer->learningRate = learningRate;
                if (learningRate < minLearningRate) learningRate = minLearningRate;
                bestEpochLoss = averageEpochLoss;
                patienceCounter = 0;
            }
        }
        else if (bestEpochLoss > averageEpochLoss) {
            patienceCounter = 0;
            bestEpochLoss = averageEpochLoss;
        }

        if (!Application::GetInstance()->isStarted) {
            break;
        }
    }

    delete[] cameraPositions;
    delete[] lightPositions;
    delete[] dataSet;

    // Set old values for texture and camera
    texture->SetID(prevImage);
    texture->SetResolution(prevImageResolution);
    renderingCameraTransform->SetLocalPosition(cPosition);
    renderingCameraTransform->SetLocalRotation(cRotation);

    // Set network finalization to true
    if (Application::GetInstance()->isStarted) {
        Application::GetInstance()->isStarted = false;
    }
    manager->finalize = true;
}

void NeuralNetworkManager::FillDataSet(float *dataSet, glm::vec3* cameraPositions, glm::vec3* lightPositions, int dataSize,
                                                                                                    int trainingSize) {
    // Load values from file
    FILE* stream;
    fopen_s(&stream, "resources/Resources/NeuralNetworkResources/Data.json", "rb");
    if (stream != nullptr) {
        int size;
        fread(&size, sizeof(int), 1, stream);

        if (size == dataSize) {
            fseek(stream, (long)sizeof(int), SEEK_CUR);
            fread(dataSet, sizeof(float), dataSize, stream);
            fseek(stream, (long)(dataSize * sizeof(float)), SEEK_CUR);

            fread(cameraPositions, sizeof(glm::vec3), trainingSize, stream);
            fseek(stream, (long)(dataSize * sizeof(glm::vec3)), SEEK_CUR);

            fread(lightPositions, sizeof(glm::vec3), trainingSize, stream);
            fclose(stream);
            return;
        }
        fclose(stream);
    }

    // Create new values if file does not exist or size of data was changed
    std::filesystem::remove("resources/Resources/NeuralNetworkResources/DataLog.txt");

    std::shared_ptr<spdlog::logger> logger = spdlog::basic_logger_mt("logger", "resources/Resources/NeuralNetworkResources/DataLog.txt");
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::info);

    for (int i = 0; i < trainingSize; ++i) {
        cameraPositions[i] = glm::normalize(glm::vec3(RNG(-1.0f, 1.0f), RNG(0.0f, 1.0f), RNG(-1.0f, 1.0f))) * 10.0f;
        lightPositions[i] = glm::normalize(glm::vec3(RNG(-1.0f, 1.0f), RNG(0.0f, 1.0f), RNG(-1.0f, 1.0f))) * 10.0f;

        float* cameraSphericalAngles = CUM::CartesianCoordsToSphericalAngles(cameraPositions[i]);
        float* lightSphericalAngles = CUM::CartesianCoordsToSphericalAngles(lightPositions[i]);

        float phi = lightSphericalAngles[0] - cameraSphericalAngles[0];
        float theta = lightSphericalAngles[1] - cameraSphericalAngles[1];
        
        // Change range from (-2PI, 2PI) to (-PI, PI)
        if (phi > (float)M_PI) phi = phi - 2.0f * (float)M_PI;
        if (phi < -(float)M_PI) phi = phi + 2.0f * (float)M_PI;

        dataSet[i * 2] = phi;
        dataSet[i * 2 + 1] = theta;

        // Log information about Camera, Light and Angles between them to file in human-readable form
        logger->info("Camera: " + STRING_VEC3(cameraPositions[i]) + " Light: " + STRING_VEC3(lightPositions[i]) +
                     " Angles: " + STRING(phi) + ", " + STRING(theta));

        delete[] cameraSphericalAngles;
        delete[] lightSphericalAngles;
    }
    spdlog::drop("logger");
    logger.reset();

    // Save new generated data to file as bytes
    fopen_s(&stream, "resources/Resources/NeuralNetworkResources/Data.json", "wb");
    if (stream != nullptr) {
        fwrite(&dataSize, sizeof(int), 1, stream);
        fseek(stream, (long)sizeof(int), SEEK_CUR);

        fwrite(dataSet, sizeof(float), dataSize, stream);
        fseek(stream, (long)(dataSize * sizeof(float)), SEEK_CUR);

        fwrite(cameraPositions, sizeof(glm::vec3), trainingSize, stream);
        fseek(stream, (long)(dataSize * sizeof(glm::vec3)), SEEK_CUR);

        fwrite(lightPositions, sizeof(glm::vec3), trainingSize, stream);

        fclose(stream);
    }
}

void NeuralNetworkManager::Forward(bool drop) {
    glm::ivec2 imageDim = glm::ivec2(224, 224);
    // Get image data in 0 - 1 range
    loadedImage = GetLoadedImageWithSize(imageDim.x, imageDim.y);

    // Group 1
    // Conv 1
    layers.emplace_back(ConvolutionLayer(loadedImage, weights[0], {1, 1}, {1, 1}, biases[0]->maps));
    //ReLU
    ReLU(layers[0]);

    //Conv 2
    layers.emplace_back(ConvolutionLayer(layers[0], weights[1], {1, 1}, {1, 1}, biases[1]->maps));
    // ReLU
    ReLU(layers[1]);
    // Max Pooling [0]
    pooledLayers.emplace_back(MaxPoolingLayer(layers[1], {2, 2}, {2, 2}));

    // Group 2
    // Conv 3
    layers.emplace_back(ConvolutionLayer(pooledLayers[0], weights[2], {1, 1}, {1, 1}, biases[2]->maps));
    // ReLU
    ReLU(layers[2]);

    // Conv 4
    layers.emplace_back(ConvolutionLayer(layers[2], weights[3], {1, 1}, {1, 1}, biases[3]->maps));
    // ReLU
    ReLU(layers[3]);
    // Max Pooling [1]
    pooledLayers.emplace_back(MaxPoolingLayer(layers[3], {2, 2}, {2, 2}));


    // Group 3
    // Conv 5
    layers.emplace_back(ConvolutionLayer(pooledLayers[1], weights[4], {1, 1}, {1, 1}, biases[4]->maps));
    // ReLU
    ReLU(layers[4]);

    // Conv 6
    layers.emplace_back(ConvolutionLayer(layers[4], weights[5], {1, 1}, {1, 1}, biases[5]->maps));
    // ReLU
    ReLU(layers[5]);

    // Conv 7
    layers.emplace_back(ConvolutionLayer(layers[5], weights[6], {1, 1}, {1, 1}, biases[6]->maps));
    // ReLU
    ReLU(layers[6]);
    // Max Pooling [2]
    pooledLayers.emplace_back(MaxPoolingLayer(layers[6], {2, 2}, {2, 2}));


    // Group 4
    // Conv 8
    layers.emplace_back(ConvolutionLayer(pooledLayers[2], weights[7], {1, 1}, {1, 1}, biases[7]->maps));
    // ReLU
    ReLU(layers[7]);

    // Conv 9
    layers.emplace_back(ConvolutionLayer(layers[7], weights[8], {1, 1}, {1, 1}, biases[8]->maps));
    // ReLU
    ReLU(layers[8]);

    // Conv 10
    layers.emplace_back(ConvolutionLayer(layers[8], weights[9], {1, 1}, {1, 1}, biases[9]->maps));
    // ReLU
    ReLU(layers[9]);
    // Max Pooling [3]
    pooledLayers.emplace_back(MaxPoolingLayer(layers[9], {2, 2}, {2, 2}));


    // Group 5
    // Conv 11
    layers.emplace_back(ConvolutionLayer(pooledLayers[3], weights[10], {1, 1}, {1, 1}, biases[10]->maps));
    // ReLU
    ReLU(layers[10]);

    // Conv 12
    layers.emplace_back(ConvolutionLayer(layers[10], weights[11], {1, 1}, {1, 1}, biases[11]->maps));
    // ReLU
    ReLU(layers[11]);

    // Conv 13
    layers.emplace_back(ConvolutionLayer(layers[11], weights[12], {1, 1}, {1, 1}, biases[12]->maps));
    // ReLU
    ReLU(layers[12]);
    // Max Pooling [4]
    pooledLayers.emplace_back(MaxPoolingLayer(layers[12], {2, 2}, {2, 2}));

    // Neurons of FCL Layer 1
    layers.emplace_back(FullyConnectedLayer(pooledLayers[4], weights[13]->filters[0].maps, 25088, 4096, biases[13]->maps));
    ReLU(layers[13]);
    // Deactivates neurons during training
    if(drop) DropoutLayer(layers[13], trainingParameters[6]);

    // Neurons of FCL Layer 2
    layers.emplace_back(FullyConnectedLayer(layers[13], weights[14]->filters[0].maps, 4096, 4096, biases[14]->maps));
    ReLU(layers[14]);
    // Deactivates neurons during training
    if(drop) DropoutLayer(layers[14], trainingParameters[6]);

    // Neurons of FCL Layer 3 (Output neurons)
    layers.emplace_back(FullyConnectedLayer(layers[14], weights[15]->filters[0].maps, 4096, outputSize, biases[15]->maps));
}

void NeuralNetworkManager::Backward(const float* target, std::vector<Gradient*>& gradients) {
    std::vector<float> outputGradients;
    outputGradients.reserve(outputSize);

    for (int i = 0; i < layers[15]->width; ++i) {
        outputGradients.push_back((2.0f / (float)outputSize) * (layers[15]->maps[i] - target[i]));
    }

    // FCL Layers
    gradients.push_back(FullyConnectedLayerBackward(layers[15], weights[15], layers[14], outputGradients));
    outputGradients.clear();
    gradients.push_back(FullyConnectedLayerBackward(layers[14], weights[14], layers[13], gradients[0]->inputGradients));
    gradients[0]->inputGradients.clear();
    gradients.push_back(FullyConnectedLayerBackward(layers[13], weights[13], pooledLayers[4], gradients[1]->inputGradients));
    gradients[1]->inputGradients.clear();

    // Group 5
    MaxPoolingBackward(pooledLayers[4], layers[12], gradients[2]->inputGradients, ivec2(2, 2), ivec2(2, 2));
    gradients.push_back(ConvolutionLayerBackward(layers[12], weights[12], layers[11], gradients[2]->inputGradients));
    gradients[2]->inputGradients.clear();
    gradients.push_back(ConvolutionLayerBackward(layers[11], weights[11], layers[10], gradients[3]->inputGradients));
    gradients[3]->inputGradients.clear();
    gradients.push_back(ConvolutionLayerBackward(layers[10], weights[10], pooledLayers[3], gradients[4]->inputGradients));
    gradients[4]->inputGradients.clear();

    // Group 4
    MaxPoolingBackward(pooledLayers[3], layers[9], gradients[5]->inputGradients, ivec2(2, 2), ivec2(2, 2));
    gradients.push_back(ConvolutionLayerBackward(layers[9], weights[9], layers[8], gradients[5]->inputGradients));
    gradients[5]->inputGradients.clear();
    gradients.push_back(ConvolutionLayerBackward(layers[8], weights[8], layers[7], gradients[6]->inputGradients));
    gradients[6]->inputGradients.clear();
    gradients.push_back(ConvolutionLayerBackward(layers[7], weights[7], pooledLayers[2], gradients[7]->inputGradients));
    gradients[7]->inputGradients.clear();

    // Group 3
    MaxPoolingBackward(pooledLayers[2], layers[6], gradients[8]->inputGradients, ivec2(2, 2), ivec2(2, 2));
    gradients.push_back(ConvolutionLayerBackward(layers[6], weights[6], layers[5], gradients[8]->inputGradients));
    gradients[8]->inputGradients.clear();
    gradients.push_back(ConvolutionLayerBackward(layers[5], weights[5], layers[4], gradients[9]->inputGradients));
    gradients[9]->inputGradients.clear();
    gradients.push_back(ConvolutionLayerBackward(layers[4], weights[4], pooledLayers[1], gradients[10]->inputGradients));
    gradients[10]->inputGradients.clear();

    // Group 2
    MaxPoolingBackward(pooledLayers[1], layers[3], gradients[11]->inputGradients, ivec2(2, 2), ivec2(2, 2));
    gradients.push_back(ConvolutionLayerBackward(layers[3], weights[3], layers[2], gradients[11]->inputGradients));
    gradients[11]->inputGradients.clear();
    gradients.push_back(ConvolutionLayerBackward(layers[2], weights[2], pooledLayers[0], gradients[12]->inputGradients));
    gradients[12]->inputGradients.clear();

    // Group 1
    MaxPoolingBackward(pooledLayers[0], layers[1], gradients[13]->inputGradients, ivec2(2, 2), ivec2(2, 2));
    gradients.push_back(ConvolutionLayerBackward(layers[1], weights[1], layers[0], gradients[13]->inputGradients));
    gradients[13]->inputGradients.clear();
    gradients.push_back(ConvolutionLayerBackward(layers[0], weights[0], loadedImage, gradients[14]->inputGradients));
    gradients[14]->inputGradients.clear();
    gradients[15]->inputGradients.clear();
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

    for (int i = 0; i < outWidth * outHeight; ++i) {
        output->maps[i] = (float)image[i * 3] / 255;
        output->maps[i + outWidth * outHeight] = (float)image[i * 3 + 1] / 255;
        output->maps[i + outWidth * outHeight * 2] = (float)image[i * 3 + 2] / 255;
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
        manager->weights.emplace_back(new Group(64, 3211264, ivec3(3, 3, 3), true));
        manager->weights.emplace_back(new Group(64, 3211264, ivec3(3, 3, 64), true));
        manager->weights.emplace_back(new Group(128, 1605632, ivec3(3, 3, 64), true));
        manager->weights.emplace_back(new Group(128, 1605632, ivec3(3, 3, 128), true));
        manager->weights.emplace_back(new Group(256, 802816, ivec3(3, 3, 128), true));
        manager->weights.emplace_back(new Group(256, 802816, ivec3(3, 3, 256), true));
        manager->weights.emplace_back(new Group(256, 802816, ivec3(3, 3, 256), true));
        manager->weights.emplace_back(new Group(512, 401408, ivec3(3, 3, 256), true));
        manager->weights.emplace_back(new Group(512, 401408, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, 401408, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, 100352, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, 100352, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(512, 100352, ivec3(3, 3, 512), true));
        manager->weights.emplace_back(new Group(1, 4096, ivec3(102760448, 1, 1), true));
        manager->weights.emplace_back(new Group(1, 4096, ivec3(16777216, 1, 1), true));
        manager->weights.emplace_back(new Group(1, 2, ivec3(4096 * manager->outputSize, 1, 1), true));
    }
    manager->state = Idle;
}

void NeuralNetworkManager::Save() {
    if (thread != nullptr) {
        if (thread->joinable()) thread->join();
        delete thread;
    }

    thread = new std::thread(&NeuralNetworkManager::ThreadSave);
}

void NeuralNetworkManager::ThreadSave() {
    NeuralNetworkManager* manager = NeuralNetworkManager::GetInstance();

    manager->state = LoadingSaving;

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
    manager->state = Idle;
}
#pragma endregion

//void NeuralNetworkManager::Test() {
//    if (thread != nullptr) {
//        if (thread->joinable()) thread->join();
//        delete thread;
//    }
//
//    thread = new std::thread(&NeuralNetworkManager::ThreadTest);
//}
//
//void NeuralNetworkManager::ThreadTest() {
//    // Get neuralNetworkManager(as manager) and renderingManager
//    NeuralNetworkManager* manager = NeuralNetworkManager::GetInstance();
//    manager->state = NetworkState::Training;
//
//    RenderingManager* renderingManager = RenderingManager::GetInstance();
//    Application* application = Application::GetInstance();
//
//    // Save texture values
//    Texture* texture = EditorManager::GetInstance()->loadedImage->GetComponentByClass<Image>()->GetTexture();
//    unsigned int prevImage = texture->GetID();
//    glm::ivec2 prevImageResolution = texture->GetResolution();
//
//    // Save Camera initial values
//    Transform* renderingCameraTransform = Camera::GetRenderingCamera()->transform;
//    glm::vec3 cPosition = renderingCameraTransform->GetLocalPosition();
//    glm::vec3 cRotation = renderingCameraTransform->GetLocalRotation();
//
//    // Set camera texture as new loaded image which is used in network forward method
//    int width = Application::viewports[0].resolution.x;
//    int height = Application::viewports[0].resolution.y;
//    texture->SetID(renderingManager->objectRenderer->renderingCameraTexture);
//    texture->SetResolution(glm::ivec2(width, height));
//
//    // Generate or read data set
//    glm::vec2* dataSet = new glm::vec2[72 * 121];
//    glm::vec3* cameraPositions = new glm::vec3[72];
//    glm::vec3* lightPositions = new glm::vec3[121];
//
//    float radOfDeg15 = 15 * M_PI / 180;
//
//    float cTheta = M_PI / 2 - radOfDeg15;
//    for (int i = 0; i < 3; ++i) {
//        if (i > 0) {
//            cTheta -= (radOfDeg15 * 2);
//        }
//        for (int j = 0; j < 24; ++j) {
//            float cPhi = M_PI * 2 - radOfDeg15 * (j + 1);
//
//            cameraPositions[j + i * 24] = CUM::SphericalAnglesToCartesianCoordinates(cPhi, cTheta) * 10.0f;
//
//            for (int k = 0; k < 6; ++k) {
//                float lTheta = M_PI / 2 - radOfDeg15 * (k + 1);
//                if (k < 5) {
//                    for (int l = 0; l < 24; ++l) {
//                        float lPhi = M_PI * 2 - radOfDeg15 * (l + 1);
//
//                        if (i == 0) lightPositions[l + k * 24] = CUM::SphericalAnglesToCartesianCoordinates(lPhi, lTheta) * 10.0f;
//
//                        float phi = lPhi - cPhi;
//                        float theta = lTheta - cTheta;
//
//                        if (phi > (float)M_PI) phi = phi - 2.0f * (float)M_PI;
//                        if (phi < -(float)M_PI) phi = phi + 2.0f * (float)M_PI;
//
//                        dataSet[l + k * 24 + (j + i * 24) * 121] = glm::vec2(phi, theta);
//                    }
//                }
//                else {
//                    float lPhi = 0;
//
//                    if (i == 0) lightPositions[120] = CUM::SphericalAnglesToCartesianCoordinates(lPhi, lTheta) * 10.0f;
//
//                    float phi = lPhi - cPhi;
//                    float theta = lTheta - cTheta;
//
//                    if (phi > (float)M_PI) phi = phi - 2.0f * (float)M_PI;
//                    if (phi < -(float)M_PI) phi = phi + 2.0f * (float)M_PI;
//
//                    dataSet[120 + (j + i * 24) * 121] = glm::vec2(phi, theta);
//                }
//
//            }
//        }
//    }
//
//    float loss = 0;
//    glm::vec2 accuracy = {0, 0};
//
//    for (int i = 0; i < 72; ++i) {
//        for (int j = 0; j < 121; ++j) {
//            // Set light position
//            renderingManager->objectRenderer->pointLights[0]->parent->transform->SetLocalPosition(lightPositions[j]);
//            // Set camera position
//            Camera::GetRenderingCamera()->transform->SetLocalPosition(cameraPositions[i]);
//
//            // Calculate camera looking direction and rotate it to look at point(0,0,0)
//            glm::vec3 direction = glm::normalize(glm::vec3(0, 0, 0) - cameraPositions[i]);
//            float angleX = (float)(asin(direction.y) * 180.0f / M_PI);
//            float angleY = (float)(-atan2(direction.x, -direction.z) * 180.0f / M_PI);
//            Camera::GetRenderingCamera()->transform->SetLocalRotation(glm::vec3(angleX, angleY, 0));
//
//            bool frameValue;
//
//            application->mutex.lock();
//            application->frameSwitch = false;
//            application->mutex.unlock();
//
//            std::this_thread::sleep_for(std::chrono::milliseconds(10));
//
//            while (true) {
//                application->mutex.lock();
//                if (application->frameSwitch) {
//                    application->mutex.unlock();
//                    break;
//                }
//                application->mutex.unlock();
//                std::this_thread::sleep_for(std::chrono::milliseconds(2));
//            }
//
//            manager->Forward(false);
//
//            ILR_INFO_MSG("Output: " + STRING(manager->layers[15]->maps[0]) + ", " + STRING(manager->layers[15]->maps[1]) +
//                         ", Target: " + STRING(dataSet[j + i * 121].x) + ", " + STRING(dataSet[j + i * 121].y));
//
//            float target[2] = {dataSet[j + i * 121].x, dataSet[j + i * 121].y};
//            loss += (MSELossFunction(manager->layers[15]->maps, target, manager->outputSize));
//            accuracy.x += abs(dataSet[j + i * 121].x - manager->layers[15]->maps[0]);
//            accuracy.y += abs(dataSet[j + i * 121].y - manager->layers[15]->maps[1]);
//
//            // Clear network layers and loaded image
//            DELETE_VECTOR_VALUES(manager->layers)
//            DELETE_VECTOR_VALUES(manager->pooledLayers)
//
//            delete manager->loadedImage;
//            manager->loadedImage = nullptr;
//            if (!Application::GetInstance()->isStarted) {
//                break;
//            }
//        }
//
//        if (!Application::GetInstance()->isStarted) {
//            break;
//        }
//    }
//
//    ILR_WARN_MSG("**********************************");
//    ILR_WARN_MSG("Loss: " + STRING((loss / 8712)) + ", Accuracy: " + STRING_VEC2((accuracy / 8712.0f)));
//    ILR_WARN_MSG("**********************************");
//
//    delete[] cameraPositions;
//    delete[] lightPositions;
//    delete[] dataSet;
//
//    // Set old values for texture and camera
//    texture->SetID(prevImage);
//    texture->SetResolution(prevImageResolution);
//    renderingCameraTransform->SetLocalPosition(cPosition);
//    renderingCameraTransform->SetLocalRotation(cRotation);
//
//    manager->currentTask = None;
//    manager->state = Idle;
//}
