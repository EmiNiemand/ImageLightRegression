#ifndef IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H
#define IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H

#include "CUDAFunctions.cuh"

#include "glm/glm.hpp"

#include <vector>
#include <thread>

class DirectionalLight;
class PointLight;
class SpotLight;

enum NetworkState {
    Idle,
    Processing,
    Training,
    LoadingSaving
};

enum NetworkTask {
    None,
    ProcessImage,
    TrainNetwork
};

class NeuralNetworkManager {
public:
    NetworkState state = Idle;

private:
    inline static NeuralNetworkManager* neuralNetworkManager;

    std::thread* thread = nullptr;

    Layer* loadedData = nullptr;

    std::vector<Layer*> layers;
    std::vector<Layer*> poolingLayers;
    std::vector<Group*> weights;
    std::vector<Layer*> biases;

    NetworkTask currentTask = None;

    int iteration = 0;

    int outputSize = 0;
public:
    NeuralNetworkManager(NeuralNetworkManager &other) = delete;
    void operator=(const NeuralNetworkManager&) = delete;
    virtual ~NeuralNetworkManager();

    static NeuralNetworkManager* GetInstance();

    void Startup();
    void Shutdown();

    void InitializeNetwork(NetworkTask task);
    void FinalizeNetwork();

private:
    explicit NeuralNetworkManager();

    void ProcessImage();

    void Train(int epoch, int trainingSize, float learningStep = 0.001f);
    static void ThreadTrain(int epoch, int trainingSize, float learningStep);
    static float* GenerateDataSet(int trainingSize, int networkOutputSize);

    void Forward();
    void Backward(float* predicted, float learningRate);

    static Layer* GetLoadedImageWithSize(int outWidth, int outHeight);

    void Load();
    static void ThreadLoad();
    void Save();
    static void ThreadSave();
};


#endif //IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H
