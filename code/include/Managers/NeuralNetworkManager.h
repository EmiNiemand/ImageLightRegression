#ifndef IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H
#define IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H

#include "NeuralNetwork/CUDAFunctions.cuh"

#include "glm/glm.hpp"

#include <vector>
#include <mutex>
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
    float trainingParameters[7] = {2000, 100000, 20, 25, 0.0001, 0.00000001, 0.25};

    bool waitForUpdate = false;
    bool waitForRender = false;

private:
    inline static NeuralNetworkManager* neuralNetworkManager;

    std::thread* thread = nullptr;

    Layer* loadedImage = nullptr;

    std::vector<Layer*> layers;
    std::vector<Layer*> pooledLayers;
    std::vector<Group*> weights;
    std::vector<Layer*> biases;

    NetworkTask currentTask = None;
    NetworkState state = Idle;

    int outputSize = 0;

    bool finalize = false;

public:
    NeuralNetworkManager(NeuralNetworkManager &other) = delete;
    void operator=(const NeuralNetworkManager&) = delete;
    virtual ~NeuralNetworkManager();

    static NeuralNetworkManager* GetInstance();
    [[nodiscard]] NetworkState GetState() const;

    void Startup();
    void Run();
    void Shutdown();

    void InitializeNetwork(NetworkTask task);
    void FinalizeNetwork();

private:
    explicit NeuralNetworkManager();

    void ProcessImage();

    void Train();
    static void ThreadTrain(int epoch, int trainingSize, int batchSize, int patience, float learningRate, float minLearningRate);

    static void FillDataSet(float* dataSet, glm::vec3* cameraPositions, glm::vec3* lightPositions, int dataSize, int trainingSize);

    void Forward(bool drop);
    void Backward(const float* target, std::vector<Gradient*>& gradients);

    static Layer* GetLoadedImageWithSize(int outWidth, int outHeight);

    // Loads weights and biases
    void Load();
    static void ThreadLoad();
    // Saves weights and biases to file
    void Save();
    static void ThreadSave();
};


#endif //IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H
