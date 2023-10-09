#ifndef IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H
#define IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H

#include <vector>

class DirectionalLight;
class PointLight;
class SpotLight;

class NeuralNetworkManager {
private:
    inline static NeuralNetworkManager* neuralNetworkManager;

    char* loadedImage;

    std::vector<float> neurons;
    std::vector<float> outputs;
    std::vector<float> weights;
    std::vector<float> weightsCopy;
    std::vector<float> biases;

    float previousAccuracy = 0;

    int iteration = 0;

    int directionalLightsNumber = 0;
    int pointLightsNumber = 0;
    int spotLightsNumber = 0;


public:
    NeuralNetworkManager(NeuralNetworkManager &other) = delete;
    void operator=(const NeuralNetworkManager&) = delete;
    virtual ~NeuralNetworkManager();

    static NeuralNetworkManager* GetInstance();

    void Startup();
    void Shutdown();

    void InitializeNetwork();
    void Finalize();

    void PreRenderUpdate();
    void PostRenderUpdate();

private:
    explicit NeuralNetworkManager();

    float CalculateAverage(char* data, int size);
    void CalculateWeights();
    void CalculateOutputs();
    bool CheckOutputValues();

    void SetLightValues();
    void DivideLightValues();

    void CombineNeuronsToDirectionalLight(DirectionalLight* light, int index);
    void CombineNeuronsToPointLight(PointLight* light, int index);
    void CombineNeuronsToSpotLight(SpotLight* light, int index);

    void DivideDirectionalLightToNeurons(DirectionalLight* light);
    void DividePointLightToNeurons(PointLight* light);
    void DivideSpotLightToNeurons(SpotLight* light);
};


#endif //IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H
