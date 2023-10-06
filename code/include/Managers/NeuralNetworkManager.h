#ifndef IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H
#define IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H

#include <vector>

class DirectionalLight;
class PointLight;
class SpotLight;

class NeuralNetworkManager {
private:
    inline static NeuralNetworkManager* neuralNetworkManager;

    std::vector<float> neurons;
    std::vector<float> weights;
    std::vector<float> lambdas;

public:
    NeuralNetworkManager(NeuralNetworkManager &other) = delete;
    void operator=(const NeuralNetworkManager&) = delete;
    virtual ~NeuralNetworkManager();

    static NeuralNetworkManager* GetInstance();

    void Startup();
    void Update();
    void Shutdown();

private:
    explicit NeuralNetworkManager();

    void SetLightValues();

    void DivideDirectionalLightToNeurons(DirectionalLight* light);
    void DividePointLightToNeurons(PointLight* light);
    void DivideSpotLightToNeurons(SpotLight* light);
};


#endif //IMAGELIGHTREGRESSION_NEURALNETWORKMANAGER_H
