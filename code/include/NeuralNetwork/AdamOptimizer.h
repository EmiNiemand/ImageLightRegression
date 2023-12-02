#ifndef IMAGELIGHTREGRESSION_ADAMOPTIMIZER_H
#define IMAGELIGHTREGRESSION_ADAMOPTIMIZER_H

#include <vector>

class AdamOptimizer {
private:
    inline static AdamOptimizer* adamOptimizer = nullptr;
    std::vector<float> m, v;  // First and second moment estimates
    float beta1, beta2;       // Exponential decay rates
    float learningRate;       // Learning rate
    float epsilon;            // Small constant to avoid division by zero
    int t = 0;                // Time step

public:
    AdamOptimizer(AdamOptimizer &other) = delete;
    void operator=(const AdamOptimizer&) = delete;
    virtual ~AdamOptimizer();

    static AdamOptimizer* GetInstance();

    void Startup();
    void Shutdown();

    void UpdateParameters(float* parameters, int size, const std::vector<float>& gradients);

    void IncrementTimeStep();
    void SetLearningRate(float inLearningRate);

private:
    explicit AdamOptimizer(float beta1 = 0.9, float beta2 = 0.999, float learningRate = 0.001, float epsilon = 1e-8);

};


#endif //IMAGELIGHTREGRESSION_ADAMOPTIMIZER_H
