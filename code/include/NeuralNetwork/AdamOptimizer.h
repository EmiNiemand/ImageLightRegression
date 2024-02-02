#ifndef IMAGELIGHTREGRESSION_ADAMOPTIMIZER_H
#define IMAGELIGHTREGRESSION_ADAMOPTIMIZER_H

#include <vector>

class AdamOptimizer {
public:
    float beta1 = 0.9;
    float beta2 = 0.999;
    float learningRate = 0.001;

private:
    inline static AdamOptimizer* adamOptimizer = nullptr;

    float epsilon = 1e-8;            // Small constant to avoid division by zero
    int t = 0;                // Time step

public:
    AdamOptimizer(AdamOptimizer &other) = delete;
    void operator=(const AdamOptimizer&) = delete;
    virtual ~AdamOptimizer();

    static AdamOptimizer* GetInstance();

    void Startup();
    void Shutdown();

    void Reset();

    void UpdateParameters(float* parameters, int size, const std::vector<float>& gradients);

    void IncrementTimeStep();

private:
    explicit AdamOptimizer();

};


#endif //IMAGELIGHTREGRESSION_ADAMOPTIMIZER_H
