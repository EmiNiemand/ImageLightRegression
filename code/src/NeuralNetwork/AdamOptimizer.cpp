#include "NeuralNetwork/AdamOptimizer.h"

AdamOptimizer::AdamOptimizer(float beta1, float beta2, float learningRate, float epsilon)
        : beta1(beta1), beta2(beta2), learningRate(learningRate), epsilon(epsilon) {}

AdamOptimizer::~AdamOptimizer() = default;

AdamOptimizer *AdamOptimizer::GetInstance() {
    if (adamOptimizer == nullptr) {
        adamOptimizer = new AdamOptimizer();
    }
    return adamOptimizer;
}

void AdamOptimizer::Startup() {

}

void AdamOptimizer::Shutdown() {
    delete adamOptimizer;
}

void AdamOptimizer::UpdateParameters(float* parameters, int size, const std::vector<float>& gradients) {
    std::vector<float> m, v;
    m.resize(size, 0.0f);
    v.resize(size, 0.0f);

    for (int i = 0; i < size; ++i) {
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
        v[i] = beta2 * v[i] + (1 - beta2) * (float)pow(gradients[i], 2);
    }

    float biasCorrection1 = 1.0f - (float)pow(beta1, t);
    float biasCorrection2 = 1.0f - (float)pow(beta2, t);

    for (int i = 0; i < size; ++i) {
        float mOut = m[i] / biasCorrection1;
        float vOut = v[i] / biasCorrection2;

        parameters[i] -= (learningRate / ((float)sqrt((double)vOut) + epsilon)) * mOut;
    }
    
    m.clear();
    v.clear();
}

void AdamOptimizer::IncrementTimeStep() {
    ++t;
}

void AdamOptimizer::SetLearningRate(float inLearningRate) {
    learningRate = inLearningRate;
}
