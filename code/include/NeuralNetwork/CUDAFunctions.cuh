#ifndef IMAGELIGHTREGRESSION_CUDAFUNCTIONS_CUH
#define IMAGELIGHTREGRESSION_CUDAFUNCTIONS_CUH

#include <cuda_runtime.h>
#include <vector>
#include <random>

#define M_PI 3.14159265358979323846
#define CLIP_VALUE 5

#pragma region Structs
struct Gradient {
    std::vector<float> weightsGradients;
    std::vector<float> biasesGradients;
    std::vector<float> inputsGradients;
};

struct ivec2 {
    int x = 0, y = 0;
};

struct ivec3 {
    int x = 0, y = 0, z = 0;
};

struct Layer {
    float* maps;

    int width;
    int height;
    int depth;

    Layer() {
        maps = nullptr;
        width = 0;
        height = 0;
        depth = 0;
    }

    Layer(int inWidth, int inHeight, int inDepth) {
        width = inWidth;
        height = inHeight;
        depth = inDepth;

        maps = new float[width * height * depth]();
    }

    ~Layer() {
        delete[] maps;
    }
};

struct Group {
    Layer* filters;

    int count;
    Group() {
        count = 0;
        filters = nullptr;
    }

    Group(int filtersCount, float prevLayerSize, float currLayerSize, ivec3 filterDim, bool fillData = false) {
        count = filtersCount;

        filters = new Layer[count];


    for (int i = 0; i < count; ++i) {
            filters[i].width = filterDim.x;
            filters[i].height = filterDim.y;
            filters[i].depth = filterDim.z;

            int filterSize = filterDim.x * filterDim.y * filterDim.z;

            filters[i].maps = new float[filterSize]();

            if (fillData) {
                std::random_device rd;
                std::mt19937 gen(rd());
                float a = std::sqrt(6 / (prevLayerSize + currLayerSize));

                std::uniform_real_distribution<float> distribution(-a, a);

                for (int j = 0; j < filterSize; ++j) {
                    filters[i].maps[j] = distribution(gen);
                }
            }
        }
    }

    ~Group() {
        delete[] filters;
    }
};
#pragma endregion

#pragma region CUDA functions
extern __global__ void CUDAConvLayer(const float* input, float* output, const float* kernel, const float* biases,
                                     int inputDimX, int inputDimY, int outputDimX, int outputDimY, int kernelDimX,
                                     int kernelDimY, int kernelDimZ, int strideDimX, int strideDimY, int paddingDimX,
                                     int paddingDimY, int kernelNumber);

extern __global__ void CUDAReLULayer(float* input, int size);

extern __global__ void CUDAPoolingLayer(const float* input, float* output, int outputDimX, int outputDimY, int outputDimZ,
                                        int poolDimX, int poolDimY, int strideDimX, int strideDimY);

extern __global__ void CUDAFullyConnectedLayer(const float* input, const float* weights, const float* biases,
                                               float* output, int inputSize, int outputSize);

extern __global__ void CUDAConvLayerGradients(float* prevGradients, float* weightGradients, const float* currentGradients,
                                              const float* prevLayer, const float* weights, int prevWidth, int prevHeight,
                                              int prevDepth, int currentWidth, int currentHeight, int currentDepth,
                                              int kernelWidth, int kernelHeight);

extern __global__ void CUDAConvLayerBiasGradients(float* biasGradients, const float* currentGradients, int currentWidth,
                                                  int currentHeight, int currentDepth);

extern __global__ void CUDAClipGradient(float* gradient, int size, float maxValue);
#pragma endregion

Layer* ConvolutionLayer(const Layer* input, const Group* filters, const ivec2& stride = {1, 1},
                        const ivec2& padding = {0, 0}, const float* biases = nullptr);

Gradient* ConvolutionLayerBackward(Layer* currentLayer, Group* weights, Layer* previousLayer, std::vector<float>& gradient);

void ReLULayer(Layer* filter);

Layer* PoolingLayer(const Layer* input, const ivec2& poolDim, const ivec2& stride);

void MaxPoolingBackward(const Layer* currentLayer, const Layer* previousLayer, std::vector<float>& gradient,
                                      ivec2 poolDim, ivec2 strideDim);

Layer* FullyConnectedLayer(const Layer* neurons, const float* weights, int inputSize, int outputSize,
                           const float* biases = nullptr);

Gradient* FullyConnectedLayerBackward(Layer* currentLayer, Group* weights, Layer* previousLayer, std::vector<float>& gradient);

float MSELossFunction(const float* input, const float* predictedResult, int size);

void MiniBatch(const std::vector<std::vector<Gradient*>>& gradients, std::vector<Group*>& weights,
               std::vector<Layer*>& biases, float learningRate);

void UpdateWeightsAndBiases(const std::vector<Gradient*>& gradients, std::vector<Group*>& weights,
               std::vector<Layer*>& biases, float learningRate);

void ClipGradient(std::vector<float>& gradient);



#endif //IMAGELIGHTREGRESSION_CUDAFUNCTIONS_CUH
