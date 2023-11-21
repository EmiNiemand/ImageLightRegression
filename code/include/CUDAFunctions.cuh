#ifndef IMAGELIGHTREGRESSION_CUDAFUNCTIONS_CUH
#define IMAGELIGHTREGRESSION_CUDAFUNCTIONS_CUH

#include <cuda_runtime.h>

#pragma region Structs
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

    Group(int filtersCount, int seed, ivec3 filterDim, bool fillData = false) {
        count = filtersCount;

        std::srand(seed);

        filters = new Layer[count];


    for (int i = 0; i < count; ++i) {
            filters[i].width = filterDim.x;
            filters[i].height = filterDim.y;
            filters[i].depth = filterDim.z;

            int filterSize = filterDim.x * filterDim.y * filterDim.z;

            filters[i].maps = new float[filterSize]();

            if (fillData) {
                for (int j = 0; j < filterSize; ++j) {
                    filters[i].maps[j] = (float)((std::rand() % 1000) - 500)/ 50000.0f;
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
extern __global__ void CUDAConvLayer(const float* input, float* output, const float* kernel, const float* biases, int inputDimX,
                                     int inputDimY, int outputDimX, int outputDimY, int kernelDimX,int kernelDimY, int kernelDimZ,
                                     int strideDimX, int strideDimY, int paddingDimX, int paddingDimY, int filterCount);

extern __global__ void CUDAReLULayer(float* input, int size);

extern __global__ void CUDAPoolingLayer(const float* input, float* output, int outputDimX, int outputDimY, int outputDimZ,
                                        int poolDimX, int poolDimY, int strideDimX, int strideDimY);

extern __global__ void CUDAFullyConnectedLayer(const float* input, const float* weights, const float* biases,
                                               float* output, int inputSize, int outputSize);

__global__ void CUDARecalculateConvWeightsAndGradient(float* weights, const float* gradients, float* outputGradients,
                                                      const float* previousLayer, int weightSize, int outGradientWidth,
                                                      int outGradientHeight, int outGradientDepth, float learningRate);

extern __global__ void CUDARecalculateConvBiases(float* biases, const float* gradient, int size, int depth, float learningRate);
#pragma endregion

Layer* ConvolutionLayer(const Layer* input, const Group* filters, const ivec2& stride = {1, 1},
                        const ivec2& padding = {0, 0}, const float* biases = nullptr);

void ConvolutionLayerBackward(Layer* currentLayer, Group* weights, Layer* biases, Layer* previousLayer,
                                 float*& gradient, float learningRate);

void ReLULayer(Layer* filter);

Layer* PoolingLayer(const Layer* input, const ivec2& poolDim, const ivec2& stride);

void MaxPoolingBackward(const Layer* currentLayer, const Layer* previousLayer, float*& gradient, ivec2 poolDim);

Layer* FullyConnectedLayer(const Layer* neurons, const float* weights, int inputSize, int outputSize,
                           const float* biases = nullptr);

void FullyConnectedLayerBackward(Layer* currentLayer, Group* weights, Layer* biases, Layer* previousLayer,
                                 float*& gradient, float learningRate);

float MSELossFunction(const float* input, const float* predictedResult, int size);




#endif //IMAGELIGHTREGRESSION_CUDAFUNCTIONS_CUH
