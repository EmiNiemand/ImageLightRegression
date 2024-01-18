#include "NeuralNetwork/CUDAFunctions.cuh"

#include "NeuralNetwork/AdamOptimizer.h"

#pragma region CUDA
__global__ void CUDAConvLayer(const float* input, float* output, const float* kernel, const float* biases,
                              int inputDimX, int inputDimY, int outputDimX, int outputDimY, int kernelDimX,
                              int kernelDimY, int kernelDimZ, int strideDimX, int strideDimY, int paddingDimX,
                              int paddingDimY, int kernelNumber) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < outputDimX * outputDimY) {
        unsigned int x = idx % outputDimX;
        unsigned int y = idx / outputDimX;

        unsigned int outputIdx = idx + kernelNumber * outputDimX * outputDimY;

        for (int kz = 0; kz < kernelDimZ; ++kz) {
            for (int ky = 0; ky < kernelDimY; ++ky) {
                for (int kx = 0; kx < kernelDimX; ++kx) {
                    int inputX = kx + x * strideDimX - paddingDimX;
                    int inputY = ky + y * strideDimY - paddingDimY;

                    if (inputX >= 0 && inputX < inputDimX && inputY >= 0 && inputY < inputDimY) {
                        int inputIdx = inputX + inputY * inputDimX + kz * inputDimX * inputDimY;
                        int kernelIdx = kx + ky * kernelDimX + kz * kernelDimX * kernelDimY;

                        output[outputIdx] += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }

        if (biases != nullptr) {
            output[outputIdx] += biases[kernelNumber];
        }
    }
}

__global__ void CUDAReLULayer(float* input, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (input[idx] < 0) {
            input[idx] = 0;
        }
    }
}

__global__ void CUDAMaxPoolingLayer(const float* input, float* output, int outputDimX, int outputDimY, int outputDimZ,
                                    int poolDimX, int poolDimY, int strideDimX, int strideDimY) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < outputDimX * outputDimY * outputDimZ) {
        unsigned int x = idx % outputDimX;
        unsigned int y = (idx / outputDimX) % outputDimY;
        unsigned int z = idx / (outputDimX * outputDimY);

        unsigned int inputWidth = strideDimX * outputDimX;
        unsigned int inputHeight = strideDimY * outputDimY;

        float max = input[x * strideDimX + y * strideDimY * inputWidth + z * inputWidth * inputHeight];

        for (int ky = 0; ky < poolDimY; ++ky) {
            for (int kx = 0; kx < poolDimX; ++kx) {
                unsigned int inputX = kx + x * strideDimX;
                unsigned int inputY = ky + y * strideDimY;
                unsigned int inputZ = z;

                int index = inputX + inputY * inputWidth + inputZ * inputWidth * inputHeight;
                if (input[index] > max) {
                    max = input[index];
                }
            }
        }

        output[idx] = max;
    }
}

__global__ void CUDAFullyConnectedLayer(const float* input, const float* weights, const float* biases,
                                        float* output, int inputSize, int outputSize) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < outputSize) {
        float neuronValue = 0.0f;

        for (int i = 0; i < inputSize; i++) {
            neuronValue += input[i] * weights[i + idx * inputSize];
        }

        if (biases != nullptr) {
            neuronValue += biases[idx];
        }

        output[idx] = neuronValue;
    }
}

__global__ void CUDAConvLayerBackward(float* prevGradients, float* weightGradients, float* biasGradients,
                                      const float* currentGradients, const float* prevLayer,
                                      const float* weights, int prevWidth, int prevHeight, int prevDepth,
                                      int currentWidth, int currentHeight, int currentDepth,
                                      int kernelWidth, int kernelHeight) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < prevWidth * prevHeight * prevDepth) {
        unsigned int x = idx % prevWidth;
        unsigned int y = (idx / prevWidth) % prevHeight;
        unsigned int z = idx / (prevWidth * prevHeight);

        for (int kh = 0; kh < kernelHeight; ++kh) {
            for (int kw = 0; kw < kernelWidth; ++kw) {
                for (int d = 0; d < currentDepth; ++d) {
                    int currentIdx = x + y * currentWidth + d * currentWidth * currentHeight;
                    int weightIdx = kw + kh * kernelWidth + d * kernelWidth * kernelHeight +
                            z * kernelWidth * kernelHeight * currentDepth;

                    // Input Gradients
                    atomicAdd(&prevGradients[idx], currentGradients[currentIdx] * weights[weightIdx]);
                    // Weight Gradients
                    atomicAdd(&weightGradients[weightIdx], currentGradients[currentIdx] * prevLayer[idx]);
                }
            }
        }

        // Bias Gradients
        atomicAdd(&biasGradients[z], currentGradients[idx]);
    }
}

__global__ void CUDAMaxPoolingLayerBackward(float* prevGradients, const float* currentGradients, const float* prevLayer,
                                            int currWidth, int currHeight, int currDepth, int prevWidth, int prevHeight,
                                            int poolDimX, int poolDimY, int strideDimX, int strideDimY) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < currWidth * currHeight * currDepth) {
        unsigned int x = idx % currWidth;
        unsigned int y = (idx / currWidth) % currHeight;
        unsigned int z = idx / (currWidth * currHeight);

        int maxPos = -1;
        float maxValue = -FLT_MAX;

        for (int poolY = 0; poolY < poolDimY; ++poolY) {
            for (int poolX = 0; poolX < poolDimX; ++poolX) {
                int inputX = poolX + x * strideDimX;
                int inputY = poolY + y * strideDimY;

                int inputIdx = inputX + inputY * prevWidth + z * prevWidth * prevHeight;

                if (prevLayer[inputIdx] > maxValue) {
                    maxValue = prevLayer[inputIdx];
                    maxPos = inputIdx;
                }
            }
        }

        atomicAdd(&prevGradients[maxPos], currentGradients[idx]);
    }
}

__global__ void CUDAFullyConnectedLayerBackward(float* prevGradients, float* weightGradients, const float* prevLayer,
                                                const float* weights, const float* currentGradients, int prevSize, int currentSize) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < prevSize * currentSize) {
        unsigned int x = idx % prevSize;
        unsigned int y = idx / prevSize;

        weightGradients[idx] += currentGradients[y] * prevLayer[x];
        atomicAdd(&prevGradients[x], currentGradients[y] * weights[idx]);
    }
}

__global__ void CUDAClipGradient(float* gradient, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (gradient[idx] > CLIP_VALUE) {
            gradient[idx] = CLIP_VALUE;
        }
        else if (gradient[idx] < -CLIP_VALUE){
            gradient[idx] = -CLIP_VALUE;
        }
    }
}
#pragma endregion


Layer* ConvolutionLayer(const Layer* currentLayer, const Group* filters, const ivec2 &stride,
                        const ivec2 &padding, const float* biases) {

    int width = (currentLayer->width - filters->filters[0].width + 2 * padding.x) / stride.x + 1;
    int height = (currentLayer->height - filters->filters[0].height + 2 * padding.y) / stride.y + 1;

    int currentLayerSize = currentLayer->width * currentLayer->height * currentLayer->depth;
    int nextLayerSize = width * height * filters->count;

    Layer* nextLayer = new Layer();
    nextLayer->depth = filters->count;
    nextLayer->width = width;
    nextLayer->height = height;
    nextLayer->maps = new float[nextLayerSize];

    int numBytesCurrentLayerSize = (int)(currentLayerSize * sizeof(float));
    int numBytesNextLayerSize = (int)(nextLayerSize * sizeof(float));

    float* deviceCurrentLayer;
    cudaMalloc((void**)&deviceCurrentLayer, numBytesCurrentLayerSize);
    cudaMemcpy(deviceCurrentLayer, currentLayer->maps, numBytesCurrentLayerSize, cudaMemcpyHostToDevice);

    float* deviceNextLayer;
    cudaMalloc((void**)&deviceNextLayer, numBytesNextLayerSize);
    cudaMemset(deviceNextLayer, 0, numBytesNextLayerSize);

    float* deviceBiases = nullptr;
    if (biases != nullptr) {
        cudaMalloc((void**)&deviceBiases, filters->count * sizeof(float));
        cudaMemcpy(deviceBiases, biases, filters->count * sizeof(float), cudaMemcpyHostToDevice);
    }

    int numBytesKernelSize = (int)(filters->filters[0].width * filters->filters[0].height * filters->filters[0].depth *
                                   sizeof(float));

    float* deviceKernels;
    cudaMalloc((void**)&deviceKernels, numBytesKernelSize);

    int blockSize = 512;
    int gridSize = (currentLayer->width * currentLayer->height + blockSize - 1) / blockSize;

    for (int i = 0; i < filters->count; ++i) {
        cudaMemcpy(deviceKernels, filters->filters[i].maps, numBytesKernelSize, cudaMemcpyHostToDevice);

        CUDAConvLayer<<<gridSize, blockSize>>>(deviceCurrentLayer, deviceNextLayer, deviceKernels, deviceBiases,
                                               currentLayer->width, currentLayer->height, nextLayer->width,
                                               nextLayer->height, filters->filters[i].width, filters->filters[i].height,
                                               filters->filters[i].depth, stride.x, stride.y, padding.x, padding.y, i);
    }
    cudaMemcpy(nextLayer->maps, deviceNextLayer, numBytesNextLayerSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceKernels);
    cudaFree(deviceCurrentLayer);
    cudaFree(deviceNextLayer);

    if (biases != nullptr) {
        cudaFree(deviceBiases);
    }

    return nextLayer;
}

Gradient* ConvolutionLayerBackward(Layer *currentLayer, Group *weights, Layer *previousLayer, std::vector<float>& gradient) {
    int currentGradientSize = (int)gradient.size();
    int previousGradientSize = previousLayer->width * previousLayer->height * previousLayer->depth;
    int weightMapSize = weights->filters[0].width * weights->filters[0].height * weights->filters[0].depth;
    int weightSize = weights->count * weightMapSize;

    Gradient* previousGradient = new Gradient();
    previousGradient->inputGradients.resize(previousGradientSize, 0.0f);
    previousGradient->weightGradients.resize(weightSize, 0.0f);
    previousGradient->biasGradients.resize(weights->count);

    std::vector<float> squashedWeights(weightSize, 0.0f);

    for (int i = 0; i < weights->count; ++i) {
        std::memcpy(&squashedWeights[0] + i * weightMapSize, weights->filters[i].maps, weightMapSize * sizeof(float));
    }

    int numBytesCurrentGradientSize = (int)(currentGradientSize * sizeof(float));
    int numBytesPreviousGradientSize = (int)(previousGradientSize * sizeof(float));
    int numBytesWeightsSize = (int)(weightSize * sizeof(float));

    float* deviceCurrentGradients;
    cudaMalloc((void**)&deviceCurrentGradients, numBytesCurrentGradientSize);
    cudaMemcpy(deviceCurrentGradients, gradient.data(), numBytesCurrentGradientSize, cudaMemcpyHostToDevice);

    float* devicePreviousGradient;
    cudaMalloc((void**)&devicePreviousGradient, numBytesPreviousGradientSize);
    cudaMemset(devicePreviousGradient, 0, numBytesPreviousGradientSize);

    float* deviceWeightGradients;
    cudaMalloc((void**)&deviceWeightGradients, numBytesWeightsSize);
    cudaMemset(deviceWeightGradients, 0, numBytesWeightsSize);

    float* deviceBiasGradients;
    cudaMalloc((void**)&deviceBiasGradients, weights->count * sizeof(float));
    cudaMemset(deviceBiasGradients, 0, weights->count * sizeof(float));

    float* devicePreviousLayer;
    cudaMalloc((void**)&devicePreviousLayer, numBytesPreviousGradientSize);
    cudaMemcpy(devicePreviousLayer, previousLayer->maps, numBytesPreviousGradientSize, cudaMemcpyHostToDevice);

    float* deviceWeights;
    cudaMalloc((void**)&deviceWeights, numBytesWeightsSize);
    cudaMemcpy(deviceWeights, squashedWeights.data(), numBytesWeightsSize, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (previousGradientSize + blockSize - 1) / blockSize;

    CUDAConvLayerBackward<<<gridSize, blockSize>>>(devicePreviousGradient, deviceWeightGradients, deviceBiasGradients,
                                                   deviceCurrentGradients, devicePreviousLayer, deviceWeights,
                                                   previousLayer->width, previousLayer->height, previousLayer->depth,
                                                   currentLayer->width, currentLayer->height, currentLayer->depth,
                                                   weights->filters[0].width, weights->filters[0].height);


    cudaMemcpy(previousGradient->inputGradients.data(), devicePreviousGradient, numBytesPreviousGradientSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(previousGradient->weightGradients.data(), deviceWeightGradients, numBytesWeightsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(previousGradient->biasGradients.data(), deviceBiasGradients, weights->count * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceWeightGradients);
    cudaFree(deviceCurrentGradients);
    cudaFree(deviceBiasGradients);
    cudaFree(devicePreviousGradient);
    cudaFree(devicePreviousLayer);
    cudaFree(deviceWeights);

    ClipGradient(previousGradient->inputGradients);
    ClipGradient(previousGradient->weightGradients);
    ClipGradient(previousGradient->biasGradients);

    return previousGradient;
}

void ReLULayer(Layer* currentLayer) {
    int currentLayerSize = currentLayer->width * currentLayer->height * currentLayer->depth;
    int numBytesCurrentLayerSize = (int)(currentLayerSize * sizeof(float));

    float* deviceCurrentLayer;
    cudaMalloc((void**)&deviceCurrentLayer, numBytesCurrentLayerSize);
    cudaMemcpy(deviceCurrentLayer, currentLayer->maps, numBytesCurrentLayerSize, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (currentLayerSize + blockSize - 1) / blockSize;

    CUDAReLULayer<<<gridSize, blockSize>>>(deviceCurrentLayer, currentLayerSize);

    cudaMemcpy(currentLayer->maps, deviceCurrentLayer, numBytesCurrentLayerSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceCurrentLayer);
}

Layer* MaxPoolingLayer(const Layer* currentLayer, const ivec2& poolDim, const ivec2& stride) {
    int width = (currentLayer->width - poolDim.x) / stride.x + 1;
    int height = (currentLayer->height - poolDim.y) / stride.y + 1;

    int currentLayerSize = currentLayer->width * currentLayer->height * currentLayer->depth;
    int nextLayerSize = width * height * currentLayer->depth;

    Layer* nextLayer = new Layer();
    nextLayer->depth = currentLayer->depth;
    nextLayer->width = width;
    nextLayer->height = height;
    nextLayer->maps = new float[nextLayerSize];

    int numBytesCurrentLayerSize = (int)(currentLayerSize * sizeof(float));
    int numBytesNextLayerSize = (int)(nextLayerSize * sizeof(float));

    float* deviceCurrentLayer;
    cudaMalloc((void**)&deviceCurrentLayer, numBytesCurrentLayerSize);
    cudaMemcpy(deviceCurrentLayer, currentLayer->maps, numBytesCurrentLayerSize, cudaMemcpyHostToDevice);

    float* deviceNextLayer;
    cudaMalloc((void**)&deviceNextLayer, numBytesNextLayerSize);

    int blockSize = 256;
    int gridSize = (nextLayerSize + blockSize - 1) / blockSize;

    CUDAMaxPoolingLayer<<<gridSize, blockSize>>>(deviceCurrentLayer, deviceNextLayer, width, height, nextLayer->depth,
                                                 poolDim.x, poolDim.y, stride.x, stride.y);

    cudaMemcpy(nextLayer->maps, deviceNextLayer, numBytesNextLayerSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceCurrentLayer);
    cudaFree(deviceNextLayer);

    return nextLayer;
}

void MaxPoolingBackward(const Layer* currentLayer, const Layer* previousLayer, std::vector<float>& gradient,
                                      ivec2 poolDim, ivec2 strideDim) {
    int prevLayerSize = previousLayer->width * previousLayer->height * previousLayer->depth;
    int numBytesPrevLayerSize = prevLayerSize * sizeof(float);

    float* devicePrevGradients;
    cudaMalloc((void**)&devicePrevGradients, numBytesPrevLayerSize);
    cudaMemset(devicePrevGradients, 0, numBytesPrevLayerSize);

    float* deviceGradients;
    cudaMalloc((void**)&deviceGradients, gradient.size() * sizeof(float));
    cudaMemcpy(deviceGradients, gradient.data(), gradient.size() * sizeof(float), cudaMemcpyHostToDevice);

    float* devicePrevLayer;
    cudaMalloc((void**)&devicePrevLayer, numBytesPrevLayerSize);
    cudaMemcpy(devicePrevLayer, previousLayer->maps, numBytesPrevLayerSize, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (currentLayer->width * currentLayer->height * currentLayer->depth + blockSize - 1) / blockSize;

    CUDAMaxPoolingLayerBackward<<<gridSize, blockSize>>>(devicePrevGradients, deviceGradients, devicePrevLayer,
                                                         currentLayer->width, currentLayer->height, currentLayer->depth,
                                                         previousLayer->width, previousLayer->height, poolDim.x,
                                                         poolDim.y, strideDim.x, strideDim.y);

    gradient.clear();
    gradient.resize(prevLayerSize, 0.0f);
    cudaMemcpy(gradient.data(), devicePrevGradients, numBytesPrevLayerSize, cudaMemcpyDeviceToHost);

    cudaFree(devicePrevGradients);
    cudaFree(deviceGradients);
    cudaFree(devicePrevLayer);
}

Layer* FullyConnectedLayer(const Layer* currentLayer, const float* weights, int currentLayerSize, int nextLayerSize,
                           const float* biases) {
    Layer* nextLayer = new Layer();
    nextLayer->depth = 1;
    nextLayer->height = 1;
    nextLayer->width = nextLayerSize;
    nextLayer->maps = new float[nextLayerSize];

    int numBytesCurrentLayerSize = (int)(currentLayerSize * sizeof(float));
    int numBytesNexLayerSize = (int)(nextLayerSize * sizeof(float));

    float* deviceCurrentLayerNeurons;
    cudaMalloc((void**)&deviceCurrentLayerNeurons, numBytesCurrentLayerSize);

    float* deviceWeights;
    cudaMalloc((void**)&deviceWeights, numBytesCurrentLayerSize * nextLayerSize);

    float* deviceNextLayerNeurons;
    cudaMalloc((void**)&deviceNextLayerNeurons, numBytesNexLayerSize);
    cudaMemset((void**)&deviceNextLayerNeurons, 0, numBytesNexLayerSize);

    float* deviceBiases = nullptr;
    if (biases != nullptr) {
        cudaMalloc((void**)&deviceBiases, numBytesNexLayerSize);
        cudaMemcpy(deviceBiases, biases, numBytesNexLayerSize, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(deviceCurrentLayerNeurons, currentLayer->maps, numBytesCurrentLayerSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceWeights, weights, numBytesCurrentLayerSize * nextLayerSize, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (currentLayerSize * nextLayerSize + blockSize - 1) / blockSize;

    CUDAFullyConnectedLayer<<<gridSize, blockSize>>>(deviceCurrentLayerNeurons, deviceWeights, deviceBiases,
                                                     deviceNextLayerNeurons, currentLayerSize, nextLayerSize);

    cudaMemcpy(nextLayer->maps, deviceNextLayerNeurons, numBytesNexLayerSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceCurrentLayerNeurons);
    cudaFree(deviceWeights);
    cudaFree(deviceNextLayerNeurons);

    if (deviceBiases != nullptr) {
        cudaFree(deviceBiases);
    }

    return nextLayer;
}

Gradient* FullyConnectedLayerBackward(Layer* currentLayer, Group* weights, Layer* previousLayer, std::vector<float>& gradient) {
    Gradient* previousGradient = new Gradient();

    int currentLayerSize = currentLayer->width * currentLayer->height * currentLayer->depth;
    int previousLayerSize = previousLayer->width * previousLayer->height * previousLayer->depth;
    previousGradient->inputGradients.resize(previousLayerSize, 0.0f);
    previousGradient->weightGradients.resize(previousLayerSize * currentLayerSize, 0.0f);
    previousGradient->biasGradients.resize(currentLayerSize, 0.0f);

    int numBytesWeightGradients = (int)(previousLayerSize * currentLayerSize * sizeof(float));
    int numBytesPreviousLayer = (int)(previousLayerSize * sizeof(float));
    int numBytesGradient = (int)(currentLayerSize * sizeof(float));

    float* devicePrevGradients;
    cudaMalloc((void**)&devicePrevGradients, numBytesPreviousLayer);
    cudaMemset(devicePrevGradients, 0, numBytesPreviousLayer);

    float* deviceWeightGradients;
    cudaMalloc((void**)&deviceWeightGradients, numBytesWeightGradients);
    cudaMemset(deviceWeightGradients, 0, numBytesWeightGradients);

    float* devicePreviousLayer;
    cudaMalloc((void**)&devicePreviousLayer, numBytesPreviousLayer);
    cudaMemcpy(devicePreviousLayer, previousLayer->maps, numBytesPreviousLayer, cudaMemcpyHostToDevice);

    float* deviceWeights;
    cudaMalloc((void**)&deviceWeights, numBytesWeightGradients);
    cudaMemcpy(deviceWeights, weights->filters[0].maps, numBytesWeightGradients, cudaMemcpyHostToDevice);

    float* deviceGradients;
    cudaMalloc((void**)&deviceGradients, numBytesGradient);
    cudaMemcpy(deviceGradients, gradient.data(), numBytesGradient, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (previousLayerSize * currentLayerSize + blockSize - 1) / blockSize;

    CUDAFullyConnectedLayerBackward<<<gridSize, blockSize>>>(devicePrevGradients, deviceWeightGradients,
                                                             devicePreviousLayer,deviceWeights, deviceGradients,
                                                             previousLayerSize, currentLayerSize);

    cudaMemcpy(previousGradient->inputGradients.data(), devicePrevGradients, numBytesPreviousLayer, cudaMemcpyDeviceToHost);
    cudaMemcpy(previousGradient->weightGradients.data(), deviceWeightGradients, numBytesWeightGradients, cudaMemcpyDeviceToHost);

    cudaFree(devicePrevGradients);
    cudaFree(deviceWeightGradients);
    cudaFree(devicePreviousLayer);
    cudaFree(deviceWeights);
    cudaFree(deviceGradients);

    std::memcpy(previousGradient->biasGradients.data(), gradient.data(), gradient.size() * sizeof(float));

    ClipGradient(previousGradient->inputGradients);
    ClipGradient(previousGradient->weightGradients);
    ClipGradient(previousGradient->biasGradients);

    return previousGradient;
}

void DropoutLayer(Layer *currentLayer, float dropoutRate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(0, 1);

    for (int i = 0; i < currentLayer->width * currentLayer->height * currentLayer->depth; ++i) {
        float randomValue = distribution(gen);
        if (randomValue < dropoutRate) {
            currentLayer->maps[i] = 0;
        }
        else {
            currentLayer->maps[i] *= 1 / (1 - dropoutRate);
        }
    }
}

float MSELossFunction(const float* input, const float* predictedResult, int size) {
    float loss = 0;

    for (int i = 0; i < size; ++i) {
        float diff = input[i] - predictedResult[i];
        loss += diff * diff;
    }

    loss /= (float)size;

    return loss;
}

void ClipGradient(std::vector<float>& gradient) {
    int dataSize = (int)gradient.size();
    int numBytesDataSize = (int)(dataSize * sizeof(float));

    float* deviceData;
    cudaMalloc((void**)&deviceData, numBytesDataSize);
    cudaMemcpy(deviceData, gradient.data(), numBytesDataSize, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (dataSize + blockSize - 1) / blockSize;

    CUDAClipGradient<<<gridSize, blockSize>>>(deviceData, dataSize);

    cudaMemcpy(gradient.data(), deviceData, numBytesDataSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceData);
}

void UpdateNetwork(const std::vector<std::vector<Gradient*>>& gradients, std::vector<Group*>& weights, std::vector<Layer*>& biases) {
    std::vector<Gradient*> avgGradients;
    avgGradients.reserve(gradients[0].size());

    for (int i = 0; i < gradients[0].size(); ++i) {
        avgGradients.push_back(new Gradient());
        avgGradients[i]->weightGradients.resize(gradients[0][i]->weightGradients.size(), 0.0f);
        avgGradients[i]->biasGradients.resize(gradients[0][i]->biasGradients.size(), 0.0f);
    }

    for (int gradient = 0; gradient < gradients.size(); ++gradient) {
        for (int i = 0; i < gradients[gradient].size(); ++i) {
            for (int j = 0; j < gradients[gradient][i]->weightGradients.size(); ++j) {
                avgGradients[i]->weightGradients[j] += (gradients[gradient][i]->weightGradients[j] / (float)gradients.size());
            }

            for (int j = 0; j < gradients[gradient][i]->biasGradients.size(); ++j) {
                avgGradients[i]->biasGradients[j] += (gradients[gradient][i]->biasGradients[j] / (float)gradients.size());
            }
        }
    }

    UpdateWeightsAndBiases(avgGradients, weights, biases);

    for(int i = 0; i < avgGradients.size(); ++i) {
        delete avgGradients[i];
    }
}

void UpdateWeightsAndBiases(const std::vector<Gradient*>& gradients, std::vector<Group*>& weights, std::vector<Layer*>& biases) {
    AdamOptimizer* adamOptimizer = AdamOptimizer::GetInstance();

    for (int layer = 0; layer < gradients.size(); ++layer) {
        int idx = 15 - layer;
        adamOptimizer->UpdateParameters(biases[idx]->maps,  biases[idx]->width * biases[idx]->height * biases[idx]->depth,
                                        gradients[layer]->biasGradients);

        for (int i = 0; i < weights[idx]->count; ++i) {
            int weightsSize = weights[idx]->filters[i].width * weights[idx]->filters[i].height * weights[idx]->filters[i].depth;
            std::vector<float> weightGradients(weightsSize);

            std::memcpy(&weightGradients[0], &gradients[layer]->weightGradients[0] + i * weightsSize, weightsSize * sizeof(float));

            adamOptimizer->UpdateParameters(weights[idx]->filters[i].maps, weightsSize, weightGradients);
        }
    }
}
