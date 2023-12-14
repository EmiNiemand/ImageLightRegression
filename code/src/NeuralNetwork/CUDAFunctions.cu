#include "NeuralNetwork/CUDAFunctions.cuh"

#include "NeuralNetwork/AdamOptimizer.h"

#pragma region CUDA
__global__ void CUDAConvLayer(const float* input, float* output, const float* kernel, const float* biases,
                              int inputDimX, int inputDimY, int outputDimX, int outputDimY, int kernelDimX,
                              int kernelDimY, int kernelDimZ, int strideDimX, int strideDimY, int paddingDimX,
                              int paddingDimY, int kernelNumber) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < inputDimX * inputDimY * kernelDimX * kernelDimY) {
        unsigned int inputIdx = idx % (inputDimX * inputDimY);
        unsigned int x = inputIdx % inputDimX;
        unsigned int y = inputIdx / inputDimX;
        unsigned int kx = (idx / (inputDimX * inputDimY)) % kernelDimX;
        unsigned int ky = idx / ((inputDimX * inputDimY) * kernelDimX);

        unsigned int outputIdx = inputIdx + kernelNumber * outputDimX * outputDimY;

        for (int kz = 0; kz < kernelDimZ; ++kz) {
            int index = x * strideDimX - paddingDimX + kx + (y * strideDimY - paddingDimY + ky) * inputDimX;

            if (index < 0 || index >= inputDimX * inputDimY) {
                output[outputIdx] += 0;
            }
            else {
                output[outputIdx] += input[inputIdx+ kz * inputDimX * inputDimY] *
                        kernel[kx + ky * kernelDimX + kz * kernelDimX * kernelDimY];
                if (biases != nullptr) {
                    output[outputIdx] += biases[kernelNumber];
                }
            }
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

__global__ void CUDAPoolingLayer(const float* input, float* output, int outputDimX, int outputDimY, int outputDimZ,
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
                unsigned int inputX = x * strideDimX + kx;
                unsigned int inputY = y * strideDimY + ky;
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
        output[idx] = neuronValue;

        if (biases != nullptr) {
            output[idx] += biases[idx];
        }
    }
}

__global__ void CUDAConvLayerGradients(float* prevGradients, float* weightGradients, const float* currentGradients,
                                       const float* prevLayer, const float* weights, int prevWidth, int prevHeight,
                                       int prevDepth, int currentWidth, int currentHeight, int currentDepth,
                                       int kernelWidth, int kernelHeight) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < prevWidth * prevHeight * prevDepth) {
        unsigned int x = idx % prevWidth;
        unsigned int y = (idx / prevWidth) % prevHeight;
        unsigned int z = idx / (prevWidth * prevHeight);

        for (int d = 0; d < currentDepth; ++d) {
            int currentIdx = d * currentWidth * currentHeight + y * currentWidth + x;

            for (int kh = 0; kh < kernelHeight; ++kh) {
                for (int kw = 0; kw < kernelWidth; ++kw) {
                    int weightIdx = z * kernelWidth * kernelHeight * currentDepth + d * kernelWidth * kernelHeight +
                                    kh * kernelWidth + kw;

                    // Input Gradients
                    atomicAdd(&prevGradients[idx], currentGradients[currentIdx] * weights[weightIdx]);
                    // Weight Gradients
                    atomicAdd(&weightGradients[weightIdx], currentGradients[currentIdx] * prevLayer[idx]);
                }
            }
        }
    }
}

__global__ void CUDAConvLayerBiasGradients(float* biasGradients, const float* currentGradients, int currentWidth,
                                           int currentHeight, int currentDepth) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < currentWidth * currentHeight * currentDepth) {
        unsigned int z = idx / (currentWidth * currentHeight);
        // Bias Gradients
        biasGradients[z] += currentGradients[idx];
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


Layer* ConvolutionLayer(const Layer* input, const Group* filters, const ivec2 &stride,
                        const ivec2 &padding, const float* biases) {

    int width = (input->width - filters->filters[0].width + 2 * padding.x) / stride.x + 1;
    int height = (input->height - filters->filters[0].height + 2 * padding.y) / stride.y + 1;

    int inputSize = input->width * input->height * input->depth;
    int outputSize = width * height * filters->count;

    Layer* output = new Layer();
    output->depth = filters->count;
    output->width = width;
    output->height = height;
    output->maps = new float[outputSize];

    int numByteInputSize = inputSize * sizeof(float);
    int numByteOutputSize = outputSize * sizeof(float);

    float* deviceInput;
    cudaMalloc((void**)&deviceInput, numByteInputSize);
    cudaMemcpy(deviceInput, input->maps, numByteInputSize, cudaMemcpyHostToDevice);

    float* deviceOutput;
    cudaMalloc((void**)&deviceOutput, numByteOutputSize);
    cudaMemset(deviceOutput, 0, numByteOutputSize);

    float* deviceBiases = nullptr;
    if (biases != nullptr) {
        cudaMalloc((void**)&deviceBiases, filters->count * sizeof(float));
        cudaMemcpy(deviceBiases, biases, filters->count * sizeof(float), cudaMemcpyHostToDevice);
    }

    int numByteKernelSize = filters->filters[0].width * filters->filters[0].height * filters->filters[0].depth * sizeof(float);

    int blockSize = 512;
    int gridSize = (input->width * input->height * filters->filters[0].width * filters->filters[0].height + blockSize - 1) / blockSize;

    float* deviceKernels;
    cudaMalloc((void**)&deviceKernels, numByteKernelSize);

    for (int i = 0; i < filters->count; ++i) {
        cudaMemcpy(deviceKernels, filters->filters[i].maps, numByteKernelSize, cudaMemcpyHostToDevice);

        CUDAConvLayer<<<gridSize, blockSize>>>(deviceInput, deviceOutput, deviceKernels, deviceBiases, input->width,
                                               input->height, output->width, output->height, filters->filters[i].width,
                                               filters->filters[i].height, filters->filters[i].depth,
                                               stride.x, stride.y, padding.x, padding.y, i);
    }
    cudaFree(deviceKernels);

    cudaMemcpy(output->maps, deviceOutput, numByteOutputSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    if (biases != nullptr) {
        cudaFree(deviceBiases);
    }

    return output;
}

Gradient* ConvolutionLayerBackward(Layer *currentLayer, Group *weights, Layer *previousLayer, std::vector<float>& gradient) {
    int currGradientSize = (int)gradient.size();
    int outGradientSize = previousLayer->width * previousLayer->height * previousLayer->depth;
    int weightMapSize = weights->filters[0].width * weights->filters[0].height * weights->filters[0].depth;
    int weightSize = weights->count * weightMapSize;

    Gradient* output = new Gradient();
    output->inputsGradients.resize(outGradientSize, 0.0f);
    output->weightsGradients.resize(weightSize, 0.0f);
    output->biasesGradients.resize(weights->count);

    std::vector<float> squashedWeights(weightSize, 0.0f);

    for (int i = 0; i < weights->count; ++i) {
        std::memcpy(&squashedWeights[0] + i * weightMapSize, weights->filters[i].maps, weightMapSize * sizeof(float));
    }

    int numBytesGradientSize = (int)(currGradientSize * sizeof(float));
    int numBytesOutGradientSize = (int)(outGradientSize * sizeof(float));
    int numBytesWeightsSize = (int)(weightSize * sizeof(float));

    float* deviceGradient;
    cudaMalloc((void**)&deviceGradient, numBytesGradientSize);
    cudaMemcpy(deviceGradient, gradient.data(), numBytesGradientSize, cudaMemcpyHostToDevice);

    float* deviceOutputGradient;
    cudaMalloc((void**)&deviceOutputGradient, numBytesOutGradientSize);
    cudaMemset(deviceOutputGradient, 0, numBytesOutGradientSize);

    float* deviceWeightGradient;
    cudaMalloc((void**)&deviceWeightGradient, numBytesWeightsSize);
    cudaMemset(deviceWeightGradient, 0, numBytesWeightsSize);

    float* deviceBiasesGradient;
    cudaMalloc((void**)&deviceBiasesGradient, weights->count * sizeof(float));
    cudaMemset(deviceBiasesGradient, 0, weights->count * sizeof(float));

    float* devicePreviousLayer;
    cudaMalloc((void**)&devicePreviousLayer, numBytesOutGradientSize);
    cudaMemcpy(devicePreviousLayer, previousLayer->maps, numBytesOutGradientSize, cudaMemcpyHostToDevice);

    float* deviceWeights;
    cudaMalloc((void**)&deviceWeights, numBytesWeightsSize);
    cudaMemcpy(deviceWeights, squashedWeights.data(), numBytesWeightsSize, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (outGradientSize + blockSize - 1) / blockSize;

    CUDAConvLayerGradients<<<gridSize, blockSize>>>(deviceOutputGradient, deviceWeightGradient, deviceGradient,
                                                    devicePreviousLayer, deviceWeights, previousLayer->width,
                                                    previousLayer->height, previousLayer->depth, currentLayer->width,
                                                    currentLayer->height, currentLayer->depth,
                                                    weights->filters[0].width,
                                                    weights->filters[0].height);

    gridSize = (currGradientSize + blockSize - 1) / blockSize;

    CUDAConvLayerBiasGradients<<<gridSize, blockSize>>>(deviceBiasesGradient, deviceGradient, currentLayer->width,
                                                        currentLayer->height, currentLayer->depth);

    cudaMemcpy(output->inputsGradients.data(), deviceOutputGradient, numBytesOutGradientSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(output->weightsGradients.data(), deviceWeightGradient, numBytesWeightsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(output->biasesGradients.data(), deviceBiasesGradient, weights->count * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceWeightGradient);
    cudaFree(deviceGradient);
    cudaFree(deviceBiasesGradient);
    cudaFree(deviceOutputGradient);
    cudaFree(devicePreviousLayer);
    cudaFree(deviceWeights);

    ClipGradient(output->inputsGradients);
    ClipGradient(output->weightsGradients);
    ClipGradient(output->biasesGradients);

    return output;
}

void ReLULayer(Layer* input) {
    int dataSize = input->width * input->height * input->depth;
    int numBytesDataSize = dataSize * sizeof(float);

    int blockSize = 256;
    int gridSize = (dataSize + blockSize - 1) / blockSize;

    float* deviceData;
    cudaMalloc((void**)&deviceData, numBytesDataSize);
    cudaMemcpy(deviceData, input->maps, numBytesDataSize, cudaMemcpyHostToDevice);

    CUDAReLULayer<<<gridSize, blockSize>>>(deviceData, dataSize);

    cudaMemcpy(input->maps, deviceData, numBytesDataSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceData);
}

Layer* PoolingLayer(const Layer* input, const ivec2& poolDim, const ivec2& stride) {
    int width = (input->width - poolDim.x) / stride.x + 1;
    int height = (input->height - poolDim.y) / stride.y + 1;

    int inputDataSize = input->width * input->height * input->depth;
    int outputDataSize = width * height * input->depth;

    Layer* output = new Layer();
    output->depth = input->depth;
    output->width = width;
    output->height = height;
    output->maps = new float[outputDataSize];

    int blockSize = 256;
    int gridSize = (outputDataSize + blockSize - 1) / blockSize;

    int numBytesInputData = inputDataSize * sizeof(float);
    int numBytesOutputData = outputDataSize * sizeof(float);

    float* deviceInputData;
    cudaMalloc((void**)&deviceInputData, numBytesInputData);
    cudaMemcpy(deviceInputData, input->maps, numBytesInputData, cudaMemcpyHostToDevice);

    float* deviceOutputData;
    cudaMalloc((void**)&deviceOutputData, numBytesOutputData);

    CUDAPoolingLayer<<<gridSize, blockSize>>>(deviceInputData, deviceOutputData, width, height, output->depth,
                                              poolDim.x, poolDim.y, stride.x, stride.y);

    cudaMemcpy(output->maps, deviceOutputData, numBytesOutputData, cudaMemcpyDeviceToHost);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    return output;
}

void MaxPoolingBackward(const Layer* currentLayer, const Layer* previousLayer, std::vector<float>& gradient,
                                      ivec2 poolDim, ivec2 strideDim) {
    std::vector<float> previousGradient(previousLayer->width * previousLayer->height * previousLayer->depth, 0.0f);

    for (int d = 0; d < previousLayer->depth; ++d) {
        for (int h = 0; h < previousLayer->height; h+=strideDim.y) {
            for (int w = 0; w < previousLayer->width; w+=strideDim.x) {
                int currIdx = w / strideDim.x + h / strideDim.y * currentLayer->width +
                        d * currentLayer->width * currentLayer->height;
                bool foundMatch = false;

                for (int poolY = 0; poolY < poolDim.y; ++poolY) {
                    for (int poolX = 0; poolX < poolDim.x; ++poolX) {
                        int prevIdx = w + poolX + (h + poolY) * previousLayer->width +
                                d * previousLayer->width * previousLayer->height;

                        if (previousLayer->maps[prevIdx] == currentLayer->maps[currIdx]) {
                            previousGradient[prevIdx] = gradient[currIdx];
                            foundMatch = true;
                            break;
                        }
                    }

                    if (foundMatch) {
                        break;  // Break out of the poolY loop
                    }
                }
            }
        }
    }

    gradient.clear();
    gradient = previousGradient;
}

Layer* FullyConnectedLayer(const Layer* input, const float* weights, int inputSize, int outputSize,
                           const float* biases) {
    Layer* output = new Layer();
    output->depth = 1;
    output->height = 1;
    output->width = outputSize;
    output->maps = new float[outputSize];

    int numBytesInput = inputSize * sizeof(float);
    int numBytesOutput = outputSize * sizeof(float);

    // Allocate memory on the device (GPU)
    float* deviceInputNeurons;
    float* deviceWeights;
    float* deviceBiases = nullptr;
    float* deviceOutputNeurons;
    cudaMalloc((void**)&deviceInputNeurons, numBytesInput);
    cudaMalloc((void**)&deviceWeights, numBytesInput * outputSize);
    cudaMalloc((void**)&deviceOutputNeurons, numBytesOutput);
    cudaMemset((void**)&deviceOutputNeurons, 0, numBytesOutput);

    if (biases != nullptr) {
        cudaMalloc((void**)&deviceBiases, numBytesOutput);
        cudaMemcpy(deviceBiases, biases, numBytesOutput, cudaMemcpyHostToDevice);
    }

    // Copy input neurons and weights from host to device
    cudaMemcpy(deviceInputNeurons, input->maps, numBytesInput, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceWeights, weights, numBytesInput * outputSize, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (inputSize * outputSize + blockSize - 1) / blockSize;

    // Launch the kernel to calculate the neuron values on the GPU
    CUDAFullyConnectedLayer<<<gridSize, blockSize>>>(deviceInputNeurons, deviceWeights, deviceBiases, deviceOutputNeurons,
                                                     inputSize, outputSize);

    // Copy the result array from device to host
    cudaMemcpy(output->maps, deviceOutputNeurons, numBytesOutput, cudaMemcpyDeviceToHost);

    // Free memory on the device
    cudaFree(deviceInputNeurons);
    cudaFree(deviceWeights);
    cudaFree(deviceOutputNeurons);

    if (deviceBiases != nullptr) {
        cudaFree(deviceBiases);
    }

    return output;
}

Gradient* FullyConnectedLayerBackward(Layer* currentLayer, Group* weights, Layer* previousLayer, std::vector<float>& gradient) {
    Gradient* output = new Gradient();

    int currLayerSize = currentLayer->width * currentLayer->height * currentLayer->depth;
    int prevLayerSize = previousLayer->width * previousLayer->height * previousLayer->depth;
    output->inputsGradients.resize(prevLayerSize, 0.0f);
    output->weightsGradients.resize(prevLayerSize * currLayerSize, 0.0f);
    output->biasesGradients.resize(currLayerSize, 0.0f);

    for (int i = 0; i < currLayerSize; ++i) {
        for (int j = 0; j < prevLayerSize; ++j) {
            output->inputsGradients[j] += gradient[i] * weights->filters[0].maps[j + i * gradient.size()];
            output->weightsGradients[j + i * currLayerSize] += gradient[i] * previousLayer->maps[j];
        }
    }

    std::memcpy(output->biasesGradients.data(), gradient.data(), gradient.size() * sizeof(float));

    ClipGradient(output->inputsGradients);
    ClipGradient(output->weightsGradients);
    ClipGradient(output->biasesGradients);

    return output;
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

    int blockSize = 256;
    int gridSize = (dataSize + blockSize - 1) / blockSize;

    float* deviceData;
    cudaMalloc((void**)&deviceData, numBytesDataSize);
    cudaMemcpy(deviceData, gradient.data(), numBytesDataSize, cudaMemcpyHostToDevice);

    CUDAClipGradient<<<gridSize, blockSize>>>(deviceData, dataSize);

    cudaMemcpy(gradient.data(), deviceData, numBytesDataSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceData);
}


void MiniBatch(const std::vector<std::vector<Gradient*>>& gradients, std::vector<Group*>& weights,
               std::vector<Layer*>& biases, float learningRate) {
    std::vector<Gradient*> avgGradients;
    avgGradients.reserve(gradients[0].size());

    for (int i = 0; i < gradients[0].size(); ++i) {
        avgGradients.push_back(new Gradient());
        avgGradients[i]->weightsGradients.resize(gradients[0][i]->weightsGradients.size(), 0.0f);
        avgGradients[i]->biasesGradients.resize(gradients[0][i]->biasesGradients.size(), 0.0f);
    }

    for (int gradient = 0; gradient < gradients.size(); ++gradient) {
        for (int i = 0; i < gradients[gradient].size(); ++i) {
            for (int j = 0; j < gradients[gradient][i]->weightsGradients.size(); ++j) {
                avgGradients[i]->weightsGradients[j] += (gradients[gradient][i]->weightsGradients[j] / (float)gradients.size());
            }

            for (int j = 0; j < gradients[gradient][i]->biasesGradients.size(); ++j) {
                avgGradients[i]->biasesGradients[j] += (gradients[gradient][i]->biasesGradients[j] / (float)gradients.size());
            }
        }
    }

    UpdateWeightsAndBiases(avgGradients, weights, biases, learningRate);

    for (int i = 0; i < avgGradients.size(); ++i) {
        delete avgGradients[i];
    }
}

void UpdateWeightsAndBiases(const std::vector<Gradient*>& gradients, std::vector<Group*>& weights,
                            std::vector<Layer*>& biases, float learningRate) {
    AdamOptimizer* adamOptimizer = AdamOptimizer::GetInstance();
    adamOptimizer->learningRate = learningRate;

    for (int layer = 0; layer < gradients.size(); ++layer) {
        int idx = 15 - layer;
        adamOptimizer->IncrementTimeStep();
        adamOptimizer->UpdateParameters(biases[idx]->maps,  biases[idx]->width * biases[idx]->height * biases[idx]->depth,
                                        gradients[layer]->biasesGradients);

        for (int i = 0; i < weights[idx]->count; ++i) {
            int weightsSize = weights[idx]->filters[i].width * weights[idx]->filters[i].height * weights[idx]->filters[i].depth;
            std::vector<float> weightGradients(weightsSize);

            std::memcpy(&weightGradients[0], &gradients[layer]->weightsGradients[0] + i * weightsSize, weightsSize * sizeof(float));

            adamOptimizer->UpdateParameters(weights[idx]->filters[i].maps, weightsSize, weightGradients);
        }
    }
}