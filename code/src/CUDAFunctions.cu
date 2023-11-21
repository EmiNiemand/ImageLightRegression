#include "CUDAFunctions.cuh"
#include <stdio.h>

#pragma region CUDA
__global__ void CUDAConvLayer(const float* input, float* output, const float* kernel, const float* biases, int inputDimX, int inputDimY,
                              int outputDimX, int outputDimY, int kernelDimX,int kernelDimY, int kernelDimZ,
                              int strideDimX, int strideDimY, int paddingDimX, int paddingDimY, int filterCount) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < outputDimX * outputDimY) {
        unsigned int x = idx % inputDimX;
        unsigned int y = idx / inputDimX;

        output[idx + filterCount * outputDimX * outputDimY] = 0;

        for (int kz = 0; kz < kernelDimZ; ++kz) {
            for (int ky = 0; ky < kernelDimY; ++ky) {
                for (int kx = 0; kx < kernelDimX; ++kx) {
                    int index = x * strideDimX - paddingDimX + kx + (y * strideDimY - paddingDimY + ky) * inputDimX;
                    if (index < 0 || index > inputDimX * inputDimY) {
                        output[idx + filterCount * outputDimX * outputDimY] += 0;
                    }
                    else {
                        output[idx + filterCount * outputDimX * outputDimY] += input[index + kz * inputDimX * inputDimY] *
                                                                               kernel[kx + ky * kernelDimX + kz * kernelDimX * kernelDimY];
                        if (biases != nullptr) {
                            output[idx + filterCount * outputDimX * outputDimY] += biases[filterCount];
                        }
                    }
                }
            }
        }
    }
}

__global__ void CUDAReLULayer(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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

        unsigned int x = idx / (outputDimY * outputDimZ);
        unsigned int y = (idx / outputDimZ) % outputDimY;
        unsigned int z = idx % outputDimZ;

        float max = input[x * strideDimX + y * outputDimX * strideDimY + z * outputDimX * outputDimY * strideDimY];

        for (int ky = 0; ky < poolDimY; ++ky) {
            for (int kx = 0; kx < poolDimX; ++kx) {
                int index = x * strideDimX + kx + y * outputDimX * strideDimY + ky + z * outputDimX * outputDimY * strideDimY;

                if (input[index] > max) max = input[index];
            }
        }

        output[x + y * outputDimX + z * outputDimX * outputDimY] = max;
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

__global__ void CUDARecalculateConvWeightsAndGradient(float* weights, const float* gradients, float* outputGradients,
                                                      const float* previousLayer, int weightSize, int outGradientWidth,
                                                      int outGradientHeight, int outGradientDepth, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int widthHeight = outGradientWidth * outGradientHeight;

    if (idx < weightSize * widthHeight) {
        int weightIdx = idx % weightSize;
        int widthHeightIdx = idx / weightSize;

        float weightUpdate = 0.0f;

        // Iterate over the output gradient
        for (int i = 0; i < outGradientDepth; ++i) {
            weightUpdate += gradients[widthHeightIdx + i * outGradientDepth] * previousLayer[widthHeightIdx + i * outGradientDepth];

            // Update the output gradient for the input layer
            outputGradients[widthHeightIdx + i * outGradientDepth] += gradients[widthHeightIdx + i * outGradientDepth] * weights[weightIdx];
        }

        // Update the weight
        weights[weightIdx] -= learningRate * weightUpdate;
    }


}

__global__ void CUDARecalculateConvBiases(float* biases, const float* gradient, int size, int depth, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size * depth) {
        int depthIndex = idx / size;

        for (int i = 0; i < size; ++i) {
            biases[depthIndex] -= learningRate * gradient[idx];
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

    float* deviceBiases = nullptr;
    if (biases != nullptr) {
        cudaMalloc((void**)&deviceBiases, filters->count * sizeof(float));
        cudaMemcpy(deviceBiases, biases, filters->count * sizeof(float), cudaMemcpyHostToDevice);
    }

    int numByteKernelSize = filters->filters[0].width * filters->filters[0].height * filters->filters[0].depth;

    int blockSize = 256;
    int gridSize = (width * height + blockSize - 1) / blockSize;

    float* deviceKernels;
    cudaMalloc((void**)&deviceKernels, numByteKernelSize);

    for (int i = 0; i < filters->count; ++i) {
        cudaMemcpy(deviceKernels, filters->filters[i].maps, numByteKernelSize, cudaMemcpyHostToDevice);

        CUDAConvLayer<<<gridSize, blockSize>>>(deviceInput, deviceOutput, deviceKernels, deviceBiases, input->width,
                                               input->height, output->width, output->height, filters->filters[i].width,
                                               filters->filters[i].height, filters->filters[i].depth, stride.x, stride.y,
                                               padding.x, padding.y, i);
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

void ConvolutionLayerBackward(Layer *currentLayer, Group *weights, Layer *biases, Layer *previousLayer,
                              float *&gradient, float learningRate) {
    int currGradientSize = currentLayer->width * currentLayer->height * currentLayer->depth;
    int outGradientSize = previousLayer->width * previousLayer->height * previousLayer->depth;
    int biasesSize = biases->width * biases->height * biases->depth;

    float* outputGradient = new float[outGradientSize];

    int numBytesGradientSize = (int)(currGradientSize * sizeof(float));
    int numBytesOutGradientSize = (int)(outGradientSize * sizeof(float));
    int numBytesBiasesSize = (int)(biases->width * biases->height * biases->depth * sizeof(float));

    float* deviceBiases;
    cudaMalloc((void**)&deviceBiases, numBytesBiasesSize);
    cudaMemcpy(deviceBiases, biases->maps, numBytesBiasesSize, cudaMemcpyHostToDevice);

    float* deviceGradient;
    cudaMalloc((void**)&deviceGradient, numBytesGradientSize);
    cudaMemcpy(deviceGradient, gradient, numBytesGradientSize, cudaMemcpyHostToDevice);

    float* deviceOutputGradient;
    cudaMalloc((void**)&deviceOutputGradient, numBytesOutGradientSize);
    cudaMemset(deviceOutputGradient, 0, numBytesOutGradientSize);

    float* devicePreviousLayer;
    cudaMalloc((void**)&devicePreviousLayer, numBytesOutGradientSize);
    cudaMemcpy(devicePreviousLayer, previousLayer->maps, numBytesOutGradientSize, cudaMemcpyHostToDevice);

    int weightSize = weights->filters[0].width * weights->filters[0].height * weights->filters[0].depth;
    int numBytesWeightsSize = (int)(weightSize * sizeof(float));

    float* deviceWeights;
    cudaMalloc((void**)&deviceWeights, numBytesWeightsSize);

    int blockSize = 256;
    int gridSize = (biasesSize + blockSize - 1) / blockSize;

    CUDARecalculateConvBiases<<<gridSize, blockSize>>>(deviceBiases, deviceGradient, biases->width * biases->height,
                                                       biases->depth, learningRate);
    cudaMemcpy(biases->maps, deviceBiases, numBytesBiasesSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceBiases);

    gridSize = (weightSize * previousLayer->width * previousLayer->height + blockSize - 1) / blockSize;

    for (int d = 0; d < weights->count; ++d) {
        cudaMemcpy(deviceWeights, weights->filters[d].maps, numBytesWeightsSize, cudaMemcpyHostToDevice);

        CUDARecalculateConvWeightsAndGradient<<<gridSize, blockSize>>>(deviceWeights, deviceGradient, deviceOutputGradient,
                                                                       devicePreviousLayer, weightSize, previousLayer->width,
                                                                       previousLayer->height, previousLayer->depth,
                                                                       learningRate);

        cudaMemcpy(weights->filters[d].maps, deviceWeights, numBytesWeightsSize, cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(outputGradient, deviceOutputGradient, numBytesOutGradientSize, cudaMemcpyDeviceToHost);

    cudaFree(devicePreviousLayer);
    cudaFree(deviceGradient);
    cudaFree(deviceOutputGradient);
    cudaFree(deviceWeights);

    delete[] gradient;
    gradient = outputGradient;
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

void MaxPoolingBackward(const Layer* currentLayer, const Layer* previousLayer, float*& gradient, ivec2 poolDim) {
    float* previousGradient = new float[previousLayer->width * previousLayer->height * previousLayer->depth]();

    for (int d = 0; d < currentLayer->depth; ++d) {
        for (int h = 0; h < currentLayer->height; ++h) {
            for (int w = 0; w < currentLayer->width; ++w) {
                int currIdx = w + h * currentLayer->width + d * currentLayer->width * currentLayer->height;
                bool foundMatch = false;

                for (int poolY = 0; poolY < poolDim.y; ++poolY) {
                    for (int poolX = 0; poolX < poolDim.x; ++poolX) {
                        int prevX = w * poolDim.x + poolX;
                        int prevY = h * poolDim.y + poolY;

                        if (prevX < previousLayer->width && prevY < previousLayer->height) {
                            int prevIdx = prevX + prevY * previousLayer->width + d * previousLayer->width * previousLayer->height;

                            if (previousLayer->maps[prevIdx] == currentLayer->maps[currIdx]) {
                                previousGradient[prevIdx] = gradient[currIdx];
                                foundMatch = true;
                                break;
                            }
                        }
                    }

                    if (foundMatch) {
                        break;  // Break out of the poolY loop
                    }
                }
            }
        }
    }

    delete[] gradient;
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

    int blockSize = 256;
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

void FullyConnectedLayerBackward(Layer* currentLayer, Group* weights, Layer* biases, Layer* previousLayer,
                                 float*& gradient, float learningRate) {
    float* previousGradient = new float[previousLayer->width * previousLayer->height * previousLayer->depth]();

    int currentLayerSize = currentLayer->width * currentLayer->height * currentLayer->depth;
    int previousLayerSize = previousLayer->width * previousLayer->height * previousLayer->depth;

    for (int i = 0; i < currentLayerSize; ++i) {
        for (int j = 0; j < previousLayerSize; ++j) {
            // Update weights using the gradient descent update rule
            weights->filters[0].maps[j + i * previousLayer->width] -= learningRate * gradient[i] * previousLayer->maps[j];
        }
        // Update biases using the gradient descent update rule
        biases->maps[i] -= learningRate * gradient[i];
    }

    for (int i = 0; i < currentLayerSize; ++i) {
        for (int j = 0; j < previousLayerSize; ++j) {
            // Calculate gradients with respect to the inputs of the previous layer
            previousGradient[j] += gradient[i] * weights->filters[0].maps[j + i * currentLayer->width];
        }
    }

    delete[] gradient;
    gradient = previousGradient;
}

float MSELossFunction(const float* input, const float* predictedResult, int size) {
    float loss = 0;

    for (int i = 0; i < size; ++i) {
        float diff = predictedResult[i] - input[i];
        loss += diff * diff;
    }

    loss /= (float)size;

    return loss;
}
