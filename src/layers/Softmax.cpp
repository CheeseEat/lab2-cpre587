#include "Softmax.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---


void SoftMaxLayer::computeNaive(const LayerData& dataIn) const
{
    LayerData& outputData = getOutputData();
    const LayerParams& inputParams  = dataIn.getParams();
    const LayerParams& outputParams = outputData.getParams();

    int inputSize = inputParams.dims[0];
    int outputSize = outputParams.dims[0];

    fp32 inputValue = 0.0;

    // Find the maximum value in the input
    fp32 maxInputValue = dataIn.get<fp32>(0);
    for (int idx = 1; idx < inputSize; idx++) {
        inputValue = dataIn.get<fp32>(idx);
        if (inputValue > maxInputValue) {
            maxInputValue = inputValue;
        }
    }

    // Calculate sum of exponentials shifted by max value
    fp32 totalExpoInput = 0.0f;
    for (int idx = 0; idx < inputSize; idx++) {
        inputValue = dataIn.get<fp32>(idx);
        totalExpoInput += exp(inputValue - maxInputValue);
    }

    // Apply the softmax formula
    for (int idx = 0; idx < outputSize; idx++) {  
        inputValue = dataIn.get<fp32>(idx);
        outputData.get<fp32>(idx) = exp(inputValue - maxInputValue) / totalExpoInput;
    }
}


// Compute the convolution using threads
void SoftMaxLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using a tiled approach
void SoftMaxLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using SIMD
void SoftMaxLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

}  // namespace ML
