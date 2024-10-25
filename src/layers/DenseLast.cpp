#include "DenseLast.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---


void DenseLastLayer::computeNaive(const LayerData& dataIn) const {
    
    LayerData& outputData = getOutputData();

    const LayerParams& inputParams  = dataIn.getParams();
    const LayerParams& outputParams = outputData.getParams();

    // flattened (only 1 dimension)
    int inputDims = inputParams.dims[0];
    int outputDims = outputParams.dims[0];

    for(int outDims = 0; outDims < outputDims; outDims++)
    {
        fp32 output = 0.0;

        for(int inDims = 0; inDims < inputDims; inDims++)
        {   
            fp32 input = dataIn.get<fp32>(inDims);
            fp32 weight = weightData.get<fp32>(outDims + (inDims * outputDims));

            
            output += input * weight;
        }

        // Output = sum at all input channels (input * weight + bias)
        output += biasData.get<fp32>(outDims);
        
        // Applying activation function
        //outputData.get<fp32>(outDims) = std::max(0.0f, output); // ReLU activation
        outputData.get<fp32>(outDims) = output;
        
    }
}

// Compute the convolution using threads
void DenseLastLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using a tiled approach
void DenseLastLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using SIMD
void DenseLastLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// JOE: Might need to override alloc data?

}  // namespace ML
