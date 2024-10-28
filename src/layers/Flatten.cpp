#include "Flatten.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the Flatten for the layer data
void FlattenLayer::computeNaive(const LayerData& dataIn) const {
    
    // TODO: Your Code Here...
    

    LayerData& outputData = getOutputData();

    const LayerParams& inputParams  = dataIn.getParams();

    int height = inputParams.dims[0];
    int width = inputParams.dims[1];
    int depth = inputParams.dims[2];

    for(int ind = 0; ind < depth * height * width; ind++)
    {
        outputData.get<fp32>(ind) = dataIn.get<fp32>(ind);
    }

}

// Compute the convolution using threads
void FlattenLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using a tiled approach
void FlattenLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using SIMD
void FlattenLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

void FlattenLayer::computeAccelerated(const LayerData& dataIn) const {
    LayerData& outputData = getOutputData();

    const LayerParams& inputParams  = dataIn.getParams();

    int height = inputParams.dims[0];
    int width = inputParams.dims[1];
    int depth = inputParams.dims[2];

    for(int ind = 0; ind < depth * height * width; ind++)
    {
        outputData.get<fp32>(ind) = dataIn.get<fp32>(ind);
    }
}

}  // namespace ML
