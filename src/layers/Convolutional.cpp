#include "Convolutional.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// // // Compute the convultion for the layer data
void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const 
{
    
    //Step 1: get the 3d array at the spot of the filter
    //Step 2: form the filter
    //Step 3: dot product them
    //Stpe 4: put in the corresponding output map at the corresponding position
    //Step 5: Collect all output maps and stack them
    //Step 6: Put them through activation function

    LayerData& outputData = getOutputData();

    const LayerParams& inputParams  = dataIn.getParams();
    const LayerParams& outputParams = outputData.getParams();

    // 60, 60, 32, example input Params
    //int height = inputParams.dims[0];
    int width = inputParams.dims[1];
    int depth = inputParams.dims[2];

    int outputHeight = outputParams.dims[0];
    int outputWidth = outputParams.dims[1];
    int numFilters = outputParams.dims[2];  // Equivlent to the depth of the output map

    int filterHeight = weightParam.dims[0];
    int filterWidth  = weightParam.dims[1];

    for(int fill = 0; fill < numFilters; fill++)
    {
        fp32 curBias = biasData.get<fp32>(fill);
        for(int out_h = 0; out_h < outputHeight; out_h++)
        {
            for(int out_w = 0; out_w < outputWidth; out_w++)
            {
                fp32 outputValue = 0.0;
                for(int dep = 0; dep < depth; dep++)
                {
                    for(int fill_h = 0; fill_h < filterHeight; fill_h++)
                    {
                        for(int fill_w = 0; fill_w < filterWidth; fill_w++)       
                        {
                            // Grab the weight at the corresponding filter position
                            // MAC with the input at the coreesponding postion?
                            
                            int iInd = (out_h + fill_h) * (width * depth) + (out_w + fill_w) * depth + dep;
                            int fInd = fill_h * (filterWidth * depth * numFilters) + fill_w * (depth * numFilters) + dep * numFilters + fill;
                            float inputValue = dataIn.get<fp32>(iInd);
                            float filterValue = weightData.get<fp32>(fInd);


                            // Multiply and accumulate
                            outputValue += inputValue * filterValue;

                        }
                    }
                }
                
                int oInd = out_h * (outputWidth * numFilters) + out_w * numFilters + fill;
                outputData.get<fp32>(oInd) = std::max(0.0f, outputValue + curBias);


            }
        }

    }
}


// Compute the convolution using threads
void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using a tiled approach
void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using SIMD
void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

}  // namespace ML
