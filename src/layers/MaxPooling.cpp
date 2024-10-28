#include "MaxPooling.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the MaxPooling for the layer data
void MaxPoolingLayer::computeNaive(const LayerData& dataIn) const {

    LayerData& outputData = getOutputData();

    const LayerParams& inputParams  = dataIn.getParams();
    const LayerParams& outputParams = outputData.getParams();

    //60, 60, 32, example input Params
    int height = inputParams.dims[0];
    int width = inputParams.dims[1];
    int depth = inputParams.dims[2];

    int outputHeight = outputParams.dims[0];
    int outputWidth = outputParams.dims[1];

    int poolHeight = height / outputHeight;
    int poolWidth = width / outputWidth;

    for(int dep = 0; dep < depth; dep++)
    {
        for(int out_h = 0; out_h < outputHeight; out_h++)
        {
            for(int out_w = 0; out_w < outputWidth; out_w++)
            {
                // Initalize to minimum
                fp32 max = std::numeric_limits<float>::min();;
                for(int pool_h = 0; pool_h < poolHeight; pool_h++)
                {
                    for(int pool_w = 0; pool_w < poolWidth; pool_w++)
                    {
                        int inputH = (out_h * 2) + pool_h;
                        int inputW = (out_w * 2) + pool_w;

                        //float val = dataIn.get<fp32>(dep * height * width + inputH * width + inputW);
                        float val = dataIn.get<fp32>(depth * inputH * width + inputW * depth + dep);

                        if(val > max)
                        {
                            max = val;
                        }

                    }
                }
                // Check to make sure you assumption about how the indexs are organized work i.e. what is .get(1) is that right down deeper? 
                //getOutputData().get<fp32>(dep * outputHeight * outputWidth + out_h * outputHeight + out_w) = max;
                getOutputData().get<fp32>(out_h * depth * outputWidth + out_w * depth + dep) = max;
            }   
        }
    }

}

// Compute the convolution using threads
void MaxPoolingLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using a tiled approach
void MaxPoolingLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

// Compute the convolution using SIMD
void MaxPoolingLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // Don't have to do, future lab
}

void MaxPoolingLayer::computeAccelerated(const LayerData& dataIn) const {
    LayerData& outputData = getOutputData();

    const LayerParams& inputParams  = dataIn.getParams();
    const LayerParams& outputParams = outputData.getParams();

    //60, 60, 32, example input Params
    int height = inputParams.dims[0];
    int width = inputParams.dims[1];
    int depth = inputParams.dims[2];

    int outputHeight = outputParams.dims[0];
    int outputWidth = outputParams.dims[1];

    int poolHeight = height / outputHeight;
    int poolWidth = width / outputWidth;

    for(int dep = 0; dep < depth; dep++)
    {
        for(int out_h = 0; out_h < outputHeight; out_h++)
        {
            for(int out_w = 0; out_w < outputWidth; out_w++)
            {
                // Initalize to minimum
                fp32 max = std::numeric_limits<float>::min();;
                for(int pool_h = 0; pool_h < poolHeight; pool_h++)
                {
                    for(int pool_w = 0; pool_w < poolWidth; pool_w++)
                    {
                        int inputH = (out_h * 2) + pool_h;
                        int inputW = (out_w * 2) + pool_w;

                        //float val = dataIn.get<fp32>(dep * height * width + inputH * width + inputW);
                        float val = dataIn.get<fp32>(depth * inputH * width + inputW * depth + dep);

                        if(val > max)
                        {
                            max = val;
                        }

                    }
                }
                // Check to make sure you assumption about how the indexs are organized work i.e. what is .get(1) is that right down deeper? 
                //getOutputData().get<fp32>(dep * outputHeight * outputWidth + out_h * outputHeight + out_w) = max;
                getOutputData().get<fp32>(out_h * depth * outputWidth + out_w * depth + dep) = max;
            }   
        }
    }
}

}  // namespace ML
