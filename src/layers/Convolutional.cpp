#include "Convolutional.h"

#include <iostream>

#include <cstdlib>
#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"
//#include "xil_io.h"
//#include "xparameters.h"
#ifdef ZEDBOARD
    #include "xllfifo_hw.h"
#endif


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
        //fp32 curBias = biasData.get<fp32>(fill);
        ui8 curBias = biasData.get<ui8>(fill);
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
                            ui8 quantized_inputValue= round(scaleValueInputs*inputValue) + zero_points;
                            //float filterValue = weightData.get<fp32>(fInd);
                            ui8 quantized_filterValue = weightData.get<ui8>(fInd);

                            // Multiply and accumulate
                            outputValue += quantized_inputValue * quantized_filterValue;

                        }
                    }
                }
                
                int oInd = out_h * (outputWidth * numFilters) + out_w * numFilters + fill;
                fp32 out = (outputValue - (zero_points*sumOfWeightData.get<fp32>(fill)))/(scaleValueInputs*scaleValueWeights);
                outputData.get<fp32>(oInd) = std::max(0.0f, out + curBias);


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

void ConvolutionalLayer::computeAccelerated(const LayerData& dataIn) const {

    LayerData& outputData = getOutputData();

    const LayerParams& inputParams  = dataIn.getParams();
    const LayerParams& outputParams = outputData.getParams();

    int width = inputParams.dims[1];
    int depth = inputParams.dims[2];

    int outputHeight = outputParams.dims[0];
    int outputWidth = outputParams.dims[1];
    int numFilters = outputParams.dims[2];  // Equivlent to the depth of the output map

    int filterHeight = weightParam.dims[0];
    int filterWidth  = weightParam.dims[1];

    #ifdef zedboard
    
    for(int out_h = 0; out_h < outputHeight; out_h++)
    {
        for(int out_w = 0; out_w < outputWidth; out_w++)
        {    
            for(int fill = 0; fill < numFilters; fill++)
            {
                for(int fill_h = 0; fill_h < filterHeight; fill_h++)
                {
                    for(int fill_w = 0; fill_w < filterWidth; fill_w++)       
                    {
                        for(int dep = 0; dep < depth; dep++)
                        {
                            // Grab the weight at the corresponding filter position
                            // MAC with the input at the coreesponding postion?
                            
                            int iInd = (out_h + fill_h) * (width * depth) + (out_w + fill_w) * depth + dep;
                            int fInd = fill_h * (filterWidth * depth * numFilters) + fill_w * (depth * numFilters) + dep * numFilters + fill;
                            
                            float inputValue = dataIn.get<fp32>(iInd);
                            int8_t quantized_value = zero_points + round(inputValue * scaleValueInputs);

                            // Assuming that your inputs are 8 bit integers in ML
                            int8_t filterValue = weightData.get<int8_t>(fInd);

                            uint16_t DataValue= ((uint8_t)filterValue << 8) | (uint8_t)quantized_value;
                            Xil_Out32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_TDFD_OFFSET, (uint32_t)DataValue);

                        }
                        
                    }
                }

                while (true) {
                    // Then read how many words are available to us right now.
                    // Bit 31 = 1 when this is all the words in the current packet
                    // Bit 31 = 0 when this is how many words are available,
                    // but no TLAST has been sent to us yet

                    uint32_t read_len = Xil_In32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_RLF_OFFSET);
                    // Read out every word we have access to right now
                    for (int i = 0; i < (read_len & 0x7FFFFFFFUL); i+=4) {
                        int32_t in_data = (int32_t)Xil_In32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_RDFD_OFFSET);
                        // Process in_data here from your MAC unit
                        in_data += biasData.get<fp32>(fill);
                        fp32 out = ((in_data * 1.0) -  (zero_points * sumOfWeightData.get<fp32>(out_h)))/(scaleValueInputs * scaleValueWeights);
                        int outIndex = out_h * (outputWidth * numFilters) + out_w * numFilters + fill;
                        
                        // Apply ReLU
                        outputData.get<fp32>(outIndex) = out < 0 ? 0 : out;
                    }
                    if (!(read_len & (1 << 31))) {
                        break; // This is all the data in this packet, done
                    }
                    // There is more in this data packet, wait for more to come in

                }
            }
        }
    }

    #endif

}  // namespace ML
