
#include <iostream>
#include <sstream>
#include <vector>

#include "Config.h"
#include "Model.h"
#include "Types.h"
#include "Utils.h"
#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/DenseLast.h"
#include "layers/Layer.h"
#include "layers/MaxPooling.h"
#include "layers/Softmax.h"
#include "layers/Flatten.h"

#ifdef ZEDBOARD
#include <file_transfer/file_transfer.h>
#endif

namespace ML {

// Build our ML toy model
Model buildToyModel(const Path modelPath) {
    Model model;
    logInfo("--- Building Toy Model ---");

    // --- Conv 1: L1 ---
    // Input shape: 64x64x3
    // Output shape: 60x60x32

    // You can pick how you want to implement your layers, both are allowed:

    // LayerParams conv1_inDataParam(sizeof(fp32), {64, 64, 3});
    // LayerParams conv1_outDataParam(sizeof(fp32), {60, 60, 32});
    // LayerParams conv1_weightParam(sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin");
    // LayerParams conv1_biasParam(sizeof(fp32), {32}, modelPath / "conv1_biases.bin");
    // auto conv1 = new ConvolutionalLayer(conv1_inDataParam, conv1_outDataParam, conv1_weightParam, conv1_biasParam);

    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {64, 64, 3}},                                    // Input Data
        LayerParams{sizeof(fp32), {60, 60, 32}},                                   // Output Data
        LayerParams{sizeof(int8_t), {5, 5, 3, 32}, modelPath / "weight_quant.bin"}, // Weights
        LayerParams{sizeof(fp32), {32}, modelPath / "bias_quant.bin"},            // Bias
        (1-.47324), 0.0, 0.30287936,
        LayerParams{sizeof(fp32), {32}, modelPath / "weight_sum_0.bin"}
    );

    // model.addLayer<ConvolutionalLayer>(
    //     LayerParams{sizeof(fp32), {64, 64, 3}},                                    // Input Data
    //     LayerParams{sizeof(fp32), {60, 60, 32}},                                   // Output Data
    //     LayerParams{sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin"}, // Weights
    //     LayerParams{sizeof(fp32), {32}, modelPath / "conv1_biases.bin"},            // Bias
    //     0.0, 0.0, 0.0
    // );

    // // --- Conv 2: L2 ---
    // // Input shape: 60x60x32
    // // Output shape: 56x56x32

    // model.addLayer<ConvolutionalLayer>(
    //     LayerParams{sizeof(fp32), {60, 60, 32}},                                    // Input Data
    //     LayerParams{sizeof(fp32), {56, 56, 32}},                                   // Output Data
    //     LayerParams{sizeof(ui8), {5, 5, 32, 32}, modelPath / "conv2d_1_quantized_weights.bin"}, // Weights
    //     LayerParams{sizeof(ui32), {32}, modelPath / "conv2d_1_quantized_biases.bin"},            // Bias
    //     (1.33663-.039343), 0.0, 0.48677802
    // );

    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {60, 60, 32}},                                    // Input Data
        LayerParams{sizeof(fp32), {56, 56, 32}},                                   // Output Data
        LayerParams{sizeof(int8_t), {5, 5, 32, 32}, modelPath / "weight_quant_1.bin"}, // Weights
        //LayerParams{sizeof(int8_t), {5, 5, 32, 32}, modelPath / "conv2d_1_quantized_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {32}, modelPath / "bias_quant_1.bin"},            // Bias
        (1.33663-.039343), 0.0, 0.48677802,
        LayerParams{sizeof(fp32), {32}, modelPath / "weight_sum_1.bin"}
    );

    // // --- MPL 1: L3 ---
    // // Input shape: 56x56x32
    // // Output shape: 28x28x32

    model.addLayer<MaxPoolingLayer>(
        LayerParams{sizeof(fp32), {56, 56, 32}},                                    // Input Data
        LayerParams{sizeof(fp32), {28, 28, 32}}                                // Output Data
    );

    // // --- Conv 3: L4 ---
    // // Input shape: 28x28x32
    // // Output shape: 26x26x64

    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {28, 28, 32}},                                    // Input Data
        LayerParams{sizeof(fp32), {26, 26, 64}},                                   // Output Data
        LayerParams{sizeof(int8_t), {3, 3, 32, 64}, modelPath / "weight_quant_3.bin"}, // Weights
        LayerParams{sizeof(fp32), {64}, modelPath / "bias_quant_3.bin"},            // Bias
        2.5, 0.0, 0.692378,
        LayerParams{sizeof(fp32), {32}, modelPath / "weight_sum_3.bin"}
    );

    // // --- Conv 4: L5 ---
    // // Input shape: 26x26x64
    // // Output shape: 24x24x64

    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {26, 26, 64}},                                    // Input Data
        LayerParams{sizeof(fp32), {24, 24, 64}},                                   // Output Data
        LayerParams{sizeof(int8_t), {3, 3, 64, 64}, modelPath / "weight_quant_4.bin"}, // Weights
        LayerParams{sizeof(fp32), {64}, modelPath / "bias_quant_4.bin"},            // Bias
        1.7, 0.0, 0.541547,
        LayerParams{sizeof(fp32), {32}, modelPath / "weight_sum_4.bin"}
    );

    // // --- MPL 2: L6 ---
    // // Input shape: 24x24x64
    // // Output shape: 12x12x64

    model.addLayer<MaxPoolingLayer>(
        LayerParams{sizeof(fp32), {24, 24, 64}},                                    // Input Data
        LayerParams{sizeof(fp32), {12, 12, 64}}                                 // Output Data
    );

    // // --- Conv 5: L7 ---
    // // Input shape: 12x12x64
    // // Output shape: 10x10x64

    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {12, 12, 64}},                                    // Input Data
        LayerParams{sizeof(fp32), {10, 10, 64}},                                   // Output Data
        LayerParams{sizeof(int8_t), {3, 3, 64, 64}, modelPath / "weight_quant_7.bin"}, // Weights
        LayerParams{sizeof(fp32), {64}, modelPath / "bias_quant_7.bin"},            // Bias
        1.87, 0.0, 0.536679,
        LayerParams{sizeof(fp32), {32}, modelPath / "weight_sum_7.bin"}
    );

    // // --- Conv 6: L8 ---
    // // Input shape: 10x10x64
    // // Output shape: 8x8x128

    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {10, 10, 64}},                                    // Input Data
        LayerParams{sizeof(fp32), {8, 8, 128}},                                   // Output Data
        LayerParams{sizeof(int8_t), {3, 3, 64, 128}, modelPath / "weight_quant_8.bin"}, // Weights
        LayerParams{sizeof(fp32), {128}, modelPath / "bias_quant_8.bin"},            // Bias
        2.11, 0.0, 0.510655,
        LayerParams{sizeof(fp32), {32}, modelPath / "weight_sum_8.bin"}
    );

    // // --- MPL 3: L9 ---
    // // Input shape: 8x8x128
    // // Output shape: 4x4x128

    model.addLayer<MaxPoolingLayer>(
        LayerParams{sizeof(fp32), {8, 8, 128}},                                    // Input Data
        LayerParams{sizeof(fp32), {4, 4, 128}}                                  // Output Data
    );

    // // --- Flatten 1: L10 ---
    // // Input shape: 4x4x128
    // // Output shape: 2048

    model.addLayer<FlattenLayer>(
        LayerParams{sizeof(fp32), {4, 4, 128}},                                    // Input Data
        LayerParams{sizeof(fp32), {2048}}                                  // Output Data
    );


    // // --- Dense 1: L11 ---
    // // Input shape: 2048
    // // Output shape: 256

    model.addLayer<DenseLayer>(
        LayerParams{sizeof(fp32), {2048}},                                    // Input Data
        LayerParams{sizeof(fp32), {256}},                                   // Output Data
        LayerParams{sizeof(int8_t), {2048*256}, modelPath / "dense_weight_quant.bin"}, // Weights
        LayerParams{sizeof(fp32), {256}, modelPath / "dense_bias_quant.bin"},            // Bias
        3.2, 0.0, 0.557585
    );

    // // --- Dense 2: L12 ---
    // // Input shape: 256
    // // Output shape: 200

    model.addLayer<DenseLastLayer>(
        LayerParams{sizeof(fp32), {256}},                                    // Input Data
        LayerParams{sizeof(fp32), {200}},                                   // Output Data
        LayerParams{sizeof(int8_t), {256*200}, modelPath / "dense_weight_1_quant.bin"}, // Weights
        LayerParams{sizeof(fp32), {200}, modelPath / "dense_bias_1_quant.bin"},            // Bias
        6.63967, 0.0, 1.32412
    );

    // // --- Softmax 1: L13 ---
    // // Input shape: 200
    // // Output shape: 200

    model.addLayer<SoftMaxLayer>(
        LayerParams{sizeof(fp32), {200}},                                    // Input Data
        LayerParams{sizeof(fp32), {200}}                                  // Output Data
    );

    return model;
}

void runBasicTest(const Model& model, const Path& basePath) {
    logInfo("--- Running Basic Test ---");

    // Load an image
    LayerData img = {{sizeof(fp32), {64, 64, 3}, "./data/image_0.bin"}};
    img.loadData();

    // Compare images
    std::cout << "Comparing image 0 to itself (max error): " << img.compare<fp32>(img) << std::endl
              << "Comparing image 0 to itself (T/F within epsilon " << ML::Config::EPSILON << "): " << std::boolalpha
              << img.compareWithin<fp32>(img, ML::Config::EPSILON) << std::endl;

    // Test again with a modified copy
    std::cout << "\nChange a value by 0.1 and compare again" << std::endl;
    
    LayerData imgCopy = img;
    imgCopy.get<fp32>(0) += 0.1;

    // Compare images
    img.compareWithinPrint<fp32>(imgCopy);

    // Test again with a modified copy
    log("Change a value by 0.1 and compare again...");
    imgCopy.get<fp32>(0) += 0.1;

    // Compare Images
    img.compareWithinPrint<fp32>(imgCopy);
}

void runLayerTest(const std::size_t layerNum, const Model& model, const Path& basePath) {
    // Load an image
    logInfo("--- Running Layer Test ---");
    //dimVec inDims = {64, 64, 3};
    dimVec inDims = model[layerNum].getInputParams().dims;

    char* inPath = (char*)malloc(50*sizeof(char));
    if(layerNum == 0)
    {
        sprintf(inPath, "image_2.bin");
    }
    else
    {
        sprintf(inPath, "image_2_data/layer_%d_output.bin", (int)layerNum-1);
    }

    // Construct a LayerData object from a LayerParams one
    LayerData img({sizeof(fp32), inDims, basePath / inPath});
    img.loadData();

    Timer timer("Layer Inference");

    // Run inference on the model
    
    timer.start();
    const LayerData output = model.inferenceLayer(img, layerNum, Layer::InfType::NAIVE);
    timer.stop();

    // Compare the output
    // Construct a LayerData object from a LayerParams one
    dimVec outDims = model[layerNum].getOutputParams().dims;
    char* outPath = (char*)malloc(50*sizeof(char));
    sprintf(outPath, "image_2_data/layer_%d_output.bin", (int)layerNum);
    LayerData expected({sizeof(fp32), outDims, basePath / outPath});
    expected.loadData();
    output.compareWithinPrint<fp32>(expected);

    free(inPath);
    free(outPath);

}

void runInferenceTest(const Model& model, const Path& basePath) {
    // Load an image
    logInfo("--- Running Inference Test ---");
    dimVec inDims = {64, 64, 3};

    // Construct a LayerData object from a LayerParams one
    LayerData img({sizeof(fp32), inDims, basePath / "image_2.bin"});
    img.loadData();

    Timer timer("Full Inference");

    // Run inference on the model
    timer.start();
    const LayerData output = model.inference(img, Layer::InfType::NAIVE);
    timer.stop();

    // Compare the output
    // Construct a LayerData object from a LayerParams one
    dimVec outDims = model.getOutputLayer().getOutputParams().dims;
    LayerData expected({sizeof(fp32), outDims, basePath / "image_2_data" / "layer_11_output.bin"});
    expected.loadData();
    output.compareWithinPrint<fp32>(expected);
}

void runTests() {
    // Base input data path (determined from current directory of where you are running the command)
    Path basePath("data");  // May need to be altered for zedboards loading from SD Cards

    // Build the model and allocate the buffers
    Model model = buildToyModel(basePath / "model");
    model.allocLayers();

    // Run some framework tests as an example of loading data
    runBasicTest(model, basePath);

    for(int i = 0; i < 13; i++)
    {
        runLayerTest(i, model, basePath);
    }

    // Run an end-to-end inference test
    //runInferenceTest(model, basePath);

    // Clean up
    model.freeLayers();
    std::cout << "\n\n----- ML::runTests() COMPLETE -----\n";
}

} // namespace ML

#ifdef ZEDBOARD
extern "C"
int main() {
    try {
        static FATFS fatfs;
        if (f_mount(&fatfs, "/", 1) != FR_OK) {
            throw std::runtime_error("Failed to mount SD card. Is it plugged in?");
        }
        ML::runTests();
    } catch (const std::exception& e) {
        std::cerr << "\n\n----- EXCEPTION THROWN -----\n" << e.what() << '\n';
    }
    std::cout << "\n\n----- STARTING FILE TRANSFER SERVER -----\n";
    FileServer::start_file_transfer_server();
}
#else
int main() {
    ML::runTests();
}
#endif