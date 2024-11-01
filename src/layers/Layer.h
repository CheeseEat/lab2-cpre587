#pragma once

#include <vector>
#include <cstring>
#include <memory>

#include "../Config.h"
#include "../Utils.h"
#include "../Types.h"

namespace ML {

// Layer Parameter structure
class LayerParams {
   public:
    LayerParams(const std::size_t elementSize, const std::vector<std::size_t> dims) : LayerParams(elementSize, dims, "") {}
    LayerParams(const std::size_t elementSize, const std::vector<std::size_t> dims, const Path filePath)
        : elementSize(elementSize), dims(dims), filePath(filePath) {}

    bool isCompatible(const LayerParams& params) const;

    inline size_t flat_count() const {
        size_t size = 1;
        for (auto dim : dims) {
            size *= dim;
        }
        return size;
    }

    inline size_t byte_size() const {
        return flat_count() * elementSize;
    }

   public:
    const std::size_t elementSize;
    const std::vector<std::size_t> dims;
    const Path filePath;
};

// Output data container of a layer inference
class LayerData {
   public:
    inline LayerData(const LayerParams& params) : params(params) {}

    inline LayerData(const LayerData& other) : params(other.params) {
        allocData();
        std::memcpy(data.get(), other.data.get(), params.byte_size());
    }

    inline bool isAlloced() const { return data != nullptr; }
    inline const LayerParams& getParams() const { return params; }

    // Get the data pointer and cast it
    template <typename T> T& get(unsigned int flat_index) {
        return ((T*)data.get())[flat_index];
    }

    template <typename T> T get(unsigned int flat_index) const {
        return ((T*)data.get())[flat_index];
    }

    // Allocate data values
    inline void allocData() {
        if (data) return;

        // std::cout << "Allocating " << params.byte_size() << " with alignment " << params.elementSize << '\n';
        data.reset((char*)(new ui64[(params.byte_size() + 7)/8])); // Assume elementSize <= sizeof(u64) for alignment
        // std::cout << "Success\n";
    }

    // Load data values
    inline void loadData(Path filePath = "");
    inline void saveData(Path filePath = "");

    // Clean up data values
    inline void freeData() {
        if (!data) return;
        data.reset();
    }

    // Get the max difference between two Layer Data arrays
    template <typename T> float compare(const LayerData& other) const;

    // Compare within an Epsilon to ensure layer datas are similar within reason
    template <typename T, typename T_EP = float> bool compareWithin(const LayerData& other, const T_EP epsilon = Config::EPSILON) const;

    // Compare within an Epsilon to ensure layer datas are similar within reason
    template <typename T, typename T_EP = float> bool compareWithinPrint(const LayerData& other, const T_EP epsilon = Config::EPSILON) const;

   private:
    LayerParams params;
    std::unique_ptr<char[]> data;
};

// Base class all layers extend from
class Layer {
   public:
    // Inference Type
    enum class InfType { NAIVE, THREADED, TILED, SIMD, ACCELERATED };

    // Layer Type
    enum class LayerType { NONE, CONVOLUTIONAL, DENSE, SOFTMAX, MAX_POOLING, FLATTEN };

   public:
    // Contructors
    Layer(const LayerParams inParams, const LayerParams outParams, LayerType lType)
        : inParams(inParams), outParams(outParams), outData(outParams), lType(lType) {}
    virtual ~Layer() {}


    // Getter Functions
    const LayerParams& getInputParams() const { return inParams; }
    const LayerParams& getOutputParams() const { return outParams; }
    LayerData& getOutputData() const { return outData; }
    LayerType getLType() const { return lType; }
    bool isOutputBufferAlloced() const { return outData.isAlloced(); }
    bool checkDataInputCompatibility(const LayerData& data) const;

    // Abstract/Virtual Functions
    virtual void allocLayer() {
        outData.allocData();
    }

    virtual void freeLayer() {
        outData.freeData();
    }

    virtual void computeNaive(const LayerData& dataIn) const = 0;
    virtual void computeThreaded(const LayerData& dataIn) const = 0;
    virtual void computeTiled(const LayerData& dataIn) const = 0;
    virtual void computeSIMD(const LayerData& dataIn) const = 0;
    virtual void computeAccelerated(const LayerData& dataIn) const = 0;

   private:
    LayerParams inParams;

    LayerParams outParams;
    mutable LayerData outData;

    LayerType lType;
};

// Load data values
inline void LayerData::loadData(Path filePath) {
    if (filePath.empty()) filePath = params.filePath;

    // Ensure a file path to load data from has been given
    if (filePath.empty()) throw std::runtime_error("No file path given for required layer data to load from");

    // If it has not already been allocated, allocate it
    allocData();

    // Open our file and check for issues
#ifdef ZEDBOARD
    FIL file;
    if (f_open(&file, params.filePath.c_str(), FA_OPEN_EXISTING | FA_READ) == FR_OK) { // Open our file on the SD card
#else
    std::ifstream file(params.filePath, std::ios::binary);  // Open our file
    if (file.is_open()) {
#endif
        std::cout << "Opened binary file " << params.filePath << std::endl;
    } else {
        throw std::runtime_error("Failed to open binary file: " + params.filePath);
    }

#ifdef ZEDBOARD
    UINT bytes_read = 0;
    if ((f_read(&file, data.get(), params.byte_size(), &bytes_read) != FR_OK) || (bytes_read != params.byte_size())) {
#else
    if (!file.read((char*)data.get(), params.byte_size())) {
#endif
        throw std::runtime_error("Failed to read file data");
    }

#ifdef ZEDBOARD
    f_close(&file);
#else
    // Close our file (ifstream deconstructor does this for us)
#endif
}


// Load data values
inline void LayerData::saveData(Path filePath) {
    if (filePath.empty()) filePath = params.filePath;
    
    // Ensure a file path to load data from has been given
    if (filePath.empty()) throw std::runtime_error("No file path given for required layer data to save to");

    // If it has not already been allocated, allocate it
    allocData();

    // Open our file and check for issues
#ifdef ZEDBOARD
    FIL file;
    if (f_open(&file, params.filePath.c_str(), FA_CREATE_ALWAYS | FA_WRITE) == FR_OK) { // Open our file on the SD card
#else
    std::ifstream file(params.filePath, std::ios::binary);  // Create and open our file
    if (file.is_open()) {
#endif
        std::cout << "Opened binary file " << params.filePath << std::endl;
    } else {
        throw std::runtime_error("Failed to open binary file: " + params.filePath);
    }

#ifdef ZEDBOARD
    UINT bytes_written = 0;
    if ((f_write(&file, data.get(), params.byte_size(), &bytes_written) != FR_OK) || (bytes_written != params.byte_size())) {
#else
    if (!file.read((char*)data.get(), params.byte_size())) {
#endif
        throw std::runtime_error("Failed to read file data");
    }

#ifdef ZEDBOARD
    f_close(&file);
#else
    // Close our file (ifstream deconstructor does this for us)
#endif
}

// Get the max difference between two Layer Data arrays
template <typename T> float LayerData::compare(const LayerData& other) const {
    LayerParams aParams = getParams();
    LayerParams bParams = other.getParams();

    // Warn if we are not comparing the same data type
    if (aParams.elementSize != bParams.elementSize) {
        throw std::runtime_error("Comparison between two LayerData arrays with different element size (and possibly data types) is not advised (" + std::to_string(aParams.elementSize)
                  + " and " + std::to_string(bParams.elementSize) + ")\n");
    }
    if (aParams.dims.size() != bParams.dims.size()) {
        throw std::runtime_error("LayerData arrays must have the same number of dimentions");
    }

    // Ensure each dimention size matches
    for (std::size_t i = 0; i < aParams.dims.size(); i++) {
        if (aParams.dims[i] != bParams.dims[i]) {
            throw std::runtime_error("LayerData arrays must have the same size dimentions to be compared");
        }
    }

    size_t flat_count = params.flat_count();
    float max_diff = 0;
    T* data1 = (T*)data.get();
    T* data2 = (T*)other.data.get();
    // Recurse as needed into each array
    for (std::size_t i = 0; i < flat_count; i++) {
        float curr_diff = fabsf(data1[i] - data2[i]);

        // Update our max difference if it is larger
        if (curr_diff > max_diff) {
            max_diff = curr_diff;
        }
    }

    return max_diff;
}

// Compare within an Epsilon to ensure layer datas are similar within reason
template <typename T, typename T_EP> bool LayerData::compareWithin(const LayerData& other, const T_EP epsilon) const {
    return epsilon > compare<T>(other);
}

template <typename T, typename T_EP> bool LayerData::compareWithinPrint(const LayerData& other, const T_EP epsilon) const {
    float max_diff = compare<T>(other);
    bool result = (epsilon > compare<T>(other));

    std::cout 
        << "Comparing images (max error): " 
        << (result ? "True" : "False")
        << " ("
        << max_diff
        << ")\n";
    
    return result;
}

}  // namespace ML
